//===- BOSCAMEAmeEmitter.cpp - Emit AME tiles for staged matmuls
//-----------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEAmeEmitter.h"
#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanner.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMETargetInfo.h"
#include "Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "Dialect/BOSCAME/Transforms/BOSCAMETileLowering.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::boscame;

namespace {

static constexpr StringLiteral kLoopAccumulatorAttr =
    "bosc_ame.loop_accumulator";
static constexpr StringLiteral kDirectAOperandAttr = "bosc_ame.direct_ddr_a";
static constexpr StringLiteral kDirectBOperandAttr = "bosc_ame.direct_ddr_b";
static constexpr StringLiteral kDirectCOperandAttr = "bosc_ame.direct_ddr_c";

LogicalResult verifyMatmul(linalg::MatmulOp op, const AmeCapability &ame) {
  if (!op.hasPureBufferSemantics() || op.getInputs().size() != 2 ||
      op.getOutputs().size() != 1)
    return op.emitError("BOSCAME AME requires a buffer-form matmul");
  auto a = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto b = dyn_cast<MemRefType>(op.getInputs()[1].getType());
  auto c = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  auto aElement = a ? dyn_cast<IntegerType>(a.getElementType()) : IntegerType();
  auto bElement = b ? dyn_cast<IntegerType>(b.getElementType()) : IntegerType();
  auto cElement = c ? dyn_cast<IntegerType>(c.getElementType()) : IntegerType();
  if (!a || !b || !c || a.getRank() != 2 || b.getRank() != 2 ||
      c.getRank() != 2 || !a.hasStaticShape() || !b.hasStaticShape() ||
      !c.hasStaticShape() || !aElement || !bElement || !cElement ||
      aElement.getWidth() != ame.a.elementBits ||
      bElement.getWidth() != ame.b.elementBits ||
      cElement.getWidth() != ame.accumulator.elementBits)
    return op.emitError() << "BOSCAME AME requires static rank-2 i"
                          << ame.a.elementBits << " x i" << ame.b.elementBits
                          << " -> i" << ame.accumulator.elementBits
                          << " buffers";
  if (op.getCast() != (ame.signedIntegerMultiply
                           ? linalg::TypeFn::cast_signed
                           : linalg::TypeFn::cast_unsigned))
    return op.emitError("BOSCAME AME requires signed integer extension");

  MLIRContext *context = op.getContext();
  AffineExpr m = getAffineDimExpr(0, context);
  AffineExpr n = getAffineDimExpr(1, context);
  AffineExpr k = getAffineDimExpr(2, context);
  SmallVector<AffineMap> expected = {AffineMap::get(3, 0, {m, k}, context),
                                     AffineMap::get(3, 0, {k, n}, context),
                                     AffineMap::get(3, 0, {m, n}, context)};
  if (op.getIndexingMapsArray() != expected)
    return op.emitError("BOSCAME AME requires standard matmul indexing maps; "
                        "describe physical B layout with memref strides");

  if (a.getDimSize(0) <= 0 || a.getDimSize(1) <= 0 || b.getDimSize(1) <= 0 ||
      a.getDimSize(0) != c.getDimSize(0) ||
      a.getDimSize(1) != b.getDimSize(0) || b.getDimSize(1) != c.getDimSize(1))
    return op.emitError("BOSCAME AME has inconsistent or empty matmul "
                        "dimensions");
  return success();
}

/// At the buffer-form boundary external DDR pointers are function block
/// arguments (possibly wrapped in view/cast operations).  Allocations are
/// compiler-managed temporaries and keep the established SPM accumulator path.
static bool isExternalDdrBuffer(Value value) {
  if (auto type = dyn_cast<MemRefType>(value.getType());
      type && spmdma::getSpmBank(type))
    return false;
  while (Operation *def = value.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
      value = transpose.getIn();
      continue;
    }
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      value = view.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      value = expand.getSrc();
      continue;
    }
    break;
  }
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner()->isEntryBlock() &&
         isa<func::FuncOp>(argument.getOwner()->getParentOp()) &&
         !spmdma::getSpmBank(argument.getType());
}

LogicalResult verifyStagedOperands(linalg::MatmulOp op,
                                   const TargetInfo &target) {
  constexpr TileRole roles[] = {TileRole::A, TileRole::B,
                                TileRole::Accumulator};
  const AmeRoleCapability *roleCapabilities[] = {&target.ame.a, &target.ame.b,
                                                 &target.ame.accumulator};
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip(op.getOperands(), roles))) {
    auto [operand, role] = pair;
    const AmeRoleCapability &roleCapability = *roleCapabilities[index];
    auto type = cast<MemRefType>(operand.getType());
    bool logicalDirect = role == TileRole::A ? op->hasAttr(kDirectAOperandAttr)
                         : role == TileRole::B
                             ? op->hasAttr(kDirectBOperandAttr)
                             : false;
    bool directOperand =
        !spmdma::getSpmBank(type) && roleCapability.supportsDirectDdr &&
        isExternalDdrBuffer(operand) &&
        ((role == TileRole::Accumulator && op->hasAttr(kDirectCOperandAttr)) ||
         logicalDirect);
    if (spmdma::getSpmBank(type) && !roleCapability.supportsSpm)
      return op.emitError(
          "BOSCAME AME target does not support this SPM tile role");
    if (!spmdma::getSpmBank(type) && !directOperand)
      return op.emitError("BOSCAME AME operands must be staged in SPM before "
                          "tile emission");
    if (type.getElementType().isInteger() &&
        cast<IntegerType>(type.getElementType()).getWidth() !=
            roleCapability.elementBits)
      return op.emitError("BOSCAME AME operand type does not match target "
                          "role capability");
    if (!hasAmeTileLayout(type, roleCapability.layout))
      return op.emitError()
             << "BOSCAME AME has an unsupported staged "
             << stringifyTileRole(role) << " layout; target requires "
             << stringifyAmeTileLayout(roleCapability.layout);
    auto layout = computeTileStorageLayout(type, role);
    if (!layout)
      return op.emitError()
             << "BOSCAME AME has an unsupported staged "
             << stringifyTileRole(role)
             << " layout; A/C require contiguous columns and B requires "
                "contiguous K in physical [N,K] storage";
    // The AME memory interface transfers complete cache lines.  A static
    // descriptor offset that is not line aligned cannot be repaired by the
    // lowering, and accepting it would make the direct path depend on an
    // undocumented partial-line transaction.  Dynamic base addresses remain
    // the caller's alignment contract (the typed ABI carries that guarantee).
    int64_t endpointAlignment = target.dma.endpointAlignmentBytes;
    if (directOperand &&
        (endpointAlignment <= 0 || layout->byteOffset % endpointAlignment != 0))
      return op.emitError(
          "BOSCAME direct-DDR operand requires a 64-byte-aligned "
          "descriptor offset");
    if (directOperand && layout->rowStrideBytes % endpointAlignment != 0)
      return op.emitError(
          "BOSCAME direct-DDR operand requires a 64-byte-aligned "
          "physical row stride");
    if (directOperand &&
        failed(spmdma::verifyEndpointPointerAlignment(
            op, operand, "BOSCAME direct-DDR", endpointAlignment)))
      return failure();
  }
  return success();
}

Value subview(OpBuilder &builder, Location loc, Value source, int64_t row,
              int64_t column, int64_t rows, int64_t columns) {
  SmallVector<OpFoldResult> offsets{builder.getIndexAttr(row),
                                    builder.getIndexAttr(column)};
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(rows),
                                  builder.getIndexAttr(columns)};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1),
                                    builder.getIndexAttr(1)};
  return memref::SubViewOp::create(builder, loc, source, offsets, sizes,
                                   strides);
}

Operation *emitLoopedMatmul(linalg::MatmulOp op,
                            const LinalgToBOSCAMEConfig &config,
                            const AmeMatmulTilePlan &tilePlan) {
  if (config.storePolicy != "direct-ddr" || !op->hasAttr(kLoopAccumulatorAttr))
    return nullptr;
  auto loop = op->getParentOfType<scf::ForOp>();
  if (!loop || op->getBlock() != loop.getBody())
    return nullptr;

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value a = op.getInputs()[0], b = op.getInputs()[1], c = op.getOutputs()[0];
  // Keep placement decisions alive across erasing the source linalg op below.
  // Reading an operation after `erase()` is invalid even when the attribute
  // value itself is immutable.
  const bool directC = op->hasAttr(kDirectCOperandAttr);
  auto aType = cast<MemRefType>(a.getType());
  auto bType = cast<MemRefType>(b.getType());
  int64_t blockM = tilePlan.tileM;
  int64_t blockN = tilePlan.tileN;
  int64_t blockK = tilePlan.tileK;

  SmallVector<Value> cTiles;
  SmallVector<Value> initialAccumulators;
  SmallVector<std::pair<int64_t, int64_t>> tileShape;
  builder.setInsertionPoint(loop);
  for (int64_t miIndex = 0; miIndex < tilePlan.mTileCount; ++miIndex) {
    int64_t mi = miIndex * blockM;
    int64_t rows = miIndex + 1 == tilePlan.mTileCount ? tilePlan.tailM : blockM;
    for (int64_t niIndex = 0; niIndex < tilePlan.nTileCount; ++niIndex) {
      int64_t ni = niIndex * blockN;
      int64_t columns =
          niIndex + 1 == tilePlan.nTileCount ? tilePlan.tailN : blockN;
      Value cTile = subview(builder, loc, c, mi, ni, rows, columns);
      auto accType = VectorType::get({rows, columns}, builder.getI32Type());
      auto loadAcc = TileLoadAccOp::create(builder, loc, accType, cTile);
      if (op->hasAttr(kDirectCOperandAttr))
        loadAcc->setAttr("bosc_ame.direct_ddr", builder.getUnitAttr());
      cTiles.push_back(cTile);
      initialAccumulators.push_back(loadAcc.getMatrix());
      tileShape.emplace_back(rows, columns);
    }
  }

  // Add one loop-carried AME accumulator for every output tile.  The original
  // loop body remains in place while the callback inserts A/B loads and MMA
  // operations immediately before the marked matmul.
  IRRewriter rewriter(op.getContext());
  unsigned originalResults = loop.getNumResults();
  auto newLoop = loop.replaceWithAdditionalYields(
      rewriter, initialAccumulators, /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &bld, Location callbackLoc,
          ArrayRef<BlockArgument> iterArgs) {
        SmallVector<Value> yielded;
        for (auto [index, shape] : llvm::enumerate(tileShape)) {
          int64_t rows = shape.first;
          int64_t columns = shape.second;
          bld.setInsertionPoint(op);
          int64_t mi = (index / tilePlan.nTileCount) * blockM;
          int64_t ni = (index % tilePlan.nTileCount) * blockN;
          Value accumulator = iterArgs[index];
          for (int64_t kiIndex = 0; kiIndex < tilePlan.kTileCount; ++kiIndex) {
            int64_t ki = kiIndex * blockK;
            int64_t inner =
                kiIndex + 1 == tilePlan.kTileCount ? tilePlan.tailK : blockK;
            Value aTile = subview(bld, callbackLoc, a, mi, ki, rows, inner);
            Value bTile = subview(bld, callbackLoc, b, ki, ni, inner, columns);
            auto loadA = TileLoadAOp::create(
                bld, callbackLoc,
                VectorType::get({rows, inner}, bld.getI8Type()), aTile);
            auto loadB = TileLoadBOp::create(
                bld, callbackLoc,
                VectorType::get({inner, columns}, bld.getI8Type()), bTile);
            Value lhs = loadA.getMatrix();
            Value rhs = loadB.getMatrix();
            if (op->hasAttr(kDirectAOperandAttr) &&
                !spmdma::getSpmBank(aType) &&
                !spmdma::getSpmBank(aTile.getType()) && isExternalDdrBuffer(a))
              loadA->setAttr("bosc_ame.direct_ddr", bld.getUnitAttr());
            if (op->hasAttr(kDirectBOperandAttr) &&
                !spmdma::getSpmBank(bType) &&
                !spmdma::getSpmBank(bTile.getType()) && isExternalDdrBuffer(b))
              loadB->setAttr("bosc_ame.direct_ddr", bld.getUnitAttr());
            auto mma = TileMmaOp::create(
                bld, callbackLoc,
                VectorType::get({rows, columns}, bld.getI32Type()), accumulator,
                lhs, rhs);
            mma->setAttr("bosc_ame.ordered", bld.getUnitAttr());
            accumulator = mma.getMatrix();
          }
          yielded.push_back(accumulator);
        }
        return yielded;
      });
  if (failed(newLoop))
    return nullptr;
  loop = cast<scf::ForOp>(*newLoop);

  // Stores after the loop complete the whole accumulator chain, including
  // every dynamic iteration. Retain the source until its metadata is moved.
  builder.setInsertionPointAfter(loop);
  Operation *completion = nullptr;
  for (auto [index, cTile] : llvm::enumerate(cTiles)) {
    auto storeAcc = TileStoreAccOp::create(
        builder, loc, loop.getResult(originalResults + index), cTile);
    if (directC)
      storeAcc->setAttr("bosc_ame.direct_ddr", builder.getUnitAttr());
    completion = storeAcc;
  }
  return completion;
}

Operation *emitMatmul(linalg::MatmulOp op, const LinalgToBOSCAMEConfig &config,
                      const AmeMatmulTilePlan &tilePlan) {
  if (Operation *completion = emitLoopedMatmul(op, config, tilePlan))
    return completion;
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value a = op.getInputs()[0], b = op.getInputs()[1], c = op.getOutputs()[0];
  auto aType = cast<MemRefType>(a.getType());
  auto bType = cast<MemRefType>(b.getType());
  auto cType = cast<MemRefType>(c.getType());
  int64_t blockM = tilePlan.tileM;
  int64_t blockN = tilePlan.tileN;
  int64_t blockK = tilePlan.tileK;
  Operation *completion = nullptr;

  for (int64_t miIndex = 0; miIndex < tilePlan.mTileCount; ++miIndex) {
    int64_t mi = miIndex * blockM;
    int64_t rows = miIndex + 1 == tilePlan.mTileCount ? tilePlan.tailM : blockM;
    for (int64_t niIndex = 0; niIndex < tilePlan.nTileCount; ++niIndex) {
      int64_t ni = niIndex * blockN;
      int64_t columns =
          niIndex + 1 == tilePlan.nTileCount ? tilePlan.tailN : blockN;
      Value cTile = subview(builder, loc, c, mi, ni, rows, columns);
      auto accType = VectorType::get({rows, columns}, builder.getI32Type());
      auto loadAcc = TileLoadAccOp::create(builder, loc, accType, cTile);
      if (config.storePolicy == "direct-ddr" &&
          op->hasAttr(kDirectCOperandAttr) && !spmdma::getSpmBank(cType) &&
          isExternalDdrBuffer(c))
        loadAcc->setAttr("bosc_ame.direct_ddr", builder.getUnitAttr());
      Value accumulator = loadAcc.getMatrix();
      // The planner completes every DMA before this point. Keep C live in
      // the AME accumulator for the whole K chain, with no runtime calls.
      for (int64_t kiIndex = 0; kiIndex < tilePlan.kTileCount; ++kiIndex) {
        int64_t ki = kiIndex * blockK;
        int64_t inner =
            kiIndex + 1 == tilePlan.kTileCount ? tilePlan.tailK : blockK;
        Value aTile = subview(builder, loc, a, mi, ki, rows, inner);
        Value bTile = subview(builder, loc, b, ki, ni, inner, columns);
        auto loadA = TileLoadAOp::create(
            builder, loc, VectorType::get({rows, inner}, builder.getI8Type()),
            aTile);
        auto loadB = TileLoadBOp::create(
            builder, loc,
            VectorType::get({inner, columns}, builder.getI8Type()), bTile);
        Value lhs = loadA.getMatrix();
        Value rhs = loadB.getMatrix();
        if (op->hasAttr(kDirectAOperandAttr) && !spmdma::getSpmBank(aType) &&
            !spmdma::getSpmBank(aTile.getType()) && isExternalDdrBuffer(a)) {
          loadA->setAttr("bosc_ame.direct_ddr", builder.getUnitAttr());
        }
        if (op->hasAttr(kDirectBOperandAttr) && !spmdma::getSpmBank(bType) &&
            !spmdma::getSpmBank(bTile.getType()) && isExternalDdrBuffer(b)) {
          loadB->setAttr("bosc_ame.direct_ddr", builder.getUnitAttr());
        }
        auto mma =
            TileMmaOp::create(builder, loc, accType, accumulator, lhs, rhs);
        // LLVM's AME MMA intrinsic is pure and does not model the mutable
        // tile configuration. Keep this serial reduction ordered through
        // its target lowering, including a shorter final K tile.
        mma->setAttr("bosc_ame.ordered", builder.getUnitAttr());
        accumulator = mma;
      }
      auto storeAcc = TileStoreAccOp::create(builder, loc, accumulator, cTile);
      if (config.storePolicy == "direct-ddr" &&
          op->hasAttr(kDirectCOperandAttr) && !spmdma::getSpmBank(cType) &&
          isExternalDdrBuffer(c))
        storeAcc->setAttr("bosc_ame.direct_ddr", builder.getUnitAttr());
      completion = storeAcc;
    }
  }
  return completion;
}

} // namespace

namespace buddy::boscame {

LogicalResult verifyBOSCAMEPlacedMatmul(Operation *operation,
                                        const TargetInfo &target) {
  auto matmul = dyn_cast<linalg::MatmulOp>(operation);
  if (!matmul || !matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme))
    return operation->emitError("resident matmul must target AME");
  if (failed(verifyMatmul(matmul, target.ame)))
    return failure();
  return verifyStagedOperands(matmul, target);
}

LogicalResult verifyBOSCAMEAmeConsumers(ModuleOp module,
                                        const DataflowPlan &plan) {
  LogicalResult result = success();
  module.walk([&](linalg::MatmulOp op) {
    if (failed(result) || !op->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      return;
    result = verifyMatmul(op, plan.target.ame);
  });
  return result;
}

LogicalResult lowerBOSCAMEAmeConsumers(ModuleOp module,
                                       const LinalgToBOSCAMEConfig &config,
                                       const DataflowPlan &plan,
                                       bool preserveComputeGraph) {
  SmallVector<linalg::MatmulOp> targets;
  module.walk([&](linalg::MatmulOp op) {
    if (op->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      targets.push_back(op);
  });
  if (targets.empty())
    return success();
  if (!config.enableAME)
    return targets.front().emitError("marked matmul requires enable-ame=true");
  if (config.bm <= 0 || config.bn <= 0 || config.bk <= 0)
    return module.emitError("BOSCAME tile sizes must be positive");
  auto target = module->getAttrOfType<StringAttr>(kAmeTargetAttrName);
  if ((!config.target.empty() && config.target != plan.target.name) ||
      (target && target.getValue() != plan.target.name))
    return module.emitError(
        "marked AME matmul requires bosc_ame.target=nr-fpga");

  // Diagnose all targets before mutating any of them.
  SmallVector<const AmeMatmulTilePlan *> tilePlans;
  for (linalg::MatmulOp op : targets) {
    if (failed(verifyMatmul(op, plan.target.ame)) ||
        failed(verifyStagedOperands(op, plan.target)))
      return failure();
    const AmeMatmulTilePlan *tilePlan =
        findAmeMatmulTilePlan(plan, op.getOperation());
    if (!tilePlan)
      return op.emitError("BOSCAME planner did not provide an AME tile plan");
    auto a = cast<MemRefType>(op.getInputs()[0].getType());
    auto b = cast<MemRefType>(op.getInputs()[1].getType());
    auto c = cast<MemRefType>(op.getOutputs()[0].getType());
    if (tilePlan->m != c.getDimSize(0) || tilePlan->n != c.getDimSize(1) ||
        tilePlan->k != a.getDimSize(1) || tilePlan->k != b.getDimSize(0) ||
        tilePlan->tileM <= 0 || tilePlan->tileN <= 0 || tilePlan->tileK <= 0 ||
        tilePlan->tileM > plan.target.ame.maxTileM ||
        tilePlan->tileN > plan.target.ame.maxTileN ||
        tilePlan->tileK > plan.target.ame.maxTileK ||
        tilePlan->mTileCount <= 0 || tilePlan->nTileCount <= 0 ||
        tilePlan->kTileCount <= 0 || tilePlan->tailM <= 0 ||
        tilePlan->tailM > tilePlan->tileM || tilePlan->tailN <= 0 ||
        tilePlan->tailN > tilePlan->tileN || tilePlan->tailK <= 0 ||
        tilePlan->tailK > tilePlan->tileK)
      return op.emitError("BOSCAME planner produced an invalid AME tile plan");
    tilePlans.push_back(tilePlan);
  }
  module->setAttr(kAmeTargetAttrName,
                  StringAttr::get(module.getContext(), kAmeTargetNrFpga));
  for (auto [index, op] : llvm::enumerate(targets)) {
    Operation *completion = emitMatmul(op, config, *tilePlans[index]);
    if (preserveComputeGraph)
      // M/N/K tiles are expanded statically. The final store is a unique
      // completion point for every tile, rather than one dynamic inner MMA.
      for (NamedAttribute attribute : op->getAttrs())
        if (attribute.getName().getValue().starts_with("bosc_ame.") &&
            attribute.getName().getValue() != lowering::kLowerToAme)
          completion->setAttr(attribute.getName(), attribute.getValue());
    op.erase();
    if (preserveComputeGraph && failed(verifyCompleteComputeGraphs(module)))
      return failure();
  }
  return success();
}

} // namespace buddy::boscame
