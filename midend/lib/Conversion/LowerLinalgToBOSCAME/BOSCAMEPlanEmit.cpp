//===- BOSCAMEPlanEmit.cpp - Mechanically expand verified planned IR
//-------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEOrderedMatmul.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanIR.h"
#include "Dialect/BOSCAME/BOSCAMEAttrs.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace ::buddy::boscame;

namespace {
DictionaryAttr plan(Operation *op, StringRef key) {
  return op->getAttrOfType<DictionaryAttr>(key);
}
DictionaryAttr nested(DictionaryAttr p, StringRef key) {
  return p.getAs<DictionaryAttr>(key);
}
int64_t number(DictionaryAttr p, StringRef key) {
  return p.getAs<IntegerAttr>(key).getInt();
}
StringRef text(DictionaryAttr p, StringRef key) {
  return p.getAs<StringAttr>(key).getValue();
}
int64_t elementBytes(Type type) { return type.getIntOrFloatBitWidth() / 8; }
Value index(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

/// This is descriptor reconstruction, not address analysis or route selection.
/// reinterpret_cast replaces the descriptor offset; subview adds its shift.
/// Keep dynamic SSA arithmetic visible and fold static descriptor arithmetic
/// so the final dialect verifier can prove the selected SPM range.
Value descriptorOffset(OpBuilder &b, Location loc, Value source) {
  if (auto constant = spmdma::getStaticDescriptorOffset(source))
    return index(b, loc, *constant);
  if (auto view = source.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto offset = view.getMixedOffsets().front();
    if (auto value = dyn_cast<Value>(offset))
      return value;
    return index(b, loc, cast<IntegerAttr>(cast<Attribute>(offset)).getInt());
  }
  if (auto view = source.getDefiningOp<memref::SubViewOp>()) {
    Value offset = descriptorOffset(b, loc, view.getSource());
    SmallVector<int64_t> strides;
    int64_t ignored;
    (void)cast<MemRefType>(view.getSource().getType())
        .getStridesAndOffset(strides, ignored);
    for (auto [part, stride] : llvm::zip(view.getMixedOffsets(), strides)) {
      Value term = dyn_cast<Value>(part);
      if (!term)
        term = index(b, loc, cast<IntegerAttr>(cast<Attribute>(part)).getInt());
      term = b.createOrFold<arith::MulIOp>(loc, term, index(b, loc, stride));
      offset = b.createOrFold<arith::AddIOp>(loc, offset, term);
    }
    return offset;
  }
  if (auto view = source.getDefiningOp<memref::CastOp>())
    return descriptorOffset(b, loc, view.getSource());
  if (auto view = source.getDefiningOp<memref::TransposeOp>())
    return descriptorOffset(b, loc, view.getIn());
  if (auto view = source.getDefiningOp<memref::CollapseShapeOp>())
    return descriptorOffset(b, loc, view.getSrc());
  if (auto view = source.getDefiningOp<memref::ExpandShapeOp>())
    return descriptorOffset(b, loc, view.getSrc());
  return memref::ExtractStridedMetadataOp::create(b, loc, source).getOffset();
}
Value endpoint(OpBuilder &b, Location loc, Value source, int64_t offsetBytes,
               int64_t countBytes) {
  auto sourceType = cast<MemRefType>(source.getType());
  int64_t width = elementBytes(sourceType.getElementType());
  Value offset = descriptorOffset(b, loc, source);
  if (offsetBytes)
    offset = b.createOrFold<arith::AddIOp>(loc, offset,
                                           index(b, loc, offsetBytes / width));
  auto type = MemRefType::get(
      {countBytes / width}, sourceType.getElementType(),
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {1}),
      sourceType.getMemorySpace());
  return memref::ReinterpretCastOp::create(
      b, loc, type, source, offset,
      ArrayRef<OpFoldResult>{b.getIndexAttr(countBytes / width)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}
void moveFacts(Operation *source, Operation *completion) {
  for (NamedAttribute attr : source->getAttrs())
    if (attr.getName().getValue().starts_with("contract."))
      completion->setAttr(attr.getName(), attr.getValue());
}
Value tileSubview(OpBuilder &b, Location loc, Value source, int64_t row,
                  int64_t column, int64_t rows, int64_t columns) {
  return memref::SubViewOp::create(
      b, loc, source,
      ArrayRef<OpFoldResult>{b.getIndexAttr(row), b.getIndexAttr(column)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(rows), b.getIndexAttr(columns)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
}
void direct(Operation *op, DictionaryAttr selection, OpBuilder &b) {
  if (text(selection, "level") == "ddr")
    op->setAttr("bosc_ame.direct_ddr", b.getUnitAttr());
}

void materializeBuffers(ModuleOp module) {
  SmallVector<Operation *> allocations;
  module.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      allocations.push_back(op);
  });
  for (Operation *op : allocations) {
    OpBuilder b(op);
    auto p = plan(op, "plan.buffer");
    auto type = cast<MemRefType>(op->getResult(0).getType());
    Value replacement;
    if (text(p, "level") == "spm") {
      int64_t bank = number(p, "bank");
      auto spmType =
          MemRefType::get(type.getShape(), type.getElementType(),
                          type.getLayout(), SPMAttr::get(b.getContext(), bank));
      replacement =
          SpmViewOp::create(b, op->getLoc(), spmType, b.getI32IntegerAttr(bank),
                            b.getI64IntegerAttr(number(p, "offset")));
    } else if (isa<memref::AllocOp>(op)) {
      replacement = memref::AllocaOp::create(b, op->getLoc(), type);
      if (Attribute alignment = op->getAttr("alignment"))
        replacement.getDefiningOp()->setAttr("alignment", alignment);
    } else {
      continue;
    }
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("contract."))
        replacement.getDefiningOp()->setAttr(attr.getName(), attr.getValue());
    SmallVector<Operation *> deallocations;
    SmallVector<Value> aliases{op->getResult(0)};
    DenseSet<Value> seen;
    while (!aliases.empty()) {
      Value alias = aliases.pop_back_val();
      if (!seen.insert(alias).second)
        continue;
      for (Operation *user : alias.getUsers()) {
        if (isa<memref::DeallocOp>(user))
          deallocations.push_back(user);
        else if (auto view = dyn_cast<ViewLikeOpInterface>(user);
                 view && view.getViewSource() == alias &&
                 user->getNumResults() == 1)
          aliases.push_back(user->getResult(0));
      }
    }
    for (Operation *dealloc : deallocations)
      dealloc->erase();
    op->getResult(0).replaceAllUsesWith(replacement);
    op->erase();
  }
  // All views preserve their shape and exact layout, changing only the
  // address space inherited from the already selected root placement.
  module.walk([&](Operation *op) {
    if (!isa<memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp,
             memref::ViewOp, memref::MemorySpaceCastOp, memref::TransposeOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(op))
      return;
    auto source = dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto result = dyn_cast<MemRefType>(op->getResult(0).getType());
    if (source && result)
      op->getResult(0).setType(
          MemRefType::get(result.getShape(), result.getElementType(),
                          result.getLayout(), source.getMemorySpace()));
  });
}

Operation *emitTransfer(memref::CopyOp copy) {
  OpBuilder b(copy);
  Location loc = copy.getLoc();
  auto p = plan(copy, "plan.transfer"), chunks = nested(p, "chunks");
  bool load = text(p, "from") == "ddr";
  int64_t sourceStride = number(chunks, load ? "ddr_stride" : "spm_stride");
  int64_t targetStride = number(chunks, load ? "spm_stride" : "ddr_stride");
  int64_t size = number(chunks, "bytes"), count = number(chunks, "count");
  int64_t channel = number(p, "channel");
  Operation *completion = nullptr;
  for (int64_t i = 0; i < count; ++i) {
    Value source = endpoint(b, loc, copy.getSource(), i * sourceStride, size);
    Value target = endpoint(b, loc, copy.getTarget(), i * targetStride, size);
    auto token = DmaStartOp::create(
        b, loc, DMATokenType::get(b.getContext(), channel), source, target,
        index(b, loc, size), b.getI32IntegerAttr(channel));
    completion = DmaWaitOp::create(b, loc, token.getToken());
  }
  return completion;
}

Operation *emitMatmul(linalg::MatmulOp matmul) {
  OpBuilder b(matmul);
  Location loc = matmul.getLoc();
  auto p = plan(matmul, "plan.route"), tile = nested(p, "tile"),
       sub = nested(p, "subtiles");
  int64_t m = number(tile, "m"), n = number(tile, "n"), k = number(tile, "k");
  Operation *completion = nullptr;
  for (int64_t ni = 0; ni < number(sub, "n"); ++ni) {
    Value c = tileSubview(b, loc, matmul.getOutputs()[0], 0, ni * n, m, n);
    auto accType = VectorType::get({m, n}, b.getI32Type());
    auto acc = TileLoadAccOp::create(b, loc, accType, c);
    direct(acc, nested(p, "accumulator"), b);
    Value accumulator = acc.getMatrix();
    for (int64_t ki = 0; ki < number(sub, "k"); ++ki) {
      Value a = tileSubview(b, loc, matmul.getInputs()[0], 0, ki * k, m, k);
      Value operandB =
          tileSubview(b, loc, matmul.getInputs()[1], ki * k, ni * n, k, n);
      auto lhs = TileLoadAOp::create(b, loc,
                                     VectorType::get({m, k}, b.getI8Type()), a);
      auto rhs = TileLoadBOp::create(
          b, loc, VectorType::get({k, n}, b.getI8Type()), operandB);
      direct(lhs, nested(p, "a"), b);
      direct(rhs, nested(p, "b"), b);
      auto mma = TileMmaOp::create(b, loc, accType, accumulator,
                                   lhs.getMatrix(), rhs.getMatrix());
      mma->setAttr("bosc_ame.ordered", b.getUnitAttr());
      accumulator = mma.getMatrix();
    }
    auto store = TileStoreAccOp::create(b, loc, accumulator, c);
    direct(store, nested(p, "output"), b);
    completion = store;
  }
  return completion;
}

Operation *emitFill(linalg::FillOp fill) {
  OpBuilder b(fill);
  Location loc = fill.getLoc();
  auto p = plan(fill, "plan.fill");
  Value output = fill.getOutputs()[0], scalar = fill.getInputs()[0];
  auto type = cast<MemRefType>(output.getType());
  if (text(p, "level") == "host")
    return memref::StoreOp::create(b, loc, scalar, output, ValueRange{});
  int64_t lanes =
      number(p, "vector_bytes") / elementBytes(type.getElementType());
  Value broadcast = vector::BroadcastOp::create(
      b, loc, VectorType::get({lanes}, type.getElementType()), scalar);
  int64_t rows = number(p, "rows");
  int64_t columns =
      number(p, "row_bytes") / elementBytes(type.getElementType());
  Value zero = index(b, loc, 0);
  auto outer =
      scf::ForOp::create(b, loc, zero, index(b, loc, rows), index(b, loc, 1));
  b.setInsertionPointToStart(outer.getBody());
  auto inner = scf::ForOp::create(b, loc, zero, index(b, loc, columns),
                                  index(b, loc, lanes));
  b.setInsertionPointToStart(inner.getBody());
  SmallVector<Value> indices;
  if (type.getRank() == 2)
    indices.push_back(outer.getInductionVar());
  indices.push_back(inner.getInductionVar());
  vector::StoreOp::create(b, loc, broadcast, output, indices);
  return outer;
}

Operation *emitGeneric(linalg::GenericOp generic) {
  OpBuilder b(generic);
  Location loc = generic.getLoc();
  auto p = plan(generic, "plan.vector");
  auto outputType = cast<MemRefType>(generic.getOutputs()[0].getType());
  int64_t lanes =
      number(p, "vector_bytes") / elementBytes(outputType.getElementType());
  auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                 index(b, loc, number(p, "elements")),
                                 index(b, loc, lanes));
  b.setInsertionPointToStart(loop.getBody());
  Block &body = generic.getRegion().front();
  IRMapping mapping;
  auto selections = p.getAs<ArrayAttr>("operands");
  for (auto [i, argument] : llvm::enumerate(body.getArguments())) {
    if (argument.use_empty())
      continue;
    Value operand = generic->getOperand(i);
    auto type = cast<MemRefType>(operand.getType());
    auto loaded = vector::LoadOp::create(
        b, loc, VectorType::get({lanes}, type.getElementType()), operand,
        ValueRange{loop.getInductionVar()});
    direct(loaded, cast<DictionaryAttr>(selections[i]), b);
    mapping.map(argument, loaded.getResult());
  }
  Operation &arithmetic = body.front();
  OperationState state(loc, arithmetic.getName());
  for (Value operand : arithmetic.getOperands())
    state.addOperands(mapping.lookup(operand));
  state.addTypes(VectorType::get({lanes}, arithmetic.getResult(0).getType()));
  state.addAttributes(arithmetic.getAttrs());
  Operation *computed = b.create(state);
  auto stored = vector::StoreOp::create(b, loc, computed->getResult(0),
                                        generic.getOutputs()[0],
                                        ValueRange{loop.getInductionVar()});
  direct(stored,
         cast<DictionaryAttr>(selections[generic->getNumOperands() - 1]), b);
  return loop;
}

Operation *emitReduce(linalg::ReduceOp reduce) {
  OpBuilder b(reduce);
  Location loc = reduce.getLoc();
  auto p = plan(reduce, "plan.reduce");
  Value source = reduce.getInputs()[0], output = reduce.getInits()[0];
  int64_t lanes = number(p, "vector_bytes") / 4;
  Value initial = memref::LoadOp::create(b, loc, output, ValueRange{});
  auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                 index(b, loc, number(p, "elements")),
                                 index(b, loc, lanes), ValueRange{initial});
  b.setInsertionPointToStart(loop.getBody());
  auto loaded =
      vector::LoadOp::create(b, loc, VectorType::get({lanes}, b.getF32Type()),
                             source, ValueRange{loop.getInductionVar()});
  direct(loaded, nested(p, "input"), b);
  Value next = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD,
                                           loaded, loop.getRegionIterArgs()[0]);
  scf::YieldOp::create(b, loc, next);
  b.setInsertionPointAfter(loop);
  return memref::StoreOp::create(b, loc, loop.getResult(0), output,
                                 ValueRange{});
}

struct EmitPass : PassWrapper<EmitPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPass)
  StringRef getArgument() const final { return "bosc-ame-emit"; }
  StringRef getDescription() const final {
    return "Mechanically expand verified BOSCAME plans";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<BOSCAMEDialect, arith::ArithDialect, func::FuncDialect,
                linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
                scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    if (failed(emitBOSCAMEPlan(getOperation())))
      signalPassFailure();
  }
};
} // namespace

LogicalResult buddy::boscame::emitBOSCAMEPlan(ModuleOp module) {
  // No mutation precedes full validation. No legacy planner is called here.
  if (failed(verifyBOSCAMEPlan(module)))
    return failure();
  DenseMap<Operation *, DenseSet<int64_t>> expected;
  module.walk([&](Operation *op) {
    if (auto task = plan(op, "contract.task"))
      expected[op->getParentOfType<func::FuncOp>()].insert(number(task, "id"));
  });
  OpBuilder builder(module.getContext());
  module->setAttr("bosc_ame.target", builder.getStringAttr("nr-fpga"));
  for (auto function : module.getOps<func::FuncOp>()) {
    if (Attribute mapping = function->getAttr("plan.task_map"))
      function->setAttr("bosc_ame.task_map", mapping);
    if (Attribute count = function->getAttr("plan.source_task_count"))
      function->setAttr("bosc_ame.source_task_count", count);
  }
  auto abi = plan(module, "contract.abi");
  if (abi && abi.getAs<IntegerAttr>("args_aligned_bytes") &&
      number(abi, "args_aligned_bytes") >= 64 &&
      number(abi, "args_aligned_bytes") % 64 == 0) {
    auto disjoint = abi.getAs<BoolAttr>("args_disjoint");
    module->setAttr(
        spmdma::kDmaAbiAttrName,
        builder.getStringAttr(disjoint && disjoint.getValue()
                                  ? spmdma::kDmaAbiAlignedNonAliasing
                                  : spmdma::kDmaAbiAligned));
  }
  materializeBuffers(module);
  SmallVector<Operation *> computations;
  module.walk([&](Operation *op) {
    if (isa<memref::CopyOp, linalg::MatmulOp, linalg::FillOp, linalg::GenericOp,
            linalg::ReduceOp>(op))
      computations.push_back(op);
  });
  for (Operation *op : computations) {
    Operation *completion = nullptr;
    if (auto copy = dyn_cast<memref::CopyOp>(op))
      completion = emitTransfer(copy);
    else if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
      completion = matmul->hasAttr("plan.compute") ? emitOrderedMatmul(matmul)
                                                   : emitMatmul(matmul);
    else if (auto fill = dyn_cast<linalg::FillOp>(op))
      completion = emitFill(fill);
    else if (auto generic = dyn_cast<linalg::GenericOp>(op))
      completion = emitGeneric(generic);
    else if (auto reduce = dyn_cast<linalg::ReduceOp>(op))
      completion = emitReduce(reduce);
    if (!completion)
      return op->emitError(
          "verified plan did not produce a completion operation");
    moveFacts(op, completion);
    op->erase();
  }
  bool invalid = false;
  DenseMap<Operation *, DenseSet<int64_t>> actual;
  module.walk([&](Operation *op) {
    if (auto task = plan(op, "contract.task"))
      if (!actual[op->getParentOfType<func::FuncOp>()]
               .insert(number(task, "id"))
               .second) {
        op->emitError("duplicate task completion after planned emission");
        invalid = true;
      }
    SmallVector<StringAttr> consumed;
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("plan."))
        consumed.push_back(attr.getName());
    for (StringAttr name : consumed)
      op->removeAttr(name);
  });
  for (auto &[function, ids] : expected)
    if (actual[function].size() != ids.size() ||
        llvm::any_of(
            ids, [&](int64_t id) { return !actual[function].contains(id); })) {
      function->emitError(
          "planned task completion IDs were lost during emission");
      invalid = true;
    }
  if (invalid)
    return failure();
  return verify(module);
}
std::unique_ptr<Pass> buddy::boscame::createBOSCAMEEmitPass() {
  return std::make_unique<EmitPass>();
}
void buddy::boscame::registerBOSCAMEPlanEmissionPasses() {
  registerPass([] { return createBOSCAMEPlanVerifyPass(); });
  registerPass([] { return createBOSCAMEEmitPass(); });
}
