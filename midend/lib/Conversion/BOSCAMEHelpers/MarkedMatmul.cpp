//===- MarkedMatmul.cpp - Buffer-form integer matmul to AME
//------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/Transforms/BOSCAMETileLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {
namespace {

Value index(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

Value flatten(OpBuilder &builder, Location loc, Value buffer) {
  auto type = cast<MemRefType>(buffer.getType());
  auto flat = MemRefType::get({type.getNumElements()}, type.getElementType());
  return memref::CollapseShapeOp::create(
      builder, loc, flat, buffer, ArrayRef<ReassociationIndices>{{0, 1}});
}

Value typedSpm(OpBuilder &builder, Location loc, Value flat,
               ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  auto source = cast<MemRefType>(flat.getType());
  auto type =
      MemRefType::get(shape, source.getElementType(),
                      StridedLayoutAttr::get(builder.getContext(), 0, strides),
                      source.getMemorySpace());
  return memref::ReinterpretCastOp::create(builder, loc, type, flat, 0, shape,
                                           strides);
}

/// All packing accesses ordinary memory, so scalar packing is safe. Its index
/// callback preserves the logical [K,N] B indexing irrespective of its strides.
template <typename Copy>
void copyRectangle(OpBuilder &builder, Location loc, int64_t rows,
                   int64_t columns, Copy copy) {
  OpBuilder::InsertionGuard outer(builder);
  Value zero = index(builder, loc, 0), one = index(builder, loc, 1);
  auto rowLoop =
      scf::ForOp::create(builder, loc, zero, index(builder, loc, rows), one);
  builder.setInsertionPointToStart(rowLoop.getBody());
  auto colLoop =
      scf::ForOp::create(builder, loc, zero, index(builder, loc, columns), one);
  builder.setInsertionPointToStart(colLoop.getBody());
  copy(rowLoop.getInductionVar(), colLoop.getInductionVar());
}

LogicalResult lowerMatmul(linalg::MatmulOp matmul, Workspace &workspace) {
  if (matmul.getNumResults() || matmul.getInputs().size() != 2 ||
      matmul.getOutputs().size() != 1)
    return matmul.emitError("marked AME matmul must have buffer semantics");
  Value a = matmul.getInputs()[0], b = matmul.getInputs()[1],
        c = matmul.getOutputs()[0];
  auto at = dyn_cast<MemRefType>(a.getType()),
       bt = dyn_cast<MemRefType>(b.getType()),
       ct = dyn_cast<MemRefType>(c.getType());
  if (!at || !bt || !ct || at.getRank() != 2 || bt.getRank() != 2 ||
      ct.getRank() != 2 || !at.hasStaticShape() || !bt.hasStaticShape() ||
      !ct.hasStaticShape() || !at.getElementType().isInteger(8) ||
      !bt.getElementType().isInteger(8) || !ct.getElementType().isInteger(32) ||
      getSpmBank(at) || getSpmBank(bt) || getSpmBank(ct))
    return matmul.emitError(
        "AME target requires static ordinary i8/i8/i32 rank-2 buffers");
  if (matmul.getCast() != linalg::TypeFn::cast_signed)
    return matmul.emitError("AME target requires signed integer extension");
  int64_t m = ct.getDimSize(0), n = ct.getDimSize(1), k = at.getDimSize(1);
  if (m <= 0 || n <= 0 || k <= 0 || at.getDimSize(0) != m ||
      bt.getDimSize(0) != k || bt.getDimSize(1) != n)
    return matmul.emitError("AME target has unsupported matmul dimensions");
  OpBuilder builder(matmul);
  Location loc = matmul.getLoc();
  auto alloc = [&](ArrayRef<int64_t> shape, Type element) -> Value {
    return memref::AllocOp::create(builder, loc,
                                   MemRefType::get(shape, element),
                                   builder.getI64IntegerAttr(64));
  };
  auto fillZero = [&](Value buffer) {
    Type element = cast<MemRefType>(buffer.getType()).getElementType();
    Value zero = arith::ConstantOp::create(builder, loc, element,
                                           builder.getIntegerAttr(element, 0));
    linalg::FillOp::create(builder, loc, ValueRange{zero}, ValueRange{buffer});
  };
  // Each static M/N tile gets its own windows. K blocks reuse the windows of
  // this access point only after the preceding values have been stored.
  constexpr int64_t tile = 64;
  for (int64_t mi = 0; mi < m; mi += tile) {
    for (int64_t ni = 0; ni < n; ni += tile) {
      int64_t tm = std::min(tile, m - mi), tn = std::min(tile, n - ni);
      int64_t pitch = llvm::alignTo(tn, int64_t(16));
      auto spmA = workspace.allocate(
          builder, matmul, MemRefType::get({tm * tile}, builder.getI8Type()),
          0);
      auto spmB = workspace.allocate(
          builder, matmul, MemRefType::get({tn * tile}, builder.getI8Type()),
          1);
      auto spmC = workspace.allocate(
          builder, matmul, MemRefType::get({tm * pitch}, builder.getI32Type()),
          0);
      if (failed(spmA) || failed(spmB) || failed(spmC))
        return failure();
      Value aPack = alloc({tm, tile}, builder.getI8Type());
      Value bPack = alloc({tn, tile}, builder.getI8Type());
      Value cPack = alloc({tm, pitch}, builder.getI32Type());
      Value aFlat = flatten(builder, loc, aPack),
            bFlat = flatten(builder, loc, bPack),
            cFlat = flatten(builder, loc, cPack);
      Value aTile = typedSpm(builder, loc, *spmA, {tm, tile}, {tile, 1});
      Value bTile = typedSpm(builder, loc, *spmB, {tile, tn}, {1, tile});
      Value cTile = typedSpm(builder, loc, *spmC, {tm, tn}, {pitch, 1});
      fillZero(cPack);
      // Seed exactly the standard matmul's C. MatmulConverter's original dot C
      // is added separately by Linalg and must not be added here again.
      copyRectangle(builder, loc, tm, tn, [&](Value i, Value j) {
        Value ci =
            arith::AddIOp::create(builder, loc, i, index(builder, loc, mi));
        Value cj =
            arith::AddIOp::create(builder, loc, j, index(builder, loc, ni));
        Value value =
            memref::LoadOp::create(builder, loc, c, ValueRange{ci, cj});
        memref::StoreOp::create(builder, loc, value, cPack, ValueRange{i, j});
      });
      emitDmaAndWait(builder, loc, cFlat, *spmC, tm * pitch * 4);
      for (int64_t ki = 0; ki < k; ki += tile) {
        int64_t tk = std::min(tile, k - ki);
        fillZero(aPack);
        fillZero(bPack);
        copyRectangle(builder, loc, tm, tk, [&](Value i, Value kk) {
          Value ai =
              arith::AddIOp::create(builder, loc, i, index(builder, loc, mi));
          Value ak =
              arith::AddIOp::create(builder, loc, kk, index(builder, loc, ki));
          Value value =
              memref::LoadOp::create(builder, loc, a, ValueRange{ai, ak});
          memref::StoreOp::create(builder, loc, value, aPack,
                                  ValueRange{i, kk});
        });
        copyRectangle(builder, loc, tn, tk, [&](Value j, Value kk) {
          Value bk =
              arith::AddIOp::create(builder, loc, kk, index(builder, loc, ki));
          Value bj =
              arith::AddIOp::create(builder, loc, j, index(builder, loc, ni));
          Value value =
              memref::LoadOp::create(builder, loc, b, ValueRange{bk, bj});
          memref::StoreOp::create(builder, loc, value, bPack,
                                  ValueRange{j, kk});
        });
        emitDmaAndWait(builder, loc, aFlat, *spmA, tm * tile);
        emitDmaAndWait(builder, loc, bFlat, *spmB, tn * tile);
        // All runtime calls finish before any AME value becomes live.
        auto accType = VectorType::get({tm, tn}, builder.getI32Type());
        Value acc = TileLoadAccOp::create(builder, loc, accType, cTile);
        Value ta = TileLoadAOp::create(
            builder, loc, VectorType::get({tm, tile}, builder.getI8Type()),
            aTile);
        Value tb = TileLoadBOp::create(
            builder, loc, VectorType::get({tile, tn}, builder.getI8Type()),
            bTile);
        Value result = TileMmaOp::create(builder, loc, accType, acc, ta, tb);
        TileStoreAccOp::create(builder, loc, result, cTile);
      }
      emitDdrPolicy(builder, loc, /*write=*/true);
      if (failed(copyVectors(builder, loc, *spmC, cFlat, tm * pitch)))
        return failure();
      copyRectangle(builder, loc, tm, tn, [&](Value i, Value j) {
        Value value =
            memref::LoadOp::create(builder, loc, cPack, ValueRange{i, j});
        Value ci =
            arith::AddIOp::create(builder, loc, i, index(builder, loc, mi));
        Value cj =
            arith::AddIOp::create(builder, loc, j, index(builder, loc, ni));
        memref::StoreOp::create(builder, loc, value, c, ValueRange{ci, cj});
      });
    }
  }
  matmul.erase();
  return success();
}

} // namespace

LogicalResult lowerMarkedMatmuls(ModuleOp module) {
  SmallVector<linalg::MatmulOp> targets;
  module.walk([&](linalg::MatmulOp op) {
    if (op->hasAttr(kLowerToAme))
      targets.push_back(op);
  });
  if (targets.empty())
    return success();
  auto target = module->getAttrOfType<StringAttr>(kAmeTargetAttrName);
  if (target && target.getValue() != kAmeTargetNrFpga)
    return module.emitError(
        "marked AME matmul requires bosc_ame.target=nr-fpga");
  module->setAttr(kAmeTargetAttrName,
                  StringAttr::get(module.getContext(), kAmeTargetNrFpga));
  Workspace workspace;
  for (linalg::MatmulOp op : targets)
    if (failed(lowerMatmul(op, workspace)))
      return failure();
  return success();
}

} // namespace buddy::boscame::lowering
