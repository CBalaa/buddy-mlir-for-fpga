//===- BOSCAMEOrderedMatmul.cpp - Ordered compatibility recipe
//-------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEOrderedMatmul.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace {
Value loadElement(OpBuilder &builder, Location loc, Value source,
                  ValueRange indices) {
  auto type = cast<MemRefType>(source.getType());
  if (!::buddy::boscame::spmdma::getSpmBank(type))
    return memref::LoadOp::create(builder, loc, source, indices);
  Value vector = vector::LoadOp::create(
      builder, loc, VectorType::get({1}, type.getElementType()), source,
      indices);
  return vector::ExtractOp::create(builder, loc, vector, int64_t{0});
}

void storeElement(OpBuilder &builder, Location loc, Value value, Value target,
                  ValueRange indices) {
  auto type = cast<MemRefType>(target.getType());
  if (!::buddy::boscame::spmdma::getSpmBank(type)) {
    memref::StoreOp::create(builder, loc, value, target, indices);
    return;
  }
  Value vector = vector::BroadcastOp::create(
      builder, loc, VectorType::get({1}, type.getElementType()), value);
  vector::StoreOp::create(builder, loc, vector, target, indices);
}
} // namespace

Operation *buddy::boscame::emitOrderedMatmul(linalg::MatmulOp matmul) {
  OpBuilder builder(matmul);
  Location loc = matmul.getLoc();
  Value a = matmul.getInputs()[0], b = matmul.getInputs()[1],
        c = matmul.getOutputs()[0];
  auto aType = cast<MemRefType>(a.getType());
  auto cType = cast<MemRefType>(c.getType());
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value mBound =
      arith::ConstantIndexOp::create(builder, loc, cType.getDimSize(0));
  Value nBound =
      arith::ConstantIndexOp::create(builder, loc, cType.getDimSize(1));
  Value kBound =
      arith::ConstantIndexOp::create(builder, loc, aType.getDimSize(1));
  auto mLoop = scf::ForOp::create(builder, loc, zero, mBound, one);
  builder.setInsertionPointToStart(mLoop.getBody());
  auto nLoop = scf::ForOp::create(builder, loc, zero, nBound, one);
  builder.setInsertionPointToStart(nLoop.getBody());
  auto kLoop = scf::ForOp::create(builder, loc, zero, kBound, one);
  builder.setInsertionPointToStart(kLoop.getBody());
  Value m = mLoop.getInductionVar(), n = nLoop.getInductionVar(),
        k = kLoop.getInductionVar();
  Value lhs = loadElement(builder, loc, a, ValueRange{m, k});
  Value rhs = loadElement(builder, loc, b, ValueRange{k, n});
  Value accumulator = loadElement(builder, loc, c, ValueRange{m, n});
  Block &payload = matmul.getRegion().front();
  IRMapping mapping;
  mapping.map(payload.getArguments(), ValueRange{lhs, rhs, accumulator});
  for (Operation &operation : payload.without_terminator())
    builder.clone(operation, mapping);
  auto yield = cast<linalg::YieldOp>(payload.getTerminator());
  storeElement(builder, loc, mapping.lookup(yield.getValues()[0]), c,
               ValueRange{m, n});
  return mLoop;
}
