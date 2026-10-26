//===- BOSCAMEPointwise.h - Supported lane-wise scalar expressions --------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPOINTWISE_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPOINTWISE_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <algorithm>

namespace buddy::boscame {

inline bool supportedPointwiseScalar(mlir::Type type) {
  return type.isF32() || type.isInteger(1) || type.isInteger(8) ||
         type.isInteger(16) || type.isInteger(32) || type.isInteger(64);
}

inline bool supportedPointwiseOperation(mlir::Operation &op) {
  using namespace mlir;
  return op.getNumResults() == 1 && op.getNumRegions() == 0 &&
         llvm::all_of(op.getOperandTypes(), supportedPointwiseScalar) &&
         supportedPointwiseScalar(op.getResult(0).getType()) &&
         isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::NegFOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
             arith::MaxNumFOp, arith::MinNumFOp, arith::FPToSIOp,
             arith::SIToFPOp, arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp,
             math::AbsFOp>(op);
}

/// Use one common lane count for the whole expression. Narrow results and
/// masks occupy smaller vectors; their memory footprint is never promoted to
/// the widest operand's footprint. All intermediate operations stay ordered.
inline int64_t pointwiseLanes(mlir::linalg::GenericOp generic) {
  int64_t widest = 8;
  for (mlir::Type type : generic.getBody()->getArgumentTypes())
    if (supportedPointwiseScalar(type))
      widest = std::max<int64_t>(widest, type.getIntOrFloatBitWidth());
  for (mlir::Operation &op : generic.getBody()->without_terminator()) {
    for (mlir::Type type : op.getOperandTypes())
      if (supportedPointwiseScalar(type))
        widest = std::max<int64_t>(widest, type.getIntOrFloatBitWidth());
    for (mlir::Type type : op.getResultTypes())
      if (supportedPointwiseScalar(type))
        widest = std::max<int64_t>(widest, type.getIntOrFloatBitWidth());
  }
  return 512 / widest;
}

} // namespace buddy::boscame

#endif
