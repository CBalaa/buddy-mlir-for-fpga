//===- BOSCAMEOrderedMatmul.h - Ordered compatibility recipe ------*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEORDEREDMATMUL_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEORDEREDMATMUL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace buddy::boscame {

/// Mechanically expand a verified scalar matmul recipe in serial M/N/K order.
/// Clone the original scalar payload, including its casts and floating flags.
/// Memory spaces have already been materialized from the operand bindings;
/// SPM accesses use one-element vectors rather than prohibited scalar loads.
/// The caller moves completion facts and erases the original operation.
mlir::Operation *emitOrderedMatmul(mlir::linalg::MatmulOp matmul);

} // namespace buddy::boscame

#endif
