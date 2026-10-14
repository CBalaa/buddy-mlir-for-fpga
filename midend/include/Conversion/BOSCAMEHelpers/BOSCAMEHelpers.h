//===- BOSCAMEHelpers.h - Shared DMA and AME lowering helpers ------*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_BOSCAMEHELPERS_H
#define BUDDY_CONVERSION_BOSCAMEHELPERS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include <array>

namespace buddy::boscame::lowering {

/// One static, non-reusing budget per function, shared by accesses and matmul.
/// Existing spm.view operations seed the budget after bufferization.
class Workspace {
public:
  mlir::FailureOr<mlir::Value> allocate(mlir::OpBuilder &builder,
                                        mlir::Operation *anchor,
                                        mlir::MemRefType type,
                                        int64_t preferredBank = -1);

private:
  llvm::DenseMap<mlir::Operation *, std::array<int64_t, 2>> used;
};

mlir::LogicalResult declareDdrPolicies(mlir::ModuleOp module);
void emitDdrPolicy(mlir::OpBuilder &builder, mlir::Location loc, bool write);
inline constexpr llvm::StringLiteral kCpuDdrPolicy = "bosc_ame.cpu_ddr_policy";
mlir::LogicalResult emitCpuDdrPolicies(mlir::ModuleOp module);
void emitDmaAndWait(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value src, mlir::Value dst, int64_t bytes);

/// Both memrefs are rank one. Each transfer is a complete 64-byte vector;
/// this helper never emits a scalar tail, including on SPM.
mlir::LogicalResult copyVectors(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value src, mlir::Value dst,
                                int64_t elementCount);

mlir::FailureOr<mlir::Value> lowerLoad(mlir::OpBuilder &builder,
                                       mlir::Operation *anchor, mlir::Value ptr,
                                       mlir::RankedTensorType type,
                                       llvm::ArrayRef<mlir::OpFoldResult> mask,
                                       mlir::Value other, Workspace &workspace);
mlir::LogicalResult lowerStore(mlir::OpBuilder &builder,
                               mlir::Operation *anchor, mlir::Value ptr,
                               mlir::Value tensor,
                               llvm::ArrayRef<mlir::OpFoldResult> mask,
                               Workspace &workspace);

inline constexpr llvm::StringLiteral kLowerToAme = "bosc_ame.lower_to_ame";
mlir::LogicalResult lowerMarkedMatmuls(mlir::ModuleOp module);
mlir::LogicalResult verifyExit(mlir::ModuleOp module);

} // namespace buddy::boscame::lowering

#endif
