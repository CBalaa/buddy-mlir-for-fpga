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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
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
mlir::LogicalResult fillVectors(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value buffer, mlir::Value value);
mlir::FailureOr<mlir::Value> getStoreBuffer(mlir::OpBuilder &builder,
                                            mlir::Operation *anchor,
                                            mlir::Value tensor);

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
mlir::LogicalResult planSpmComputationBuffers(mlir::ModuleOp module);
/// Check the exact operand placement accepted by SPM computation lowering.
/// This is a pure legality query: it does not rewrite the operation. Callers
/// may use a detached clone to model resident and operation-local buffers.
mlir::LogicalResult checkSpmComputation(mlir::Operation *operation);
/// Positional generic inputs whose private fill buffers can stay in registers.
/// Null entries retain ordinary buffer loads. A buffer may have successive
/// fills: each reader receives the broadcast from its latest dominating fill.
using SpmBroadcastMap =
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>>;
/// Replace private, disjoint SPM fill materializations with register
/// broadcasts. Readers keep their buffer operands until computation lowering
/// consumes the map. Preserve fill task IDs and logical dependencies when
/// verifying graphs.
mlir::LogicalResult forwardSpmBroadcasts(
    mlir::ModuleOp module, SpmBroadcastMap &broadcasts,
    llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> verifyGraph =
        nullptr);
/// When supplied, preserve each computation's graph metadata on its unique
/// completion operation and verify the graph after each lowering rewrite.
/// Static-prefix selects drop their mask dependencies before dead private
/// mask tasks are removed and the explicit graph user tables are refreshed.
mlir::LogicalResult lowerSpmComputations(
    mlir::ModuleOp module,
    llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> verifyGraph =
        nullptr);
/// Forward private rank-zero scalar storage to SSA, preserving task anchors.
void promotePrivateScalarBuffers(mlir::ModuleOp module);
/// Remove trivially dead lowering artifacts without erasing graph tasks.
void eraseDeadSpmComputationArtifacts(mlir::ModuleOp module);
mlir::LogicalResult verifyExit(mlir::ModuleOp module);

} // namespace buddy::boscame::lowering

#endif
