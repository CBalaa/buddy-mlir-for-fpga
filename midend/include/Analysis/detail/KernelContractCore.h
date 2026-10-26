//===- KernelContractCore.h - Buffer Contract implementation -----*- C++
//-*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_DETAIL_KERNELCONTRACTCORE_H
#define BUDDY_ANALYSIS_DETAIL_KERNELCONTRACTCORE_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class ModuleOp;
class Operation;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace buddy::contract::detail {

/// Annotation and verification must use the same policy for synthetic copies.
/// Such copies cannot supply a missing source-language boundary event.
struct SourcePolicy {
  bool (*isSyntheticTransfer)(mlir::Operation *) = nullptr;
};

mlir::LogicalResult checkKernelContractVersions(mlir::ModuleOp module);
bool hasEntryArgumentABI(mlir::func::FuncOp function);

mlir::LogicalResult annotateKernelContract(mlir::ModuleOp module,
                                           llvm::StringRef producer,
                                           SourcePolicy sourcePolicy = {});
mlir::LogicalResult verifyKernelContract(mlir::ModuleOp module,
                                         SourcePolicy sourcePolicy = {});

/// Rebuild graph metadata after planning without repeating producer
/// normalization. Existing plan.* attributes are accepted; logical memory
/// space must remain zero.
mlir::LogicalResult rebuildPlannedContract(mlir::ModuleOp module,
                                           llvm::StringRef producer,
                                           SourcePolicy sourcePolicy);
mlir::LogicalResult verifyPlannedContract(mlir::ModuleOp module,
                                          SourcePolicy sourcePolicy);

} // namespace buddy::contract::detail

#endif // BUDDY_ANALYSIS_DETAIL_KERNELCONTRACTCORE_H
