//===- KernelContract.h - Target-independent buffer contracts ----*- C++
//-*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_KERNELCONTRACT_H
#define BUDDY_ANALYSIS_KERNELCONTRACT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace buddy::contract {

/// Module argument ABI promises apply only to entry functions with no symbol
/// uses in the module. Called/address-taken helpers can receive aliasing or
/// unaligned views, and require separate call-site proofs.
bool hasEntryArgumentABI(mlir::func::FuncOp function);

/// Rebuild scopes, tasks, content versions and loop summaries from buffer-form
/// standard IR. Retain neutral access, ABI, provenance and origin facts. No
/// target dialect, bank, memory-space or route decision is introduced.
mlir::LogicalResult
annotateKernelContract(mlir::ModuleOp module,
                       llvm::StringRef producer = "buddy-neutral",
                       bool planned = false);

/// Verify the neutral contract against actual operands, payloads and program
/// order. planned permits target plan attributes; it does not relax memory or
/// content-version correctness. Diagnostics identify the failing operation.
mlir::LogicalResult verifyKernelContract(mlir::ModuleOp module,
                                         bool planned = false);

std::unique_ptr<mlir::Pass> createKernelContractAnnotatePass();
std::unique_ptr<mlir::Pass> createKernelContractVerifyPass();
void registerKernelContractPasses();

} // namespace buddy::contract

#endif // BUDDY_ANALYSIS_KERNELCONTRACT_H
