// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Contract passes and planner staging policy.
#include "Analysis/KernelContract.h"
#include "Analysis/detail/KernelContractCore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
namespace buddy::contract {
namespace {
bool isPlannerStaging(Operation *op) {
  auto provenance = op->getAttrOfType<DictionaryAttr>("contract.provenance");
  auto producer =
      provenance ? provenance.getAs<StringAttr>("generated_by") : StringAttr();
  return producer && producer.getValue() == "planner-staging";
}
detail::SourcePolicy sourcePolicy() { return {isPlannerStaging}; }

struct AnnotatePass : PassWrapper<AnnotatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotatePass)
  StringRef getArgument() const final { return "kernel-contract-annotate"; }
  StringRef getDescription() const final { return "Annotate neutral Contract"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    if (failed(annotateKernelContract(getOperation())))
      signalPassFailure();
  }
};
struct VerifyPass : PassWrapper<VerifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPass)
  VerifyPass() = default;
  VerifyPass(const VerifyPass &other) : PassWrapper(other) {}
  Option<bool> planned{*this, "planned", llvm::cl::init(false),
                       llvm::cl::desc("Accept Buddy plan attributes")};
  StringRef getArgument() const final { return "kernel-contract-verify"; }
  StringRef getDescription() const final { return "Verify Contract semantics"; }
  void runOnOperation() override {
    if (failed(verifyKernelContract(getOperation(), planned)))
      signalPassFailure();
  }
};
} // namespace

bool hasEntryArgumentABI(func::FuncOp function) {
  return detail::hasEntryArgumentABI(function);
}
LogicalResult annotateKernelContract(ModuleOp module, StringRef producer,
                                     bool planned) {
  if (planned)
    return detail::rebuildPlannedContract(module, producer, sourcePolicy());
  return detail::annotateKernelContract(module, producer, sourcePolicy());
}
LogicalResult verifyKernelContract(ModuleOp module, bool planned) {
  if (planned)
    return detail::verifyPlannedContract(module, sourcePolicy());
  return detail::verifyKernelContract(module, sourcePolicy());
}
std::unique_ptr<Pass> createKernelContractAnnotatePass() {
  return std::make_unique<AnnotatePass>();
}
std::unique_ptr<Pass> createKernelContractVerifyPass() {
  return std::make_unique<VerifyPass>();
}
void registerKernelContractPasses() {
  PassRegistration<AnnotatePass>();
  PassRegistration<VerifyPass>();
}
} // namespace buddy::contract
