// RUN: buddy-kernel-contract-policy-test
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Exercise the public callback with a producer-neutral spelling.
#include "Analysis/detail/KernelContractCore.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static bool synthetic(Operation *op) { return op->hasAttr("test.synthetic"); }

int main() {
  MLIRContext context;
  context.loadDialect<func::FuncDialect, memref::MemRefDialect>();
  for (bool retainedComplete : {false, true}) {
    for (bool usePolicy : {false, true}) {
      std::string source = R"mlir(
        module {func.func @f(%source: memref<4xi32>) {
          %snapshot = memref.alloc() : memref<4xi32>
          memref.copy %source, %snapshot {
            test.synthetic,
            contract.access = {kind = "load", boundary = "external_ddr",
              mask = {kind = "none"}, origin_id = 999 : i64},
            contract.provenance = {source_load_origin_ids = array<i64: 42>,
              source_lineage_complete = )mlir";
      source += retainedComplete ? "true" : "false";
      source += R"mlir(}} : memref<4xi32> to memref<4xi32>
          return
        }})mlir";
      auto module = parseSourceString<ModuleOp>(source, &context);
      buddy::contract::detail::SourcePolicy policy =
          usePolicy ? buddy::contract::detail::SourcePolicy{synthetic}
                    : buddy::contract::detail::SourcePolicy{};
      if (!module ||
          failed(buddy::contract::detail::annotateKernelContract(
              *module, "callback-test", policy)) ||
          failed(
              buddy::contract::detail::verifyKernelContract(*module, policy)))
        return 1;
      auto function = *module->getOps<func::FuncOp>().begin();
      auto graph =
          function->getAttrOfType<DictionaryAttr>("contract.compute_graph");
      bool actual = graph.getAs<BoolAttr>("source_lineage_complete").getValue();
      if (actual != (retainedComplete || !usePolicy)) {
        llvm::errs()
            << "synthetic transfer upgraded incomplete source history\n";
        return 1;
      }
      if (usePolicy && !retainedComplete &&
          succeeded(buddy::contract::detail::verifyKernelContract(*module))) {
        llvm::errs() << "verification did not apply the same source policy\n";
        return 1;
      }
    }
  }
  return 0;
}
