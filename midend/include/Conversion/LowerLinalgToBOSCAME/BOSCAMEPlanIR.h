//===- BOSCAMEPlanIR.h - Verified planned IR emission -----------*- C++ -*-===//
//
// Part of the Buddy Compiler, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPLANIR_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPLANIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
}
namespace llvm {
class raw_ostream;
}

namespace buddy::boscame {

/// Verify the complete concrete plan before any emission mutates the module.
mlir::LogicalResult verifyBOSCAMEPlan(mlir::ModuleOp module);
/// Prove that both private roots and all aliases remain in distinct exclusive
/// arms and have only supported synchronous, non-escaping memory uses.
bool canShareExclusiveSPMWindow(mlir::Value first, mlir::Value second);
/// Target input constraints are passed out of band: the neutral interchange
/// IR never contains fixed target windows. Roots are standard private memrefs
/// produced by the compatibility adapter after validating the physical input.
struct FixedSPMBuffer {
  mlir::Value root;
  int64_t bank;
  int64_t offset;
  int64_t capacity;
};

/// Physical placement debugging does not affect semantic fusion. Disabling
/// automatic placement disables optional SPM reuse optimization; required
/// materializations still use DDR, and explicit fixed windows remain valid.
struct PlanOptions {
  bool automaticPlacement = true;
  /// V1 is a frozen compatibility/reference planner; production defaults to V2.
  int64_t version = 2;
};

/// Select and record target routes on a target-independent buffer contract.
mlir::LogicalResult
planBOSCAME(mlir::ModuleOp module,
            llvm::ArrayRef<FixedSPMBuffer> fixedBuffers = {},
            PlanOptions options = {});
/// Report recorded decisions without rerunning analysis or selecting routes.
void printBOSCAMEPlan(mlir::ModuleOp module, llvm::raw_ostream &stream);
/// Mechanically expand a verified plan, preserving task/source provenance.
mlir::LogicalResult emitBOSCAMEPlan(mlir::ModuleOp module);
std::unique_ptr<mlir::Pass> createBOSCAMEPlanPass();
std::unique_ptr<mlir::Pass> createBOSCAMEEmitPass();
std::unique_ptr<mlir::Pass> createBOSCAMEPlanVerifyPass();
void registerBOSCAMEPlanEmissionPasses();
void registerBOSCAMEPlanPasses();

} // namespace buddy::boscame

#endif
