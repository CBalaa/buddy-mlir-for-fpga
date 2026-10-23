//===- BOSCAMEPlacement.h - Root storage placement ------------*- C++ -*-===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_PLACEMENT_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_PLACEMENT_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFootprint.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace buddy::boscame {

/// Fixed windows can span serial, single-block if/for/execute_region scopes.
/// Storage may not escape through region arguments/results or unknown users.
bool hasBOSCAMEPlacementScope(mlir::Operation *operation);

/// Allocations confined to opposite arms of one scf.if may share a window.
bool haveExclusiveBOSCAMEPlacementScopes(mlir::Operation *first,
                                         mlir::Operation *second);

/// Model all selected roots and the planner's operation-local staging on a
/// detached clone, then invoke the same legality checks as SPM lowering.
/// Neither the module nor its plan is changed on success or failure. The
/// caller owns the unified resident/explicit/staging capacity reservation.
mlir::LogicalResult
preflightBufferPlacements(mlir::ModuleOp module, const DataflowPlan &plan,
                          llvm::ArrayRef<TransferPlan> staging,
                          std::string &reason);

/// Apply a fully preflighted plan. Structural and graph checks first run on
/// a detached clone, so an invalid request never partially rewrites a group.
/// The rewrite preserves each surviving compute task and its logical values.
mlir::LogicalResult applyBufferPlacements(mlir::ModuleOp module,
                                          DataflowPlan &plan);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_PLACEMENT_H
