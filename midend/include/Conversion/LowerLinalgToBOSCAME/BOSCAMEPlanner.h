//===- BOSCAMEPlanner.h - Linalg BOSCAME planner ----------------*- C++ -*-===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_PLANNER_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_PLANNER_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFootprint.h"
#include "Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace buddy::boscame {

/// Analyze buffer-form Linalg and produce a bounded transfer/workspace plan.
/// The planner never performs implicit bufferization and never treats a
/// scalar SPM memref as a legal transfer.  Emitters consume this plan later.
class BOSCAMEPlanner {
public:
  explicit BOSCAMEPlanner(const LinalgToBOSCAMEConfig &config)
      : config(config) {}

  mlir::FailureOr<DataflowPlan> plan(mlir::ModuleOp module) const;

private:
  LinalgToBOSCAMEConfig config;
};

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_PLANNER_H
