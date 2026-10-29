//===- BOSCAMERegisterSchedule.h - Register-resident AME recipes -*- C++
//-*-===//
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEREGISTERSCHEDULE_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEREGISTERSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace buddy::boscame {

/// Prove and record integer AME schedules without changing the logical graph.
/// Run after accumulator fusion and before snapshot forwarding/placement. A
/// second call refreshes recipes after contract task IDs have been rebuilt.
mlir::LogicalResult
analyzeRegisterSchedules(mlir::ModuleOp module,
                         llvm::ArrayRef<mlir::Value> fixedRoots = {});

/// These logical operations/allocations belong to a verified physical recipe;
/// ordinary storage selection and operation emission must leave them alone.
bool isRegisterScheduleMember(mlir::Operation *operation);
bool isRegisterScheduleVirtual(mlir::Operation *operation);

/// Reconstruct every proof from the logical graph and compare the complete
/// recorded recipe, membership and virtual-root ownership before mutation.
mlir::LogicalResult verifyRegisterSchedules(mlir::ModuleOp module);

/// Emit register schedules before materializeBuffers. Each physical group has
/// bosc_ame.execution_id and bosc_ame.register_tasks; covered contract.task
/// facts move to function bosc_ame.register_logical_tasks and are mapped by
/// bosc_ame.register_task_map. No covered contract.task remains on a physical
/// anchor. The caller excludes register members from its ordinary task set.
mlir::LogicalResult emitRegisterSchedules(mlir::ModuleOp module);

} // namespace buddy::boscame

#endif
