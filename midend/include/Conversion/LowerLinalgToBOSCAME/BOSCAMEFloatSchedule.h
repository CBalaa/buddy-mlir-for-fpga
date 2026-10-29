//===- BOSCAMEFloatSchedule.h - Ordered scalar register microtiles
//---------===//
#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEFLOATSCHEDULE_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEFLOATSCHEDULE_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
namespace buddy::boscame {
mlir::LogicalResult
analyzeFloatSchedules(mlir::ModuleOp module,
                      llvm::ArrayRef<mlir::Value> fixedRoots = {});
mlir::LogicalResult verifyFloatSchedules(mlir::ModuleOp module);
mlir::LogicalResult emitFloatSchedules(mlir::ModuleOp module);
bool isFloatScheduleMember(mlir::Operation *op);
bool isFloatScheduleVirtual(mlir::Operation *op);
} // namespace buddy::boscame
#endif
