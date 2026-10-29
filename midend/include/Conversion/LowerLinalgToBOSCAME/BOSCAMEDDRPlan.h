//===- BOSCAMEDDRPlan.h - Exact CPU recipes for materialized DDR
//------------===//
#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEDDRPLAN_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEDDRPLAN_H

#include "mlir/IR/BuiltinOps.h"

namespace buddy::boscame {
/// A recipe is derived from the actual operation, never a permission to change
/// its arithmetic. These recipes deliberately do not access SPM.
mlir::FailureOr<mlir::DictionaryAttr> getDDRRecipe(mlir::Operation *op);
mlir::LogicalResult verifyDDRRecipe(mlir::Operation *op);
mlir::Operation *emitDDRRecipe(mlir::Operation *op);
} // namespace buddy::boscame
#endif
