//===- ContractTileSpecialization.h - Rectangular tiles
//---------------------===//

#ifndef BUDDY_CONTRACTTILESPECIALIZATION_H
#define BUDDY_CONTRACTTILESPECIALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton {
/// Specialize proven rectangular load/store bounds before graph annotation.
LogicalResult specializeContractTiles(ModuleOp module);
} // namespace mlir::triton

#endif
