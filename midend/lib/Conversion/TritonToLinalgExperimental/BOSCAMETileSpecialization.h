//===- BOSCAMETileSpecialization.h - Rectangular tiles
//---------------------===//

#ifndef BUDDY_BOSCAMETILESPECIALIZATION_H
#define BUDDY_BOSCAMETILESPECIALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton {
/// Specialize proven rectangular load/store bounds before graph annotation.
LogicalResult specializeBOSCAMETiles(ModuleOp module);
} // namespace mlir::triton

#endif
