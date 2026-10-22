//===- BOSCAMEDmaEmitter.h - Typed DMA emitter ------------------*- C++ -*-===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_DMA_EMITTER_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_DMA_EMITTER_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFootprint.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
} // namespace mlir

namespace buddy::boscame {

/// Express frontend-provenance copies as ordinary buffer Linalg consumers.
/// The planner then sees the actual DDR endpoints and exact valid subviews.
mlir::LogicalResult normalizeBOSCAMEMemoryCopies(mlir::ModuleOp module,
                                                 bool directDDR = false);

/// Emit one packet from an already planned transfer.  The emitter is kept
/// small and typed so callers cannot silently pass element counts as bytes.
mlir::LogicalResult emitBOSCAMEDmaTransfers(mlir::ModuleOp module,
                                            const WorkspacePlan &workspace);

/// Lower pure copies between SPM windows to typed DMA packets and immediate
/// waits. Computations, including casts and transposes, are not byte copies.
mlir::LogicalResult emitBOSCAMEDmaCopies(mlir::ModuleOp module,
                                         const TargetInfo &target);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_DMA_EMITTER_H
