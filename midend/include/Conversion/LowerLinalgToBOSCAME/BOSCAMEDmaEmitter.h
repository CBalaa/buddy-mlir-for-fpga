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
/// waits. When requested, retain the copy's graph node at its final DMA wait,
/// which completes the entire event, including every packet.
mlir::LogicalResult emitBOSCAMEDmaCopies(mlir::ModuleOp module,
                                         const TargetInfo &target,
                                         bool preserveGraph = false);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_DMA_EMITTER_H
