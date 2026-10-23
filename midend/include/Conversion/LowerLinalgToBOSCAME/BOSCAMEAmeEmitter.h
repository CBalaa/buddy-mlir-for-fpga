//===- BOSCAMEAmeEmitter.h - AME emitter -------------------------*- C++
//-*-===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_AME_EMITTER_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_AME_EMITTER_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace buddy::boscame {

struct LinalgToBOSCAMEConfig;
struct DataflowPlan;
struct TargetInfo;

/// Check a prospective placed matmul with the emitter's actual tile layouts.
mlir::LogicalResult verifyBOSCAMEPlacedMatmul(mlir::Operation *operation,
                                              const TargetInfo &target);

/// Validate the AME eligibility of a marked buffer-form matmul.  Floating
/// point and unmarked integer matmuls stay on the ordinary RVV/memory path.
mlir::LogicalResult verifyBOSCAMEAmeConsumers(mlir::ModuleOp module,
                                              const DataflowPlan &plan);

/// Lower marked signed i8 matmuls whose A/B/C operands have already been
/// staged in SPM. A and C are row-major; logical B[K,N] has K contiguous in
/// the physical [N,K] storage. This emitter inserts only static subviews and
/// AME tile operations: no allocation, packing, scalar memory access or DMA.
/// Each C tile is seeded once, accumulated across K in SSA, then stored once.
/// Graph preservation moves each task's metadata to its final tile store and
/// verifies the graph after every rewrite; the consumed lowering marker is
/// removed. The default preserves the established unannotated emitted IR.
mlir::LogicalResult lowerBOSCAMEAmeConsumers(
    mlir::ModuleOp module, const LinalgToBOSCAMEConfig &config,
    const DataflowPlan &plan, bool preserveComputeGraph = false);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_AME_EMITTER_H
