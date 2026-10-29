//===- BOSCAMEStreamPlan.h - Content-version streaming plans -----*- C++
//-*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMESTREAMPLAN_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMESTREAMPLAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace buddy::boscame {

/// Record a deterministic, content-version-based streaming plan without
/// changing the logical computations. Fixed hardware input windows are not
/// candidates for virtualization. Unsupported or unproved components retain
/// their ordinary buffer representation.
mlir::LogicalResult analyzeStreams(mlir::ModuleOp module,
                                   llvm::ArrayRef<mlir::Value> fixedRoots = {});

/// Recompute the semantic proof and compare every recorded recipe. This must
/// succeed before either buffer materialization or logical IR destruction.
mlir::LogicalResult verifyStreams(mlir::ModuleOp module);

/// Expand verified sinks, retaining exact scalar payloads and the original
/// ordered 16-lane reductions. Archive logical tasks separately from physical
/// execution anchors; one virtual producer may map to several anchors.
/// inputVerified is reserved for the combined emitter after its full-module
/// verification, which may have expanded other disjoint execution groups. In
/// that mode consume only the recorded set, still checking member boundaries
/// and every recipe, without rerunning source-stability analysis over already
/// emitted physical groups.
mlir::LogicalResult emitStreams(mlir::ModuleOp module,
                                bool inputVerified = false);

bool isStreamMember(mlir::Operation *operation);
bool isStreamVirtual(mlir::Value value);
bool isStreamVirtual(mlir::Operation *operation);

} // namespace buddy::boscame

#endif
