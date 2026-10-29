//===- Endpoint.h - AME memory endpoint provenance ---------------*- C++
//-*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef BUDDY_DIALECT_BOSCAME_ENDPOINT_H
#define BUDDY_DIALECT_BOSCAME_ENDPOINT_H
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>
namespace buddy::boscame {
enum class EndpointKind {
  Unknown,
  ExternalDDR,
  PrivateAllocation,
  PrivateDDRFrame,
  SPM
};
inline constexpr int64_t kPrivateDDRFrameAlignment = 64;
inline constexpr int64_t kPrivateDDRFrameMaxBytes = 262016;
mlir::Value getEndpointRoot(mlir::Value value);
EndpointKind classifyEndpoint(mlir::Value value);
std::optional<int64_t> getStaticStorageBytes(mlir::Value value);
mlir::LogicalResult verifyPrivateAllocation(mlir::Operation *diagnostic,
                                            mlir::Value value);
mlir::LogicalResult verifyPrivateEndpoint(mlir::Operation *diagnostic,
                                          mlir::Value value);
} // namespace buddy::boscame
#endif
