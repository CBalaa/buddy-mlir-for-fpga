//===- LowerLinalgToBOSCAME.h - Linalg BOSCAME lowering --------*- C++ -*-===//
//
// The public entry points for the buffer-form Linalg BOSCAME backend.  The
// planner is deliberately separate from the legacy pattern lowering so that
// the default experimental pipeline remains unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H

#include <cstdint>
#include <memory>
#include <string>

namespace buddy::boscame {

struct LinalgToBOSCAMEConfig {
  bool enableSpmDma = false;
  bool enableAME = false;
  bool bufferPlacement = false;
  /// Policy used for stores whose destination is external DDR.  Direct DDR
  /// stores are the default; `spm-dma` is an explicit compatibility mode for
  /// kernels that still require the historical SPM staging path.
  std::string storePolicy = "direct-ddr";
  std::string dmaAbi;
  std::string target;
  int64_t bm = 64;
  int64_t bn = 64;
  int64_t bk = 128;
};

} // namespace buddy::boscame

namespace mlir {
class Pass;
namespace buddy {
std::unique_ptr<Pass> createLowerLinalgToBOSCAMEPass();
std::unique_ptr<Pass> createLinalgToBOSCAMEPass();
} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H
