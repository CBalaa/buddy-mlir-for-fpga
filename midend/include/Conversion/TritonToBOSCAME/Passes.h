//===----------------------------------------------------------------------===//
//
// TTIR -> BOSCAME conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_BOSCAME_CONVERSION_PASSES_H
#define TRITON_TO_BOSCAME_CONVERSION_PASSES_H

#include "Conversion/TritonToBOSCAME/TritonToBOSCAME.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "Conversion/TritonToBOSCAME/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_BOSCAME_CONVERSION_PASSES_H
