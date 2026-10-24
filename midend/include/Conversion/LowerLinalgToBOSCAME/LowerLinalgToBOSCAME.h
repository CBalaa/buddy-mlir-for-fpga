//===- LowerLinalgToBOSCAME.h - Linalg BOSCAME lowering --------*- C++ -*-===//
//
// Compatibility pass names for the shared contract -> plan -> emit pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H

#include <memory>

namespace mlir {
class Pass;
namespace buddy {
std::unique_ptr<Pass> createLowerLinalgToBOSCAMEPass();
std::unique_ptr<Pass> createLinalgToBOSCAMEPass();
} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H
