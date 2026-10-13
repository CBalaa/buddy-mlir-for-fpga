//===----------------------------------------------------------------------===//
//
// TTIR -> BOSCAME: the entry point of the FPGA AME matmul conversion.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOBOSCAME_TRITONTOBOSCAME_H
#define TRITON_CONVERSION_TRITONTOBOSCAME_TRITONTOBOSCAME_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToBOSCAMEPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOBOSCAME_TRITONTOBOSCAME_H
