#ifndef MLIR_DIALECT_TRITON_STRUCTURED_IR_TRITON_STRUCTURED_DIALECT_H_
#define MLIR_DIALECT_TRITON_STRUCTURED_IR_TRITON_STRUCTURED_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "Triton/IR/Dialect.h"

namespace mlir {
namespace tts {
namespace utils {
mlir::Value getScalarValue(mlir::Value operand, mlir::Location loc,
                           mlir::OpBuilder &builder);
}
} // namespace tts
} // namespace mlir

//===----------------------------------------------------------------------===//
// TritonStructured Operations
//===----------------------------------------------------------------------===//
#include "TritonStructured/IR/TritonStructuredDialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES
#include "TritonStructured/IR/TritonStructuredOps.h.inc"

#endif
