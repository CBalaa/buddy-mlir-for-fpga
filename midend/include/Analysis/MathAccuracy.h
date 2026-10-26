//===- MathAccuracy.h - Target-independent mathematical accuracy
//-----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_MATHACCURACY_H
#define BUDDY_ANALYSIS_MATHACCURACY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace buddy::contract {

/// Accuracy is an explicit promise by the caller, scoped to one function.
/// Absence of a promise never authorizes an approximate implementation.
/// For finite reference results the permitted error is abs + rel * |result|.
/// all-f32 includes nonfinite inputs; ieee requires the mathematical special
/// values and gradual underflow under round-to-nearest, ties-to-even. NaN
/// payloads and floating-point exception flags are outside this value contract.
inline mlir::DictionaryAttr mathAccuracy(mlir::Operation &operation) {
  auto function = operation.getParentOfType<mlir::func::FuncOp>();
  auto accuracy =
      function ? function->getAttrOfType<mlir::DictionaryAttr>("contract.math")
               : mlir::DictionaryAttr();
  return accuracy ? accuracy.getAs<mlir::DictionaryAttr>(
                        operation.getName().stripDialect())
                  : mlir::DictionaryAttr();
}

inline bool permitsMathError(mlir::Operation &operation, double absolute,
                             double relative) {
  auto accuracy = mathAccuracy(operation);
  auto abs = accuracy ? accuracy.getAs<mlir::FloatAttr>("max_abs_error")
                      : mlir::FloatAttr();
  auto rel = accuracy ? accuracy.getAs<mlir::FloatAttr>("max_rel_error")
                      : mlir::FloatAttr();
  return abs && rel && abs.getValueAsDouble() >= absolute &&
         rel.getValueAsDouble() >= relative;
}

} // namespace buddy::contract

#endif // BUDDY_ANALYSIS_MATHACCURACY_H
