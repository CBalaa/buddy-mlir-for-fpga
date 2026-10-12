//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES_H
#define UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES_H

#include "Conversion/UnstructuredToMemref/UnstructuredToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "Conversion/UnstructuredToMemref/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
