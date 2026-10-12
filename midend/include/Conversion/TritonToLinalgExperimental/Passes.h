//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES_H

#include "Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "Conversion/TritonToLinalgExperimental/TritonToPtr.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "Conversion/TritonToLinalgExperimental/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
