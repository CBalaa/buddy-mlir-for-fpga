//===- BOSCAMEBufferNormalization.h - Compatibility buffer forms -*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEBUFFERNORMALIZATION_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEBUFFERNORMALIZATION_H

#include "mlir/IR/BuiltinOps.h"

namespace buddy::boscame {

/// Express supported handwritten buffer generics using ordinary copies,
/// explicit scalar broadcasts and one arithmetic operation per generic.
/// This adapter does not select routes, move boundary operations or change
/// floating evaluation order. Unsupported payloads remain for diagnostics.
void normalizeBOSCAMEBufferInput(mlir::ModuleOp module);

} // namespace buddy::boscame

#endif
