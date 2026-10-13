//===- DmaAlignment.h - Provable 64 B alignment of DMA endpoints -*- C++
//-*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Where the effective address of a DMA endpoint comes from, and what the
// compiler has to be able to *prove* about it.
//
// `bosc_ame.dma.start` transfers a byte range whose source and destination must
// both be 64 B aligned.  Two of the three contributions to an endpoint address
// are static and visible in the memref type:
//
//   * the storage root: bank 0/1 is anchored at a 64 B aligned bank base, and a
//     DDR pointer argument is covered by `bosc_ame.dma.abi` (the caller's
//     promise), and
//   * the descriptor offset, which `getDmaEndpoint` already rejects unless it
//     is static and aligned.
//
// The third one is a *`memref.view` byte shift*: `memref.view %root[%shift]` is
// the only operation in the view chain that moves the base pointer while
// leaving the descriptor offset at zero, which is exactly what a regular
// dynamic address (a row of a tile, a K sub-block) needs.  A `view` shift is
// therefore not a property of the type; the endpoint can only be accepted when
// every shift on the chain is either static and aligned or a runtime value the
// compiler can prove to be a multiple of the 64 B granularity.
//
// `isProvablyMultipleOf` is that proof, kept deliberately small: it walks the
// integer expression that produces the shift and only accepts constructions
// whose divisibility follows from the operation itself (constants, addition,
// multiplication by a divisible factor, widening casts, `select`/`min`/`max` of
// divisible values and an `scf.for` induction variable whose lower bound and
// step are divisible).  Anything else - a raw function argument, a load, a
// division - is *not* provable, and the caller then has to fall back to an
// ordinary copy instead of issuing a transfer whose alignment is a guess.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_DMAALIGNMENT_H
#define BUDDY_DIALECT_BOSCAME_DMAALIGNMENT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace buddy {
namespace boscame {
namespace spmdma {

/// True when `value` is provably an exact multiple of `factor`.
///
/// Never optimistic: every `false` only makes the caller use the ordinary copy
/// path, while a wrong `true` would emit a misaligned hardware transfer.
inline bool isProvablyMultipleOf(mlir::Value value, int64_t factor) {
  if (factor <= 0 || !value)
    return false;

  // An `scf.for` induction variable is `lowerBound + i * step`, so it is a
  // multiple of `factor` when both are.
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    auto forOp = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
        argument.getOwner()->getParentOp());
    if (!forOp || forOp.getInductionVar() != value)
      return false;
    return isProvablyMultipleOf(forOp.getLowerBound(), factor) &&
           isProvablyMultipleOf(forOp.getStep(), factor);
  }

  mlir::Operation *defining = value.getDefiningOp();
  if (!defining)
    return false;

  if (auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(defining)) {
    if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt() % factor == 0;
    return false;
  }
  if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(defining))
    return isProvablyMultipleOf(add.getLhs(), factor) &&
           isProvablyMultipleOf(add.getRhs(), factor);
  if (auto sub = llvm::dyn_cast<mlir::arith::SubIOp>(defining))
    return isProvablyMultipleOf(sub.getLhs(), factor) &&
           isProvablyMultipleOf(sub.getRhs(), factor);
  if (auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(defining)) {
    if (isProvablyMultipleOf(mul.getLhs(), factor) ||
        isProvablyMultipleOf(mul.getRhs(), factor))
      return true;
    // `a * c` is a multiple of `factor` when the constant factor divides it
    // and the other operand carries the remaining divisibility: a row offset
    // that is a 16-element multiple of i32 is a 64-byte multiple.
    if (std::optional<int64_t> constant = getConstantIntValue(mul.getRhs()))
      if (*constant != 0 && factor % *constant == 0)
        return isProvablyMultipleOf(mul.getLhs(), factor / *constant);
    if (std::optional<int64_t> constant = getConstantIntValue(mul.getLhs()))
      if (*constant != 0 && factor % *constant == 0)
        return isProvablyMultipleOf(mul.getRhs(), factor / *constant);
    return false;
  }
  if (auto indexCast = llvm::dyn_cast<mlir::arith::IndexCastOp>(defining))
    return isProvablyMultipleOf(indexCast.getIn(), factor);
  if (auto ext = llvm::dyn_cast<mlir::arith::ExtSIOp>(defining))
    return isProvablyMultipleOf(ext.getIn(), factor);
  if (auto ext = llvm::dyn_cast<mlir::arith::ExtUIOp>(defining))
    return isProvablyMultipleOf(ext.getIn(), factor);
  if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(defining))
    return isProvablyMultipleOf(select.getTrueValue(), factor) &&
           isProvablyMultipleOf(select.getFalseValue(), factor);
  if (auto min = llvm::dyn_cast<mlir::arith::MinSIOp>(defining))
    return isProvablyMultipleOf(min.getLhs(), factor) &&
           isProvablyMultipleOf(min.getRhs(), factor);
  if (auto min = llvm::dyn_cast<mlir::arith::MinUIOp>(defining))
    return isProvablyMultipleOf(min.getLhs(), factor) &&
           isProvablyMultipleOf(min.getRhs(), factor);
  if (auto max = llvm::dyn_cast<mlir::arith::MaxSIOp>(defining))
    return isProvablyMultipleOf(max.getLhs(), factor) &&
           isProvablyMultipleOf(max.getRhs(), factor);
  if (auto max = llvm::dyn_cast<mlir::arith::MaxUIOp>(defining))
    return isProvablyMultipleOf(max.getLhs(), factor) &&
           isProvablyMultipleOf(max.getRhs(), factor);
  // A constant-folded `affine.apply` / `arith.muli` chain is covered above; a
  // division, a load, a call result and everything else is not provable.
  return false;
}

/// Walk the view chain of a DMA endpoint and check that the base pointer it
/// yields is `alignmentBytes` aligned, given that the storage root is.
///
/// Only the operations that really move the base pointer need a proof:
/// `memref.view` shifts the pointer by its `byte_shift` operand and keeps the
/// descriptor offset at zero.  Everything else in the chain either keeps the
/// pointer (`memref.cast`, `memref.memory_space_cast`) or is already rejected
/// by the endpoint type check (`memref.subview`/`memref.reinterpret_cast`
/// encode their offset in the type, which `getDmaEndpoint` requires to be
/// static and aligned).
inline mlir::LogicalResult
verifyEndpointPointerAlignment(mlir::Operation *op, mlir::Value endpoint,
                               llvm::StringRef what, int64_t alignmentBytes) {
  mlir::Value current = endpoint;
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (mlir::Operation *defining = current.getDefiningOp()) {
    if (!visited.insert(defining).second)
      return op->emitOpError()
             << what << " endpoint " << current
             << " has a cyclic view chain; the alignment of its effective "
                "address cannot be established";
    if (auto view = llvm::dyn_cast<mlir::memref::ViewOp>(defining)) {
      mlir::Value shift = view.getByteShift();
      std::optional<int64_t> constant = mlir::getConstantIntValue(shift);
      if (constant) {
        if (*constant % alignmentBytes != 0)
          return op->emitOpError()
                 << what << " endpoint " << current
                 << " shifts the base pointer by " << *constant
                 << " bytes, which is not a multiple of the " << alignmentBytes
                 << "-byte DMA granularity";
      } else if (!isProvablyMultipleOf(shift, alignmentBytes)) {
        return op->emitOpError()
               << what << " endpoint " << current
               << " shifts the base pointer by the runtime value " << shift
               << ", which cannot be proven to be a multiple of the "
               << alignmentBytes
               << "-byte DMA granularity; use an ordinary copy for this range "
                  "or make the shift provably aligned";
      }
      current = view.getSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::MemorySpaceCastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    if (auto subview = llvm::dyn_cast<mlir::memref::SubViewOp>(defining)) {
      current = subview.getSource();
      continue;
    }
    if (auto reinterpret =
            llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(defining)) {
      current = reinterpret.getSource();
      continue;
    }
    if (auto collapse =
            llvm::dyn_cast<mlir::memref::CollapseShapeOp>(defining)) {
      current = collapse.getSrc();
      continue;
    }
    if (auto expand = llvm::dyn_cast<mlir::memref::ExpandShapeOp>(defining)) {
      current = expand.getSrc();
      continue;
    }
    // `memref.alloc`/`memref.alloca` with an alignment attribute, a storage
    // root (block argument, `bosc_ame.spm.view`) or anything else: the root
    // alignment is the caller's/allocator's guarantee, not a view shift.
    break;
  }
  return mlir::success();
}

} // namespace spmdma
} // namespace boscame
} // namespace buddy

#endif // BUDDY_DIALECT_BOSCAME_DMAALIGNMENT_H
