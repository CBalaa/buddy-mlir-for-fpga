//===- TileOps.cpp - SPM <-> AME tile operation contracts -----------------===//
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
// Verifiers and layout rules of the `bosc_ame.tile.*` operations.  Keeping the
// rules in one translation unit (rather than in each op verifier) is what lets
// the LLVM lowering reuse exactly the layout the verifier accepted: the
// lowering only re-derives the strides and the row stride, and never has to
// repeat a check that could drift from the verifier's.
//
//===----------------------------------------------------------------------===//

#include "Dialect/BOSCAME/TileOps.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace buddy::boscame;

StringRef buddy::boscame::stringifyTileRole(TileRole role) {
  switch (role) {
  case TileRole::A:
    return "A";
  case TileRole::B:
    return "B";
  case TileRole::Accumulator:
    return "accumulator";
  }
  llvm_unreachable("unknown tile role");
}

std::optional<unsigned>
buddy::boscame::getTileElementByteWidth(TileRole role, Type elementType) {
  auto integer = dyn_cast<IntegerType>(elementType);
  if (!integer)
    return std::nullopt;
  switch (role) {
  case TileRole::A:
  case TileRole::B:
    // Signed i8 x i8 is the datapath this dialect implements today.  The AME
    // has wider tile loads/stores, but accepting them here would pick an mtype
    // phase and a register file that are not wired up.
    return integer.getWidth() == 8 ? std::optional<unsigned>(1) : std::nullopt;
  case TileRole::Accumulator:
    // Raw i32 accumulator memory: the exact representation the AME
    // accumulator stores.
    return integer.getWidth() == 32 ? std::optional<unsigned>(4) : std::nullopt;
  }
  llvm_unreachable("unknown tile role");
}

/// Human-readable description of what a role accepts, for diagnostics.
static StringRef getTileRoleRequirement(TileRole role) {
  switch (role) {
  case TileRole::A:
    return "a memref<MxKxi8> with K contiguous (strides[1] == 1) and a row "
           "stride of at least K (strides[0] >= K)";
  case TileRole::B:
    return "a memref<KxNxi8> describing the physical NxK B buffer, i.e. with "
           "K contiguous inside one physical row (strides[0] == 1) and a row "
           "pitch of at least K (strides[1] >= K)";
  case TileRole::Accumulator:
    return "a memref<MxNxi32> with N contiguous (strides[1] == 1) and a row "
           "stride of at least N (strides[0] >= N)";
  }
  llvm_unreachable("unknown tile role");
}

std::optional<TileStorageLayout>
buddy::boscame::computeTileStorageLayout(MemRefType type, TileRole role) {
  if (!type || type.getRank() != 2 || !type.hasStaticShape())
    return std::nullopt;

  std::optional<unsigned> elementBytes =
      getTileElementByteWidth(role, type.getElementType());
  if (!elementBytes)
    return std::nullopt;

  llvm::SmallVector<int64_t, 2> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (offset == ShapedType::kDynamic || strides[0] == ShapedType::kDynamic ||
      strides[1] == ShapedType::kDynamic)
    return std::nullopt;
  if (offset < 0 || strides[0] <= 0 || strides[1] <= 0)
    return std::nullopt;

  TileStorageLayout layout;
  layout.shape = {type.getShape()[0], type.getShape()[1]};
  layout.strides = {strides[0], strides[1]};
  layout.elementBytes = *elementBytes;
  layout.byteOffset = offset * layout.elementBytes;

  // The hardware row stride is the *physical* row distance.  For B, whose
  // storage is `N x K` with K contiguous, that is the stride between two
  // physical rows of the staged buffer, i.e. the logical trailing stride.
  const int rowIndex = tileRowStrideIndex(role);
  const int innerIndex = 1 - rowIndex;
  // The inner dimension must be the contiguous one, and the row pitch must not
  // make two physical rows overlap.
  if (strides[innerIndex] != 1)
    return std::nullopt;
  if (strides[rowIndex] < layout.shape[innerIndex])
    return std::nullopt;
  layout.rowStrideBytes = strides[rowIndex] * layout.elementBytes;
  return layout;
}

LogicalResult buddy::boscame::verifyTileStorageLayout(
    Operation *op, MemRefType type, TileRole role, TileStorageLayout &layout) {
  if (!type)
    return op->emitOpError() << "operand must be a ranked memref in "
                                "`#bosc_ame.spm<bank>`, got an "
                                "unranked memref";

  if (!isa_and_nonnull<SPMAttr>(type.getMemorySpace()))
    return op->emitOpError()
           << "operand " << type
           << " must carry the `#bosc_ame.spm<bank>` storage space; a DDR "
              "buffer has no SPM address for the AME "
           << stringifyTileRole(role) << " load/store";

  if (type.getRank() != 2)
    return op->emitOpError() << "operand " << type
                             << " must be a 2-D memref (the first version "
                                "supports static 2-D tiles only), got rank "
                             << type.getRank();

  if (!type.hasStaticShape())
    return op->emitOpError()
           << "operand " << type
           << " must have a static shape: the tile configuration is compiled "
              "into the instruction sequence";

  if (!getTileElementByteWidth(role, type.getElementType()))
    return op->emitOpError()
           << "operand element type " << type.getElementType() << " is not "
           << "supported for the " << stringifyTileRole(role)
           << " tile; the first version implements signed i8 A/B tiles and "
              "i32 accumulators";

  for (int64_t dim : type.getShape())
    if (dim < 1 || dim > kAmeMaxTileDim)
      return op->emitOpError() << "tile dimension " << dim << " of " << type
                               << " is outside the AME tile capacity [1, "
                               << kAmeMaxTileDim << "]";

  std::optional<TileStorageLayout> computed =
      computeTileStorageLayout(type, role);
  if (!computed)
    return op->emitOpError()
           << "operand " << type << " has a layout the AME "
           << stringifyTileRole(role)
           << " load/store cannot address; the hardware needs "
           << getTileRoleRequirement(role);

  layout = *computed;
  return success();
}

LogicalResult buddy::boscame::verifyTileMemRefOperand(
    Operation *op, Value operand, TileRole role, TileStorageLayout &layout) {
  auto type = dyn_cast<MemRefType>(operand.getType());
  if (!type)
    return op->emitOpError()
           << "operand " << operand << " must be a ranked memref, got "
           << operand.getType();
  return verifyTileStorageLayout(op, type, role, layout);
}

LogicalResult
buddy::boscame::verifyTileMatrixValue(Operation *op, Value value, TileRole role,
                                      std::array<int64_t, 2> &shape) {
  auto vector = dyn_cast<VectorType>(value.getType());
  if (!vector)
    return op->emitOpError() << "matrix value " << value
                             << " must be a 2-D vector (BOSCAME_MatrixType), "
                                "got "
                             << value.getType();

  if (vector.getRank() != 2)
    return op->emitOpError()
           << "matrix value " << value << " must be a 2-D vector with the real "
           << "logical tile shape, got rank " << vector.getRank();

  if (vector.isScalable())
    return op->emitOpError()
           << "matrix value " << value
           << " must be a fixed 2-D vector at this level; the scalable "
              "register type is chosen by the lowering";

  if (!getTileElementByteWidth(role, vector.getElementType()))
    return op->emitOpError()
           << "matrix element type " << vector.getElementType() << " is not "
           << "supported for the " << stringifyTileRole(role)
           << " tile; the first version implements signed i8 A/B tiles and "
              "i32 accumulators";

  shape = {vector.getShape()[0], vector.getShape()[1]};
  for (int64_t dim : shape)
    if (dim < 1 || dim > kAmeMaxTileDim)
      return op->emitOpError()
             << "matrix tile dimension " << dim << " of " << value
             << " is outside the AME tile capacity [1, " << kAmeMaxTileDim
             << "]";
  return success();
}

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

/// Shape and element type of the memory operand and of the matrix value must
/// describe the same logical tile.
static LogicalResult verifyLoadShape(Operation *op, TileRole role,
                                     const TileStorageLayout &layout,
                                     ArrayRef<int64_t> shape) {
  if (layout.shape[0] != shape[0] || layout.shape[1] != shape[1])
    return op->emitOpError()
           << "the " << stringifyTileRole(role) << " memref shape ["
           << layout.shape[0] << ", " << layout.shape[1]
           << "] does not match the matrix shape [" << shape[0] << ", "
           << shape[1]
           << "]; logical and physical tile descriptions must agree";
  return success();
}

LogicalResult TileLoadAOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getSource(), TileRole::A, layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::A, shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::A, layout, shape);
}

LogicalResult TileLoadBOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getSource(), TileRole::B, layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::B, shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::B, layout, shape);
}

LogicalResult TileLoadAccOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getSource(), TileRole::Accumulator,
                                     layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::Accumulator,
                                   shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::Accumulator, layout, shape);
}

LogicalResult TileStoreAccOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getDest(), TileRole::Accumulator,
                                     layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::Accumulator,
                                   shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::Accumulator, layout, shape);
}

LogicalResult TileMmaOp::verify() {
  std::array<int64_t, 2> lhsShape, rhsShape, accShape, resultShape;
  if (failed(verifyTileMatrixValue(*this, getLhs(), TileRole::A, lhsShape)))
    return failure();
  if (failed(verifyTileMatrixValue(*this, getRhs(), TileRole::B, rhsShape)))
    return failure();
  if (failed(verifyTileMatrixValue(*this, getAcc(), TileRole::Accumulator,
                                   accShape)))
    return failure();
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::Accumulator,
                                   resultShape)))
    return failure();

  // The accumulator dependency is explicit: C_in and C_out are the same value
  // type, so a chain of MMAs cannot silently change shape or element type.
  if (getAcc().getType() != getMatrix().getType())
    return emitOpError() << "the accumulator operand type "
                         << getAcc().getType() << " and the result type "
                         << getMatrix().getType()
                         << " must be identical; a tile.mma returns the "
                            "updated accumulator";

  if (accShape != resultShape)
    return emitOpError() << "accumulator shape [" << accShape[0] << ", "
                         << accShape[1] << "] and result shape ["
                         << resultShape[0] << ", " << resultShape[1]
                         << "] must agree";

  if (lhsShape[1] != rhsShape[0])
    return emitOpError() << "the inner dimension of A (K = " << lhsShape[1]
                         << ") and the outer dimension of B (K = "
                         << rhsShape[0] << ") must agree";

  if (accShape[0] != lhsShape[0] || accShape[1] != rhsShape[1])
    return emitOpError() << "C must be an M x N tile with M = " << lhsShape[0]
                         << " (from A) and N = " << rhsShape[1]
                         << " (from B), but has shape [" << accShape[0] << ", "
                         << accShape[1] << "]";

  return success();
}

//===----------------------------------------------------------------------===//
// Memory effects
//===----------------------------------------------------------------------===//

void TileLoadAOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

void TileLoadBOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

void TileLoadAccOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

void TileStoreAccOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable());
}
