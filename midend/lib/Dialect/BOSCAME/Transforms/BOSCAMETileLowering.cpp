//===- BOSCAMETileLowering.cpp - Tile ops -> FPGA intrinsics --------------===//
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
// `bosc_ame.tile.*` -> `bosc_ame.intr.*`, in one file, sharing the
// LLVMTypeConverter and the conversion target of the unified BOSCAME pass.
//
// The lowering owns everything the high-level kernel deliberately does not say:
//
//   * the hardware *configuration*.  A tile load/store/mma carries shape and
//     element type through its operands and results, so the pattern emits the
//     register-form `msettilem/m/n/k` and `msettype` intrinsics it needs right
//     before the instruction.  Configuration is repeated on purpose: the first
//     version favours "the hardware state is always correct here" over a
//     configuration-elimination optimization, and a load must never depend on
//     an `msettype` that only a later MMA would have executed.
//   * the *byte stride*.  Derived from the memref layout, with the physical row
//     of the operand (leading stride for A/ACC, trailing stride for B) folded
//     into bytes - never blindly `strides[0]`.
//   * the *effective pointer*.  Decoded from the converted memref descriptor
//     once (`alignedPtr + offset * elementBytes`), so a subview offset is
//     neither lost nor applied twice.
//
// Register allocation is *not* part of this lowering: the patterns emit the
// same `bosc_ame.intr.*` operations the rest of the dialect uses, and the
// backend decides which tile and accumulator registers hold the values.  The
// only value-level rule kept here is the first version's lack of a
// matrix-value ABI (see `verifyBlockTileValues`).
//
// The patterns are added on top of the ones `-lower-bosc-ame` already
// registers; they never re-implement memref, func, arith or vector conversion.
//
//===----------------------------------------------------------------------===//

#include "Dialect/BOSCAME/Transforms/BOSCAMETileLowering.h"
#include "Dialect/BOSCAME/BOSCAMEAttrs.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/TileOps.h"

#include "BOSCAMEConversionUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace buddy::boscame;

namespace {

//===----------------------------------------------------------------------===//
// Hardware configuration
//===----------------------------------------------------------------------===//

/// Emit the register-form configuration of one datapath phase.  `m`, `n` and
/// `k` are the tile extents the following instruction depends on; a `nullopt`
/// extent is not programmed, because an unrelated extent of the previous
/// instruction is still valid hardware state.
void emitTileConfiguration(ConversionPatternRewriter &rewriter, Location loc,
                           std::optional<int64_t> m, std::optional<int64_t> n,
                           std::optional<int64_t> k, int64_t mtype) {
  if (m)
    buddy::boscame::detail::createConfigIntrinsic(rewriter, loc, "msettilem",
                                                  *m);
  if (n)
    buddy::boscame::detail::createConfigIntrinsic(rewriter, loc, "msettilen",
                                                  *n);
  if (k)
    buddy::boscame::detail::createConfigIntrinsic(rewriter, loc, "msettilek",
                                                  *k);
  buddy::boscame::detail::createConfigIntrinsic(rewriter, loc, "msettype",
                                                mtype);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Pattern benefit above the generic BOSCAME-to-intrinsic pattern, which would
/// otherwise fabricate a `bosc_ame.intr.tile.load_a` that does not exist.
PatternBenefit tilePatternBenefit() { return PatternBenefit(3); }

/// Shared part of the three loads: decode the SPM descriptor, emit the
/// configuration the load depends on and replace the tile value by the
/// intrinsic that selects the AME load instruction.
class TileLoadPattern : public ConvertToLLVMPattern {
public:
  TileLoadPattern(StringRef opName, StringRef intrinsicName, TileRole role,
                  const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(opName, context, converter, tilePatternBenefit()),
        intrinsicName(intrinsicName.str()), role(role) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto source = op->getOperand(0);
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    std::optional<TileStorageLayout> layout =
        computeTileStorageLayout(sourceType, role);
    if (!layout)
      return rewriter.notifyMatchFailure(
          op, "the SPM operand has no layout the AME load can address");
    Type resultType = buddy::boscame::detail::getBOSCAMERegisterType(
        op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "unsupported matrix value type");

    Location loc = op->getLoc();
    Value base = buddy::boscame::detail::extractPointerFromDescriptor(
        rewriter, loc, operands[0], sourceType, *getTypeConverter());
    if (!base)
      return rewriter.notifyMatchFailure(
          op, "cannot decode the SPM memref descriptor");

    // The load reads whatever the tile registers currently describe, so its
    // shape and its element type are programmed first - in this order.
    if (role == TileRole::B)
      emitTileConfiguration(rewriter, loc, std::nullopt, layout->cols(),
                            layout->rows(), kFpgaMtypeMmaI8);
    else
      emitTileConfiguration(
          rewriter, loc, layout->rows(),
          role == TileRole::Accumulator ? layout->cols()
                                        : std::optional<int64_t>(),
          role == TileRole::A ? layout->cols() : std::optional<int64_t>(),
          role == TileRole::Accumulator ? kFpgaMtypeAccumulatorI32
                                        : kFpgaMtypeMmaI8);

    Value stride = buddy::boscame::detail::createI64Constant(
        rewriter, loc, layout->rowStrideBytes);
    Operation *intrinsic = buddy::boscame::detail::createBOSCAMEIntrinsicOp(
        rewriter, loc, intrinsicName, TypeRange{resultType},
        ValueRange{base, stride});
    rewriter.replaceOp(op, intrinsic->getResults());
    return success();
  }

private:
  std::string intrinsicName;
  TileRole role;
};

/// `bosc_ame.tile.load_a` -> `bosc_ame.intr.mlae8.m`.
class TileLoadAOpConversion : public TileLoadPattern {
public:
  TileLoadAOpConversion(const LLVMTypeConverter &converter,
                        MLIRContext *context)
      : TileLoadPattern(TileLoadAOp::getOperationName(), "bosc_ame.mlae8.m",
                        TileRole::A, converter, context) {}
};

/// `bosc_ame.tile.load_b` -> `bosc_ame.intr.mlbe8.m`.
///
/// The stride is the *physical* row stride of the operand.  B's storage is an
/// `N x K` buffer with K contiguous, described logically as `[K, N]` with
/// `strides[0] == 1`; the physical row pitch is therefore `strides[1]`, which
/// `TileStorageLayout::rowStrideBytes` already selects for the B role.
class TileLoadBOpConversion : public TileLoadPattern {
public:
  TileLoadBOpConversion(const LLVMTypeConverter &converter,
                        MLIRContext *context)
      : TileLoadPattern(TileLoadBOp::getOperationName(), "bosc_ame.mlbe8.m",
                        TileRole::B, converter, context) {}
};

/// `bosc_ame.tile.load_acc` -> `bosc_ame.intr.mlce32.m`.
class TileLoadAccOpConversion : public TileLoadPattern {
public:
  TileLoadAccOpConversion(const LLVMTypeConverter &converter,
                          MLIRContext *context)
      : TileLoadPattern(TileLoadAccOp::getOperationName(), "bosc_ame.mlce32.m",
                        TileRole::Accumulator, converter, context) {}
};

/// `bosc_ame.tile.mma` -> `bosc_ame.intr.mqma.b.mm`.
///
/// The accumulator is carried by SSA, so the MMA itself only has to guarantee
/// that the tile shape and the compute datapath are programmed; which tile and
/// accumulator registers hold the operands is the backend's decision.
class TileMmaOpConversion : public ConvertToLLVMPattern {
public:
  TileMmaOpConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(TileMmaOp::getOperationName(), context, converter,
                             tilePatternBenefit()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mma = cast<TileMmaOp>(op);
    auto lhsType = dyn_cast<VectorType>(mma.getLhs().getType());
    auto rhsType = dyn_cast<VectorType>(mma.getRhs().getType());
    if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "unsupported tile shapes");

    Type resultType = buddy::boscame::detail::getBOSCAMERegisterType(
        mma.getMatrix().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "unsupported accumulator type");

    // M and N come from A and B, K is their shared inner dimension.
    emitTileConfiguration(rewriter, op->getLoc(), lhsType.getShape()[0],
                          rhsType.getShape()[1], lhsType.getShape()[1],
                          kFpgaMtypeMmaI8);

    Operation *intrinsic = buddy::boscame::detail::createBOSCAMEIntrinsicOp(
        rewriter, op->getLoc(), "bosc_ame.mqma.b.mm", TypeRange{resultType},
        operands);
    rewriter.replaceOp(op, intrinsic->getResults());
    return success();
  }
};

/// `bosc_ame.tile.store_acc` -> `bosc_ame.intr.msce32.m`.
///
/// The write-back stores raw i32, so the accumulator phase of `mtype` is
/// programmed first and the row stride is the leading stride in bytes.
class TileStoreAccOpConversion : public ConvertToLLVMPattern {
public:
  TileStoreAccOpConversion(const LLVMTypeConverter &converter,
                           MLIRContext *context)
      : ConvertToLLVMPattern(TileStoreAccOp::getOperationName(), context,
                             converter, tilePatternBenefit()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto store = cast<TileStoreAccOp>(op);
    auto destType = dyn_cast<MemRefType>(store.getDest().getType());
    std::optional<TileStorageLayout> layout =
        computeTileStorageLayout(destType, TileRole::Accumulator);
    if (!layout)
      return rewriter.notifyMatchFailure(
          op, "the SPM operand has no layout the AME store can address");

    Location loc = op->getLoc();
    Value base = buddy::boscame::detail::extractPointerFromDescriptor(
        rewriter, loc, operands[1], destType, *getTypeConverter());
    if (!base)
      return rewriter.notifyMatchFailure(
          op, "cannot decode the SPM memref descriptor");

    emitTileConfiguration(rewriter, loc, layout->rows(), layout->cols(),
                          std::nullopt, kFpgaMtypeAccumulatorI32);
    Value stride = buddy::boscame::detail::createI64Constant(
        rewriter, loc, layout->rowStrideBytes);
    buddy::boscame::detail::createBOSCAMEIntrinsicOp(
        rewriter, loc, "bosc_ame.msce32.m", TypeRange(),
        ValueRange{operands[0], base, stride});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Matrix value scope
//===----------------------------------------------------------------------===//

/// True when `op` produces an AME matrix value.
bool isTileValueProducer(Operation *op) {
  return isa_and_nonnull<TileLoadAOp, TileLoadBOp, TileLoadAccOp, TileMmaOp>(
      op);
}

/// The first version has no matrix-value ABI: a matrix value has to be produced
/// by a tile operation in the same block, and it may only feed a `tile.mma` or
/// a `tile.store_acc`.
///
/// This is a rule about the *values*, not about registers: which tile or
/// accumulator register holds a value is the backend's decision, and neither
/// the tile operations nor this pass ever name one.
LogicalResult verifyBlockTileValues(Block &block) {
  bool ok = true;
  for (Operation &op : block) {
    auto requireTileValue = [&](Value value, StringRef what) -> void {
      if (!isTileValueProducer(value.getDefiningOp())) {
        op.emitOpError()
            << what << " (" << value
            << ") must be produced by a `bosc_ame.tile.*` operation; the first "
               "version has no matrix-value ABI, so a matrix value has to come "
               "from a tile load, a tile load_acc or a tile.mma";
        ok = false;
        return;
      }
      if (value.getDefiningOp()->getBlock() != &block) {
        op.emitOpError()
            << what << " (" << value
            << ") is defined in another block; the first version has no "
               "matrix-value ABI and does not carry a matrix value across "
               "control flow";
        ok = false;
      }
    };

    if (auto mma = dyn_cast<TileMmaOp>(&op)) {
      requireTileValue(mma.getAcc(), "the accumulator operand");
      requireTileValue(mma.getLhs(), "the A operand");
      requireTileValue(mma.getRhs(), "the B operand");
      continue;
    }
    if (auto store = dyn_cast<TileStoreAccOp>(&op))
      requireTileValue(store.getMatrix(), "the stored matrix value");
  }

  for (Operation &op : block) {
    if (!isTileValueProducer(&op))
      continue;
    for (Value result : op.getResults())
      for (Operation *user : result.getUsers())
        if (!isa<TileMmaOp, TileStoreAccOp>(user)) {
          user->emitOpError()
              << "uses the matrix value " << result
              << " as a regular operand; a matrix value may only feed a "
                 "`bosc_ame.tile.mma` or a `bosc_ame.tile.store_acc` in the "
                 "first version";
          ok = false;
        }
  }
  return success(ok);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

bool buddy::boscame::hasTileOps(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    found |=
        isa<TileLoadAOp, TileLoadBOp, TileLoadAccOp, TileMmaOp, TileStoreAccOp>(
            op);
  });
  return found;
}

LogicalResult
buddy::boscame::resolveTileTargetProfile(ModuleOp module,
                                         TileTargetProfile &profile) {
  auto target = module->getAttrOfType<StringAttr>(kAmeTargetAttrName);
  if (!target)
    return module.emitError()
           << "the module uses `bosc_ame.tile.*` operations, which lower to "
              "the FPGA AME intrinsic set, but does not select an AME hardware "
              "contract; add `"
           << kAmeTargetAttrName << " = \"" << kAmeTargetNrFpga
           << "\"` to the module (FPGA semantics are never selected "
              "implicitly)";

  if (target.getValue() == kAmeTargetNrFpga) {
    profile = TileTargetProfile::NrFpga;
    return success();
  }
  return module.emitError()
         << "`" << kAmeTargetAttrName << " = \"" << target.getValue()
         << "\"` has no verified tile datapath in this version; the "
            "implemented contract is \""
         << kAmeTargetNrFpga
         << "\" (register-form configuration and raw i32 accumulator memory)";
}

LogicalResult buddy::boscame::verifyBOSCAMETileValues(ModuleOp module) {
  bool ok = true;
  module.walk([&](func::FuncOp function) {
    for (Block &block : function.getBody())
      ok &= succeeded(verifyBlockTileValues(block));
  });
  return success(ok);
}

void buddy::boscame::populateBOSCAMETileLoweringPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    const TileLoweringOptions &options) {
  assert(options.profile == TileTargetProfile::NrFpga &&
         "only the nr-fpga tile datapath is implemented");
  (void)options;
  MLIRContext *context = patterns.getContext();
  patterns.add<TileLoadAOpConversion, TileLoadBOpConversion,
               TileLoadAccOpConversion, TileMmaOpConversion,
               TileStoreAccOpConversion>(converter, context);
}
