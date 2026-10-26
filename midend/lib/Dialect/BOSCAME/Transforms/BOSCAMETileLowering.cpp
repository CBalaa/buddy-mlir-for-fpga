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
#include "Dialect/BOSCAME/MatrixTransfer.h"
#include "Dialect/BOSCAME/TileOps.h"

#include "BOSCAMEConversionUtils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// No accumulator input or memory read is introduced. The output constraint
/// makes the physical destination explicit to LLVM, and spelling the same
/// output three times enforces the only board-verified subtraction subset.
class TileZeroAccOpConversion : public ConvertToLLVMPattern {
public:
  TileZeroAccOpConversion(const LLVMTypeConverter &converter,
                          MLIRContext *context)
      : ConvertToLLVMPattern(TileZeroAccOp::getOperationName(), context,
                             converter, tilePatternBenefit()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto zero = cast<TileZeroAccOp>(op);
    auto type = cast<VectorType>(zero.getMatrix().getType());
    Type resultType = buddy::boscame::detail::getBOSCAMERegisterType(type);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "unsupported accumulator type");
    SmallVector<Value> arguments;
    // The standalone clear_acc probe uses TYPE_I16=0x21 for both tested
    // widening clear forms, then TYPE_I8=0x10 for signed-i8 MMA.
    for (int64_t value :
         {type.getShape()[0], type.getShape()[1], int64_t(0x21)})
      arguments.push_back(buddy::boscame::detail::createI64Constant(
          rewriter, op->getLoc(), value));
    auto clear = LLVM::InlineAsmOp::create(
        rewriter, op->getLoc(), resultType, arguments,
        "fence rw, rw\n\tmsettilem zero, $1\n\tmsettilen zero, $2\n\t"
        "msettype zero, $3\n\tmwsub.h.mm $0, $0, $0\n\tfence rw, rw",
        "={acc0},r,r,r,~{memory}",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::TailCallKind::None, /*convergent=*/false, LLVM::AsmDialectAttr(),
        ArrayAttr());
    rewriter.replaceOp(op, clear.getResults());
    return success();
  }
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

    if (op->hasAttr("bosc_ame.ordered")) {
      // The independent emitter has one serial accumulator and consumes each
      // A/B pair before loading the next pair. LLVM's existing pure MMA
      // intrinsic cannot express its dependence on msettile*/msettype. A
      // side-effecting assembly bundle keeps configuration and MMA together;
      // its tied accumulator and memory clobber order both surrounding tile
      // loads and the reduction. Named matrix constraints are understood by
      // the existing backend and require no matrix moves for this schedule.
      SmallVector<Value> arguments(operands);
      for (int64_t value : {lhsType.getShape()[0], rhsType.getShape()[1],
                            lhsType.getShape()[1], kFpgaMtypeMmaI8})
        arguments.push_back(buddy::boscame::detail::createI64Constant(
            rewriter, op->getLoc(), value));
      auto ordered = LLVM::InlineAsmOp::create(
          rewriter, op->getLoc(), resultType, arguments,
          "msettilem zero, $4\n\tmsettilen zero, $5\n\t"
          "msettilek zero, $6\n\tmsettype zero, $7\n\t"
          "mqma.b.mm $0, $2, $3",
          "={acc0},0,{tr0},{tr1},r,r,r,r,~{memory}",
          /*has_side_effects=*/true, /*is_align_stack=*/false,
          LLVM::TailCallKind::None, /*convergent=*/false,
          LLVM::AsmDialectAttr(), ArrayAttr());
      rewriter.replaceOp(op, ordered.getResults());
      return success();
    }

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

/// A matrix copy is one indivisible side-effecting operation. The memory
/// clobber preserves its dependencies; the ACC0 clobber is visible to LLVM's
/// allocator. No matrix value or runtime call is introduced between the load
/// and store. The leading fence orders prior CPU/RVV accesses; the final fence
/// completes the store before later DMA/AME consumers.
class MatrixCopyOpConversion : public ConvertToLLVMPattern {
public:
  MatrixCopyOpConversion(const LLVMTypeConverter &converter,
                         MLIRContext *context)
      : ConvertToLLVMPattern(MatrixCopyOp::getOperationName(), context,
                             converter, tilePatternBenefit()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto copy = cast<MatrixCopyOp>(op);
    Location loc = copy.getLoc();
    auto pointer = [&](unsigned index, int64_t offset) -> Value {
      Value base = buddy::boscame::detail::extractPointerFromDescriptor(
          rewriter, loc, operands[index], op->getOperand(index).getType(),
          *getTypeConverter());
      if (!base || !offset)
        return base;
      Value shift =
          buddy::boscame::detail::createI64Constant(rewriter, loc, offset);
      return LLVM::GEPOp::create(rewriter, loc, base.getType(),
                                 rewriter.getI8Type(), base, ValueRange{shift});
    };
    Value source = pointer(0, copy.getSourceOffset());
    Value target = pointer(1, copy.getTargetOffset());
    if (!source || !target)
      return rewriter.notifyMatchFailure(
          op, "cannot decode matrix copy endpoint descriptors");
    SmallVector<Value> arguments{source, target};
    for (int64_t value :
         {copy.getRows(), copy.getRowBytes() / 4, copy.getSourcePitch(),
          copy.getTargetPitch(), copy.getMtype()})
      arguments.push_back(
          buddy::boscame::detail::createI64Constant(rewriter, loc, value));
    LLVM::InlineAsmOp::create(
        rewriter, loc, TypeRange{}, arguments,
        "fence rw, rw\n\tmsettilem zero, $2\n\tmsettilen zero, $3\n\t"
        "msettype zero, $6\n\tmlce32.m acc0, ($0), $4\n\t"
        "msce32.m acc0, ($1), $5\n\tfence rw, rw",
        "r,r,r,r,r,r,r,~{acc0},~{memory}",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::TailCallKind::None, /*convergent=*/false, LLVM::AsmDialectAttr(),
        ArrayAttr());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Matrix value scope
//===----------------------------------------------------------------------===//

/// True when `op` produces an AME matrix value.
bool isTileValueProducer(Operation *op) {
  return isa_and_nonnull<TileLoadAOp, TileLoadBOp, TileLoadAccOp, TileZeroAccOp,
                         TileMmaOp>(op);
}

/// True when `value` is a legal *source* of a matrix value.
///
/// There is still no matrix-value ABI - a function argument, a call result or a
/// value produced by an unrelated operation is rejected - but a matrix value
/// may travel through structured control flow, which is what lets an
/// accumulator stay in the register file across a reduction loop:
///
///   * the result of a `bosc_ame.tile.*` operation,
///   * an `scf.for` region iter-argument (a loop-carried matrix value),
///   * for nested reductions, the result of an `scf.for` that yielded one.
///
/// `origin` is filled with the accepted producer for the diagnostics.
bool isMatrixValueSource(Value value, StringRef &origin) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto forOp =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (forOp && llvm::is_contained(forOp.getRegionIterArgs(), value)) {
      origin = "an `scf.for` region iter-argument";
      return true;
    }
    return false;
  }
  Operation *defining = value.getDefiningOp();
  if (isTileValueProducer(defining)) {
    origin = "a `bosc_ame.tile.*` operation";
    return true;
  }
  if (isa_and_nonnull<scf::ForOp>(defining)) {
    origin = "the result of an `scf.for`";
    return true;
  }
  return false;
}

bool isMatrixValueType(Value value) {
  auto vector = dyn_cast<VectorType>(value.getType());
  return vector && vector.getRank() == 2;
}

/// Also recognize the lower-level one-dimensional scalable matrix spelling,
/// including a value carried through a loop or a temporary conversion cast.
bool isMatrixTransferClobberedValue(Value value, unsigned depth = 0) {
  auto vector = dyn_cast<VectorType>(value.getType());
  if (!vector || depth > 32)
    return false;
  if (vector.getRank() == 2)
    return true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    return loop && argument.getArgNumber() > 0 &&
           isMatrixTransferClobberedValue(
               loop.getInitArgs()[argument.getArgNumber() - 1], depth + 1);
  }
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return false;
  if (producer->getName().getStringRef().starts_with("bosc_ame."))
    return true;
  if (auto loop = dyn_cast<scf::ForOp>(producer))
    return isMatrixTransferClobberedValue(
        loop.getInitArgs()[cast<OpResult>(value).getResultNumber()], depth + 1);
  if (auto conversion = dyn_cast<UnrealizedConversionCastOp>(producer))
    return llvm::any_of(conversion.getOperands(), [&](Value operand) {
      return isMatrixTransferClobberedValue(operand, depth + 1);
    });
  return false;
}

/// The first version has no matrix-value ABI: a matrix value has to come from a
/// tile operation or from an enclosing `scf.for` (a loop-carried accumulator),
/// and it may only feed a `tile.mma`, a `tile.store_acc`, a loop yield or a
/// loop's initial value.
///
/// This is a rule about the *values*, not about registers: which tile or
/// accumulator register holds a value is the backend's decision, and neither
/// the tile operations nor this pass ever name one.
LogicalResult verifyBlockTileValues(Block &block) {
  bool ok = true;
  auto requireMatrixSource = [&](Operation *consumer, Value value,
                                 StringRef what) -> void {
    StringRef origin;
    if (isMatrixValueSource(value, origin))
      return;
    consumer->emitOpError()
        << what << " (" << value
        << ") must be produced by a `bosc_ame.tile.*` operation or carried by "
           "an enclosing `scf.for`; the first version has no matrix-value ABI, "
           "so a matrix value may not enter from a function argument, a call "
           "or an unrelated operation";
    ok = false;
  };

  for (Operation &op : block) {
    if (auto mma = dyn_cast<TileMmaOp>(&op)) {
      requireMatrixSource(&op, mma.getAcc(), "the accumulator operand");
      requireMatrixSource(&op, mma.getLhs(), "the A operand");
      requireMatrixSource(&op, mma.getRhs(), "the B operand");
      continue;
    }
    if (auto store = dyn_cast<TileStoreAccOp>(&op))
      requireMatrixSource(&op, store.getMatrix(), "the stored matrix value");
  }

  // A matrix value that leaves the block: only the structured loop forms may
  // carry it, and the type has to be preserved along the way.  A region
  // terminator that carries nothing - the usual `scf.if` body - is not this
  // rule's business.
  if (block.mightHaveTerminator())
    if (auto yield = dyn_cast<scf::YieldOp>(block.getTerminator())) {
      auto forOp = dyn_cast<scf::ForOp>(block.getParentOp());
      for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) {
        if (!isMatrixValueType(yielded))
          continue;
        if (!forOp) {
          yield.emitOpError()
              << "yields the matrix value " << yielded
              << " from a region that is not an `scf.for`; matrix values only "
                 "cross structured loops";
          ok = false;
          continue;
        }
        if (index >= forOp.getNumRegionIterArgs() ||
            yielded.getType() != forOp.getRegionIterArg(index).getType()) {
          yield.emitOpError()
              << "yields " << yielded.getType() << " at position " << index
              << " but the loop does not carry a matching value";
          ok = false;
          continue;
        }
        requireMatrixSource(yield.getOperation(), yielded,
                            "the yielded matrix value");
      }
    }

  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      for (auto [index, init] : llvm::enumerate(forOp.getInitArgs())) {
        if (!isMatrixValueType(forOp.getRegionIterArg(index)))
          continue;
        if (init.getType() != forOp.getRegionIterArg(index).getType()) {
          forOp.emitOpError()
              << "initializes the loop-carried " << init.getType()
              << " at position " << index << " but carries "
              << forOp.getRegionIterArg(index).getType();
          ok = false;
          continue;
        }
        requireMatrixSource(forOp.getOperation(), init,
                            "the loop-carried initial value");
      }
    }
    if (!isTileValueProducer(&op))
      continue;
    for (Value result : op.getResults())
      for (Operation *user : result.getUsers())
        if (!isa<TileMmaOp, TileStoreAccOp, scf::YieldOp, scf::ForOp>(user)) {
          user->emitOpError()
              << "uses the matrix value " << result
              << " as a regular operand; a matrix value may only feed a "
                 "`bosc_ame.tile.mma`, a `bosc_ame.tile.store_acc` or an "
                 "`scf.for` that carries it";
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
    found |= isa<TileLoadAOp, TileLoadBOp, TileLoadAccOp, TileZeroAccOp,
                 TileMmaOp, TileStoreAccOp, MatrixCopyOp>(op);
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
  // Walk every region, not just the entry block of every function: a matrix
  // value that crosses a loop is checked in the block that actually carries it,
  // and a nested region with a broken matrix value must not be able to hide
  // behind a legal outer block.
  bool ok = true;
  module.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        ok &= succeeded(verifyBlockTileValues(block));
  });
  // A transfer reserves one physical accumulator and changes the global
  // matrix configuration. Reject any live matrix SSA value, including a TR
  // value or an enclosing loop's carried accumulator. Checking enclosing
  // operations also catches a value read before/after an entire nested scope
  // even when it has no direct use inside the copy's own block.
  Liveness liveness(module);
  module.walk([&](MatrixCopyOp copy) {
    for (Operation *position = copy; position != module;
         position = position->getParentOp()) {
      const LivenessBlockInfo *info =
          liveness.getLiveness(position->getBlock());
      if (!info)
        continue;
      for (Value value : info->currentlyLiveValues(position)) {
        Operation *producer = value.getDefiningOp();
        // An enclosing region operation's result does not exist until that
        // region returns. The expansive block-liveness query includes it at
        // the parent operation, but it is not a value crossing this copy.
        if (position != copy && producer &&
            (producer == position || position->isProperAncestor(producer)))
          continue;
        if (!isMatrixTransferClobberedValue(value))
          continue;
        copy.emitOpError()
            << "cannot execute while TR/ACC matrix value " << value
            << " is live; matrix copies reserve ACC0 and reconfigure the "
               "matrix unit, and cannot cross a live or loop-carried matrix "
               "value";
        ok = false;
        return;
      }
    }
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
               TileLoadAccOpConversion, TileZeroAccOpConversion,
               TileMmaOpConversion, TileStoreAccOpConversion,
               MatrixCopyOpConversion>(converter, context);
}
