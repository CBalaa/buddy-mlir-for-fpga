//====- BOSCAMEDialect.cpp - MLIR BOSCAME dialect implementation ----------===//
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

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/SpmDma.h"

using namespace mlir;
using namespace buddy::boscame;
using namespace buddy::boscame::spmdma;

#define GET_ATTRDEF_CLASSES
#include "BOSCAME/BOSCAMEAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "BOSCAME/BOSCAMETypes.cpp.inc"

#include "BOSCAME/BOSCAMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "BOSCAME/BOSCAME.cpp.inc"

namespace {

using bufferization::AnalysisState;
using bufferization::BufferizationOptions;
using bufferization::BufferizationState;
using bufferization::DstBufferizableOpInterfaceExternalModel;

static LogicalResult verifyTensorBufferForm(Operation *op,
                                            ValueRange destinations) {
  const bool tensorForm = llvm::all_of(destinations, [](Value value) {
    return isa<TensorType>(value.getType());
  });
  const bool bufferForm = llvm::all_of(destinations, [](Value value) {
    return isa<BaseMemRefType>(value.getType());
  });
  if (!tensorForm && !bufferForm)
    return op->emitOpError("does not support mixed tensor/buffer destinations");
  if (tensorForm) {
    if (op->getNumResults() != destinations.size())
      return op->emitOpError("tensor form requires one result per destination");
    for (auto [result, destination] : llvm::zip(op->getResults(), destinations))
      if (result.getType() != destination.getType())
        return op->emitOpError("result types must match destination types");
  } else if (op->getNumResults() != 0) {
    return op->emitOpError("buffer form must not produce results");
  }
  return success();
}

static LogicalResult requireStaticShape(Operation *op, Value value,
                                        StringRef name, unsigned rank,
                                        Type elementType) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasRank() || shaped.getRank() != rank)
    return op->emitOpError()
           << name << " must be a rank-" << rank << " shaped value";
  if (!shaped.hasStaticShape())
    return op->emitOpError() << name << " must have a static shape";
  if (shaped.getElementType() != elementType)
    return op->emitOpError()
           << name << " must have element type " << elementType;
  return success();
}

static FailureOr<Value> getBufferForOperand(RewriterBase &rewriter, Value value,
                                            const BufferizationOptions &options,
                                            BufferizationState &state) {
  if (isa<BaseMemRefType>(value.getType()))
    return value;
  return bufferization::getBuffer(rewriter, value, options, state);
}

struct QuantizePerGroupOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          QuantizePerGroupOpInterface, QuantizePerGroupOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto quantize = cast<QuantizePerGroupOp>(op);
    return &operand == &quantize.getInputMutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto quantize = cast<QuantizePerGroupOp>(op);
    SmallVector<Value> buffers;
    for (Value value : quantize->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(quantize.getLoc(), quantize->getName());
    newState.addOperands(buffers);
    newState.addAttributes(quantize->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(
        rewriter, op, ValueRange{buffers[1], buffers[2]});
    return success();
  }
};

struct SiluMulQuantizePerGroupOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          SiluMulQuantizePerGroupOpInterface, SiluMulQuantizePerGroupOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto quantize = cast<SiluMulQuantizePerGroupOp>(op);
    return &operand == &quantize.getSiluMutable() ||
           &operand == &quantize.getUpMutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto quantize = cast<SiluMulQuantizePerGroupOp>(op);
    SmallVector<Value> buffers;
    for (Value value : quantize->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(quantize.getLoc(), quantize->getName());
    newState.addOperands(buffers);
    newState.addAttributes(quantize->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(
        rewriter, op, ValueRange{buffers[2], buffers[3]});
    return success();
  }
};

struct W8A8LinearOpInterface
    : public DstBufferizableOpInterfaceExternalModel<W8A8LinearOpInterface,
                                                     W8A8LinearOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto linear = cast<W8A8LinearOp>(op);
    return &operand != &linear.getOutputMutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto linear = cast<W8A8LinearOp>(op);
    SmallVector<Value> buffers;
    for (Value value : linear->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(linear.getLoc(), linear->getName());
    newState.addOperands(buffers);
    newState.addAttributes(linear->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(rewriter, op, buffers[4]);
    return success();
  }
};

struct W8A8LinearPairOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          W8A8LinearPairOpInterface, W8A8LinearPairOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto pair = cast<W8A8LinearPairOp>(op);
    return &operand != &pair.getOutput0Mutable() &&
           &operand != &pair.getOutput1Mutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto pair = cast<W8A8LinearPairOp>(op);
    SmallVector<Value> buffers;
    for (Value value : pair->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(pair.getLoc(), pair->getName());
    newState.addOperands(buffers);
    newState.addAttributes(pair->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(
        rewriter, op, ValueRange{buffers[6], buffers[7]});
    return success();
  }
};

} // namespace

LogicalResult QuantizePerGroupOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getInput(), "input", 2, f32)) ||
      failed(requireStaticShape(*this, getQuantized(), "quantized", 2, i8)) ||
      failed(requireStaticShape(*this, getScales(), "scales", 2, f32)) ||
      failed(verifyTensorBufferForm(*this,
                                    ValueRange{getQuantized(), getScales()})))
    return failure();

  auto inputType = cast<ShapedType>(getInput().getType());
  auto quantizedType = cast<ShapedType>(getQuantized().getType());
  auto scalesType = cast<ShapedType>(getScales().getType());
  int64_t groupSize = getGroupSize();
  int64_t rows = inputType.getDimSize(0);
  int64_t width = inputType.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (quantizedType.getShape() != inputType.getShape())
    return emitOpError("quantized shape must match input shape");
  if (scalesType.getShape() != ArrayRef<int64_t>{rows, width / groupSize})
    return emitOpError("scales shape must be [T, K / group_size]");
  return success();
}

void QuantizePerGroupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Tensor form is a deterministic value computation.  Describing it as
  // effect-free lets CSE share activation quantization across linears that
  // consume the same activation (Q/K/V and Gate/Up in Qwen3).  Buffer form
  // must retain its explicit reads and writes so two calls with different
  // destinations are never folded after bufferization.
  if (isa<BaseMemRefType>(getInput().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable());
  if (isa<BaseMemRefType>(getQuantized().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getQuantizedMutable());
  if (isa<BaseMemRefType>(getScales().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getScalesMutable());
}

LogicalResult SiluMulQuantizePerGroupOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getSilu(), "silu", 2, f32)) ||
      failed(requireStaticShape(*this, getUp(), "up", 2, f32)) ||
      failed(requireStaticShape(*this, getQuantized(), "quantized", 2, i8)) ||
      failed(requireStaticShape(*this, getScales(), "scales", 2, f32)) ||
      failed(verifyTensorBufferForm(*this,
                                    ValueRange{getQuantized(), getScales()})))
    return failure();

  auto siluType = cast<ShapedType>(getSilu().getType());
  auto upType = cast<ShapedType>(getUp().getType());
  auto quantizedType = cast<ShapedType>(getQuantized().getType());
  auto scalesType = cast<ShapedType>(getScales().getType());
  // All four shaped values participate in one DPS form; mixed tensor/memref
  // operands cannot be bufferized or reasoned about safely.
  const bool tensorForm = isa<TensorType>(getQuantized().getType());
  const bool inputsMatchForm =
      tensorForm
          ? isa<RankedTensorType>(getSilu().getType()) &&
                isa<RankedTensorType>(getUp().getType())
          : isa<MemRefType>(getSilu().getType()) &&
                isa<MemRefType>(getUp().getType());
  if (!inputsMatchForm)
    return emitOpError(
        "silu/up and destinations must use the same tensor/buffer form");
  int64_t groupSize = getGroupSize();
  int64_t rows = siluType.getDimSize(0);
  int64_t width = siluType.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (upType.getShape() != siluType.getShape())
    return emitOpError("up shape must match silu shape");
  if (quantizedType.getShape() != siluType.getShape())
    return emitOpError("quantized shape must match silu shape");
  if (scalesType.getShape() != ArrayRef<int64_t>{rows, width / groupSize})
    return emitOpError("scales shape must be [T, K / group_size]");
  return success();
}

void SiluMulQuantizePerGroupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (isa<BaseMemRefType>(getSilu().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getSiluMutable());
  if (isa<BaseMemRefType>(getUp().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getUpMutable());
  if (isa<BaseMemRefType>(getQuantized().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getQuantizedMutable());
  if (isa<BaseMemRefType>(getScales().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getScalesMutable());
}

LogicalResult W8A8LinearOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getXq(), "xq", 2, i8)) ||
      failed(requireStaticShape(*this, getXs(), "xs", 2, f32)) ||
      failed(requireStaticShape(*this, getWq(), "wq", 4, i8)) ||
      failed(requireStaticShape(*this, getWs(), "ws", 2, f32)) ||
      failed(requireStaticShape(*this, getOutput(), "output", 2, f32)) ||
      failed(verifyTensorBufferForm(*this, ValueRange{getOutput()})))
    return failure();

  if (getWeightLayout() != "ame_outblk64")
    return emitOpError("weight_layout must be 'ame_outblk64'");

  auto xqType = cast<ShapedType>(getXq().getType());
  auto xsType = cast<ShapedType>(getXs().getType());
  auto wqType = cast<ShapedType>(getWq().getType());
  auto wsType = cast<ShapedType>(getWs().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  int64_t groupSize = getGroupSize();
  int64_t tokens = xqType.getDimSize(0);
  int64_t width = xqType.getDimSize(1);
  int64_t outputWidth = outputType.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (width % 64 != 0)
    return emitOpError("K must be divisible by the AME K tile (64)");
  if (groupSize % 64 != 0)
    return emitOpError("group_size must be divisible by the AME K tile (64)");
  if (outputWidth % 64 != 0)
    return emitOpError("D must be divisible by OUTBLK (64)");
  int64_t groups = width / groupSize;
  if (xsType.getShape() != ArrayRef<int64_t>{tokens, groups})
    return emitOpError("xs shape must be [T, K / group_size]");
  if (wqType.getShape() !=
      ArrayRef<int64_t>{outputWidth / 64, groups, 64, groupSize})
    return emitOpError(
        "wq shape must be [D / 64, K / group_size, 64, group_size]");
  if (wsType.getShape() != ArrayRef<int64_t>{groups, outputWidth})
    return emitOpError("ws shape must be [K / group_size, D]");
  if (outputType.getDimSize(0) != tokens)
    return emitOpError("output token dimension must match xq");
  return success();
}

LogicalResult W8A8LinearPairOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getXq(), "xq", 2, i8)) ||
      failed(requireStaticShape(*this, getXs(), "xs", 2, f32)) ||
      failed(requireStaticShape(*this, getWq0(), "wq0", 4, i8)) ||
      failed(requireStaticShape(*this, getWs0(), "ws0", 2, f32)) ||
      failed(requireStaticShape(*this, getWq1(), "wq1", 4, i8)) ||
      failed(requireStaticShape(*this, getWs1(), "ws1", 2, f32)) ||
      failed(requireStaticShape(*this, getOutput0(), "output0", 2, f32)) ||
      failed(requireStaticShape(*this, getOutput1(), "output1", 2, f32)) ||
      failed(verifyTensorBufferForm(
          *this, ValueRange{getOutput0(), getOutput1()})))
    return failure();

  if (getWeightLayout() != "ame_outblk64")
    return emitOpError("weight_layout must be 'ame_outblk64'");

  auto xqType = cast<ShapedType>(getXq().getType());
  auto xsType = cast<ShapedType>(getXs().getType());
  auto wq0Type = cast<ShapedType>(getWq0().getType());
  auto ws0Type = cast<ShapedType>(getWs0().getType());
  auto wq1Type = cast<ShapedType>(getWq1().getType());
  auto ws1Type = cast<ShapedType>(getWs1().getType());
  auto output0Type = cast<ShapedType>(getOutput0().getType());
  auto output1Type = cast<ShapedType>(getOutput1().getType());
  int64_t groupSize = getGroupSize();
  int64_t tokens = xqType.getDimSize(0);
  int64_t width = xqType.getDimSize(1);
  int64_t outputWidth = output0Type.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (width % 64 != 0)
    return emitOpError("K must be divisible by the AME K tile (64)");
  if (groupSize % 64 != 0)
    return emitOpError("group_size must be divisible by the AME K tile (64)");
  if (outputWidth % 64 != 0)
    return emitOpError("D must be divisible by OUTBLK (64)");
  if (output1Type.getShape() != output0Type.getShape())
    return emitOpError("paired output shapes must match");
  if (isa<BaseMemRefType>(getOutput0().getType()) &&
      getOperation()->getOperand(6) == getOperation()->getOperand(7))
    return emitOpError("buffer-form paired outputs must not alias");
  int64_t groups = width / groupSize;
  if (xsType.getShape() != ArrayRef<int64_t>{tokens, groups})
    return emitOpError("xs shape must be [T, K / group_size]");
  SmallVector<int64_t> expectedWqShape =
      {outputWidth / 64, groups, 64, groupSize};
  SmallVector<int64_t> expectedWsShape = {groups, outputWidth};
  if (wq0Type.getShape() != ArrayRef<int64_t>(expectedWqShape) ||
      wq1Type.getShape() != ArrayRef<int64_t>(expectedWqShape))
    return emitOpError(
        "both wq shapes must be [D / 64, K / group_size, 64, group_size]");
  if (ws0Type.getShape() != ArrayRef<int64_t>(expectedWsShape) ||
      ws1Type.getShape() != ArrayRef<int64_t>(expectedWsShape))
    return emitOpError("both ws shapes must be [K / group_size, D]");
  if (output0Type.getDimSize(0) != tokens)
    return emitOpError("output token dimension must match xq");
  return success();
}

//===----------------------------------------------------------------------===//
// SPM storage space attribute
//===----------------------------------------------------------------------===//

LogicalResult SPMAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t bank) {
  if (bank < 0 || bank >= kSpmBankCount)
    return emitError() << "SPM bank must be in [0, " << (kSpmBankCount - 1)
                       << "], got " << bank;
  return success();
}

//===----------------------------------------------------------------------===//
// DMA request token
//===----------------------------------------------------------------------===//

LogicalResult DMATokenType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   int64_t channel) {
  if (channel < 0 || channel >= kDmaChannelCount)
    return emitError() << "DMA channel must be in [0, "
                       << (kDmaChannelCount - 1) << "], got " << channel;
  return success();
}

//===----------------------------------------------------------------------===//
// SPM / DMA verifiers
//===----------------------------------------------------------------------===//

/// A byte range inside one SPM bank, expressed relative to the start of that
/// bank.  Only ranges that are provably bank-anchored may be compared with each
/// other; an opaque `memref` that merely *carries* the SPM storage space is not
/// necessarily anchored at the bank base.
struct BankAnchoredRange {
  int64_t bank;
  int64_t begin;
  int64_t end;
};

/// Derive the bank-anchored byte range that a transfer of `bytes` bytes starting
/// at `value` occupies, or `std::nullopt` when the value cannot be traced back
/// to a bank-anchored origin.  The two supported origins are the result of a
/// `bosc_ame.spm.view` (whose `offset` is bank-relative by construction) and an
/// SPM block argument, which the workspace contract pins to the start of its
/// bank.
static std::optional<BankAnchoredRange> getBankAnchoredRange(Value value,
                                                             int64_t bytes) {
  if (auto view = value.getDefiningOp<SpmViewOp>()) {
    int64_t begin = static_cast<int64_t>(view.getOffset());
    return BankAnchoredRange{view.getBank(), begin, begin + bytes};
  }
  // A block argument counts as bank-anchored only when it is exactly one
  // complete bank: the workspace contract maps each bank used by a kernel to one
  // such parameter.  A narrower SPM memref argument has no statically known
  // position inside its bank.
  if (isa<BlockArgument>(value)) {
    auto memref = dyn_cast<MemRefType>(value.getType());
    if (!memref)
      return std::nullopt;
    std::optional<int64_t> bank = getSpmBank(memref);
    std::optional<int64_t> size = getContiguousZeroOffsetByteSize(memref);
    if (bank && size && *size == kSpmBankCapacityBytes)
      return BankAnchoredRange{*bank, 0, bytes};
  }
  return std::nullopt;
}

/// Verify the static part of an `bosc_ame.spm.view`.  Everything that is a
/// property of the operation itself lives here; the cross-operation rules
/// (tokens, channel occupancy, buffer conflicts) live in
/// `bosc-ame-spm-dma-check`.
LogicalResult SpmViewOp::verify() {
  int64_t bank = getBank();
  if (bank < 0 || bank >= kSpmBankCount)
    return emitOpError() << "bank must be in [0, " << (kSpmBankCount - 1)
                         << "], got " << bank;

  auto viewType = dyn_cast<MemRefType>(getView().getType());
  if (!viewType)
    return emitOpError("result must be a ranked memref");

  auto space = dyn_cast_or_null<SPMAttr>(viewType.getMemorySpace());
  if (!space)
    return emitOpError() << "result must carry the #bosc_ame.spm<bank> storage "
                            "space, got "
                         << getView().getType();
  if (space.getBank() != bank)
    return emitOpError() << "bank " << bank
                         << " does not match the result storage space "
                         << space;

  if (!viewType.hasStaticShape())
    return emitOpError("result must have a static shape");

  // The size of the view comes from the result type, never from a second
  // redundant size operand: two independent descriptions of the same byte range
  // could disagree.
  if (getSpmElementByteWidth(viewType.getElementType()) == 0)
    return emitOpError() << "element type " << viewType.getElementType()
                         << " is not addressable in SPM; supported element "
                            "types are i8/i16/i32/i64/f16/bf16/f32/f64";

  std::optional<int64_t> sizeBytes = getContiguousZeroOffsetByteSize(viewType);
  if (!sizeBytes)
    return emitOpError("result must be contiguous with a zero descriptor "
                       "offset; a strided or otherwise shifted SPM view has no "
                       "single byte range");
  if (*sizeBytes <= 0)
    return emitOpError("result must not be empty");
  if (*sizeBytes > kSpmBankCapacityBytes)
    return emitOpError() << "view of " << *sizeBytes
                         << " bytes does not fit in a " << kSpmBankCapacityBytes
                         << "-byte SPM bank";

  uint64_t offset = getOffset();
  if (static_cast<int64_t>(offset) < 0)
    return emitOpError() << "offset must be non-negative, got "
                         << static_cast<int64_t>(offset);
  // `requiredBytes` is <= 128 KiB, so this addition cannot overflow.
  uint64_t requiredBytes = static_cast<uint64_t>(*sizeBytes);
  if (offset > static_cast<uint64_t>(kSpmBankCapacityBytes) - requiredBytes)
    return emitOpError() << "byte range [" << offset << ", "
                         << (offset + requiredBytes)
                         << ") exceeds the " << kSpmBankCapacityBytes
                         << "-byte SPM bank " << bank;

  return success();
}

/// Verify the static part of a DMA request.  The verifier answers, without
/// looking at any other operation, whether the request is a well-formed
/// description of one contiguous transfer of the patch7 DMA engines.
LogicalResult DmaStartOp::verify() {
  int64_t channel = getChannel();
  if (channel < 0 || channel >= kDmaChannelCount)
    return emitOpError() << "channel must be in [0, " << (kDmaChannelCount - 1)
                         << "], got " << channel;

  auto tokenType = dyn_cast<DMATokenType>(getToken().getType());
  if (!tokenType)
    return emitOpError() << "result must be !bosc_ame.dma_token<channel>, got "
                         << getToken().getType();
  if (tokenType.getChannel() != channel)
    return emitOpError() << "result token channel " << tokenType.getChannel()
                         << " does not match the request channel " << channel;

  auto srcType = dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(getDst().getType());
  if (!srcType)
    return emitOpError() << "src must be a ranked memref, got "
                         << getSrc().getType();
  if (!dstType)
    return emitOpError() << "dst must be a ranked memref, got "
                         << getDst().getType();

  std::optional<DmaEndpoint> src = getDmaEndpoint(srcType);
  std::optional<DmaEndpoint> dst = getDmaEndpoint(dstType);
  if (!src)
    return emitOpError() << "src must be a 1-D i8 memref with a static shape, "
                            "unit stride and a static descriptor offset, got "
                         << srcType;
  if (!dst)
    return emitOpError() << "dst must be a 1-D i8 memref with a static shape, "
                            "unit stride and a static descriptor offset, got "
                         << dstType;
  if (!src->isSpm() && !dst->isSpm())
    return emitOpError("at least one endpoint must be SPM storage; DDR to DDR "
                       "transfers are not part of the v1 contract");

  std::optional<int64_t> bytesValue = getConstantIntValue(getBytes());
  if (!bytesValue)
    return emitOpError("v1 requires a statically known transfer length: `bytes` "
                       "must fold to a constant number of bytes");
  int64_t bytes = *bytesValue;
  if (bytes <= 0)
    return emitOpError() << "bytes must be positive, got " << bytes;
  if (bytes % kDmaAlignmentBytes != 0)
    return emitOpError() << "bytes must be a multiple of the "
                         << kDmaAlignmentBytes << "-byte DMA granularity, got "
                         << bytes;

  if (bytes > src->byteLength)
    return emitOpError() << "transfer of " << bytes
                         << " bytes exceeds the src byte range of "
                         << src->byteLength << " bytes";
  if (bytes > dst->byteLength)
    return emitOpError() << "transfer of " << bytes
                         << " bytes exceeds the dst byte range of "
                         << dst->byteLength << " bytes";

  for (const DmaEndpoint *endpoint : {&*src, &*dst}) {
    if (!endpoint->isSpm())
      continue;
    if (endpoint->byteOffset + bytes > kSpmBankCapacityBytes)
      return emitOpError() << "byte range [" << endpoint->byteOffset << ", "
                           << (endpoint->byteOffset + bytes)
                           << ") leaves SPM bank " << *endpoint->bank;
    if (!isStaticallyAlignedEndpoint(*endpoint))
      return emitOpError()
             << "SPM endpoint must be 64 B aligned, but its descriptor offset "
             << endpoint->byteOffset << " is not";
  }

  // A DDR base address is not a compile-time constant.  The module-level ABI
  // assertion records that the caller handed us 64 B aligned, pairwise
  // non-overlapping buffers, which is what makes the DDR-side alignment and
  // aliasing guarantees available to the compiler.
  if ((!src->isSpm() || !dst->isSpm()) &&
      getOperation()->getParentOfType<ModuleOp>()->getAttr(kDmaAbiAttrName) !=
          StringAttr::get(getContext(), kDmaAbiAlignedNonAliasing))
    return emitOpError()
           << "a DDR endpoint cannot be proven aligned and non-overlapping; "
              "annotate the module with `"
           << kDmaAbiAttrName << " = \"" << kDmaAbiAlignedNonAliasing << "\"`";

  if (src->isSpm() && dst->isSpm() && *src->bank == *dst->bank) {
    std::optional<BankAnchoredRange> srcRange =
        getBankAnchoredRange(getSrc(), bytes);
    std::optional<BankAnchoredRange> dstRange =
        getBankAnchoredRange(getDst(), bytes);
    if (!srcRange || !dstRange)
      return emitOpError()
             << "cannot prove that the src and dst ranges in SPM bank "
             << *src->bank
             << " are disjoint: at least one endpoint is not a direct "
                "bosc_ame.spm.view or an SPM block argument";
    if (srcRange->begin < dstRange->end && dstRange->begin < srcRange->end)
      return emitOpError() << "src range [" << srcRange->begin << ", "
                           << srcRange->end << ") and dst range ["
                           << dstRange->begin << ", " << dstRange->end
                           << ") overlap in SPM bank " << *src->bank;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DMA memory effects
//===----------------------------------------------------------------------===//

namespace {
/// The DMA engines as a side-effect resource.  Both `dma.start` and `dma.wait`
/// write this resource, so no optimization may reorder, eliminate or duplicate a
/// transfer or its completion.  Volatile MMIO alone would not be enough: the
/// effect interface is what keeps the request alive and its status read where
/// the program put it.
struct DmaDeviceResource
    : public SideEffects::Resource::Base<DmaDeviceResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DmaDeviceResource)
  StringRef getName() const final { return "bosc_ame.dma"; }
};
} // namespace

void DmaStartOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // The request reads the source range and writes the destination range, but
  // the transfer is asynchronous: these effects describe the window that starts
  // here and is closed by the matching `dma.wait`.  Because no interface can
  // express "written later, by the device", the op also keeps the conservative
  // default-resource barrier below.
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       DmaDeviceResource::get());
}

void DmaWaitOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Waiting publishes the destination range written by the device without the
  // compiler knowing which range that is; only the paired `dma.start` does.
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       DmaDeviceResource::get());
}

void BOSCAMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BOSCAME/BOSCAME.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "BOSCAME/BOSCAMETypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "BOSCAME/BOSCAMEAttrs.cpp.inc"
      >();
  QuantizePerGroupOp::attachInterface<QuantizePerGroupOpInterface>(
      *getContext());
  SiluMulQuantizePerGroupOp::attachInterface<
      SiluMulQuantizePerGroupOpInterface>(*getContext());
  W8A8LinearOp::attachInterface<W8A8LinearOpInterface>(*getContext());
  W8A8LinearPairOp::attachInterface<W8A8LinearPairOpInterface>(*getContext());
}
