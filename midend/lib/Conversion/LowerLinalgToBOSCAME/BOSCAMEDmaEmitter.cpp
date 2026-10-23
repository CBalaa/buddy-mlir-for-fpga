//===- BOSCAMEDmaEmitter.cpp - Emit the planned typed transfers
//------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEDmaEmitter.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanner.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>

using namespace mlir;
using namespace ::buddy::boscame;

namespace {
Value index(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

/// Rebuild only the descriptor offset, retaining the SSA divisibility proof.
/// extract_strided_metadata alone would hide a dynamic subview's aligned
/// arithmetic from the verifier. A reinterpret_cast replaces this offset;
/// a subview adds its shifts using the source's physical strides.
Value descriptorOffset(OpBuilder &builder, Location loc, Value source) {
  if (auto offset = spmdma::getStaticDescriptorOffset(source))
    return index(builder, loc, *offset);
  if (auto reinterpret = source.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto part = reinterpret.getMixedOffsets().front();
    if (auto value = dyn_cast<Value>(part))
      return value;
    return index(builder, loc,
                 cast<IntegerAttr>(cast<Attribute>(part)).getInt());
  }
  if (auto sub = source.getDefiningOp<memref::SubViewOp>()) {
    Value offset = descriptorOffset(builder, loc, sub.getSource());
    SmallVector<int64_t> strides;
    int64_t ignored;
    (void)cast<MemRefType>(sub.getSource().getType())
        .getStridesAndOffset(strides, ignored);
    for (auto [part, stride] : llvm::zip(sub.getMixedOffsets(), strides)) {
      Value value;
      if (auto attr = dyn_cast<Attribute>(part))
        value = index(builder, loc, cast<IntegerAttr>(attr).getInt());
      else
        value = cast<Value>(part);
      value = arith::MulIOp::create(builder, loc, value,
                                    index(builder, loc, stride));
      offset = arith::AddIOp::create(builder, loc, offset, value);
    }
    return offset;
  }
  if (auto cast = source.getDefiningOp<memref::CastOp>())
    return descriptorOffset(builder, loc, cast.getSource());
  if (auto transpose = source.getDefiningOp<memref::TransposeOp>())
    return descriptorOffset(builder, loc, transpose.getIn());
  if (auto collapse = source.getDefiningOp<memref::CollapseShapeOp>())
    return descriptorOffset(builder, loc, collapse.getSrc());
  if (auto expand = source.getDefiningOp<memref::ExpandShapeOp>())
    return descriptorOffset(builder, loc, expand.getSrc());
  // The planner rejects any offset without a proof, so this case is only a
  // defensive fallback for future supported view operations.
  return memref::ExtractStridedMetadataOp::create(builder, loc, source)
      .getOffset();
}

Value endpoint(OpBuilder &builder, Location loc, Value source, Value offset,
               int64_t count) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto type = MemRefType::get(
      {count}, sourceType.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, {1}),
      sourceType.getMemorySpace());
  return memref::ReinterpretCastOp::create(
      builder, loc, type, source, offset,
      ArrayRef<OpFoldResult>{builder.getIndexAttr(count)},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
}

void emitPacket(OpBuilder &builder, Location loc, const TransferPlan &transfer,
                const DmaPacket &packet, Value spm, bool write) {
  Value offset = descriptorOffset(builder, loc, transfer.buffer);
  if (packet.ddrOffsetBytes)
    offset = arith::AddIOp::create(
        builder, loc, offset,
        index(builder, loc, packet.ddrOffsetBytes / transfer.elementBytes));
  Value ddr = endpoint(builder, loc, transfer.buffer, offset,
                       packet.bytes / transfer.elementBytes);
  Value spmEnd = endpoint(
      builder, loc, spm,
      index(builder, loc, packet.spmOffsetBytes / transfer.elementBytes),
      packet.bytes / transfer.elementBytes);
  Value bytes = index(builder, loc, packet.bytes);
  auto token = DmaStartOp::create(
      builder, loc, DMATokenType::get(builder.getContext(), transfer.channel),
      write ? spmEnd : ddr, write ? ddr : spmEnd, bytes,
      builder.getI32IntegerAttr(transfer.channel));
  DmaWaitOp::create(builder, loc, token.getToken());
}

struct SpmCopyPlan {
  Operation *operation;
  TransferPlan source;
  TransferPlan target;
};

static void copyBOSCAMEGraphAttrs(Operation *source, Operation *target);

/// The compact packet offsets use physical-axis order. Both endpoints must
/// agree on that order: DMA cannot transpose the elements of a row.
int unitAxis(const TransferPlan &transfer) {
  for (int axis = transfer.strides.size() - 1; axis >= 0; --axis)
    if (transfer.strides[axis] == 1)
      return axis;
  return -1;
}

LogicalResult verifySpmCopyRanges(const SpmCopyPlan &copy,
                                  const TargetInfo &targetInfo) {
  auto baseAddress = [&](const TransferPlan &transfer) -> FailureOr<int64_t> {
    auto origin = spmdma::getStaticPointerOrigin(transfer.buffer);
    auto view = origin ? origin->root.getDefiningOp<SpmViewOp>() : SpmViewOp();
    auto offset = spmdma::getStaticDescriptorOffset(transfer.buffer);
    if (!view || !offset || *offset < 0 ||
        *offset > targetInfo.spmBankCapacityBytes / transfer.elementBytes)
      return failure();
    auto capacity = spmdma::getContiguousZeroOffsetByteSize(
        cast<MemRefType>(view.getView().getType()));
    int64_t relative;
    if (!capacity ||
        llvm::AddOverflow(origin->byteShift, *offset * transfer.elementBytes,
                          relative) ||
        relative < 0 || relative > *capacity)
      return failure();
    for (const DmaPacket &packet : transfer.packets)
      if (packet.bytes > *capacity - relative ||
          packet.ddrOffsetBytes > *capacity - relative - packet.bytes)
        return failure();
    return static_cast<int64_t>(view.getOffset()) + relative;
  };
  auto sourceBase = baseAddress(copy.source);
  auto targetBase = baseAddress(copy.target);
  if (failed(sourceBase) || failed(targetBase))
    return copy.operation->emitError(
        "SPM DMA copy requires statically bounded endpoints within a bank");
  if (copy.source.ddrRoot.getDefiningOp<SpmViewOp>().getBank() !=
      copy.target.ddrRoot.getDefiningOp<SpmViewOp>().getBank())
    return success();
  // Check the entire copy, not just each individual DMA. An early destination
  // packet must not overwrite a later source packet: DMA is not memmove.
  size_t source = 0, target = 0;
  while (source < copy.source.packets.size() &&
         target < copy.target.packets.size()) {
    const DmaPacket &from = copy.source.packets[source];
    const DmaPacket &to = copy.target.packets[target];
    int64_t fromBegin = *sourceBase + from.ddrOffsetBytes;
    int64_t toBegin = *targetBase + to.ddrOffsetBytes;
    if (fromBegin < toBegin + to.bytes && toBegin < fromBegin + from.bytes)
      return copy.operation->emitError(
          "SPM DMA copy source and destination overlap");
    if (fromBegin + from.bytes <= toBegin)
      ++source;
    else
      ++target;
  }
  return success();
}

LogicalResult emitSpmCopy(const SpmCopyPlan &copy, const TargetInfo &targetInfo,
                          bool preserveGraph) {
  OpBuilder builder(copy.operation);
  Location loc = copy.operation->getLoc();
  const TransferPlan &source = copy.source, &target = copy.target;
  int64_t channel = *spmdma::getSpmBank(target.buffer.getType());
  if (channel < 0 || channel >= targetInfo.dmaChannelCount)
    return copy.operation->emitError(
        "SPM DMA copy has no channel in the target capability");
  size_t sourcePacket = 0, targetPacket = 0;
  int64_t sourceConsumed = 0, targetConsumed = 0;
  Value sourceOffset = descriptorOffset(builder, loc, source.buffer);
  Value targetOffset = descriptorOffset(builder, loc, target.buffer);
  Operation *completion = nullptr;
  while (sourcePacket < source.packets.size()) {
    const DmaPacket &from = source.packets[sourcePacket];
    const DmaPacket &to = target.packets[targetPacket];
    int64_t bytes =
        std::min(from.bytes - sourceConsumed, to.bytes - targetConsumed);
    auto slice = [&](const TransferPlan &transfer, Value base,
                     int64_t offsetBytes) {
      // The graph-preserving pipeline does not run the legacy greedy fold.
      // Keep proved SPM offsets manifestly static for descriptor verification.
      if (preserveGraph)
        if (auto known = spmdma::getStaticDescriptorOffset(transfer.buffer))
          return endpoint(
              builder, loc, transfer.buffer,
              index(builder, loc, *known + offsetBytes / transfer.elementBytes),
              bytes / transfer.elementBytes);
      Value offset = base;
      if (offsetBytes)
        offset = arith::AddIOp::create(
            builder, loc, offset,
            index(builder, loc, offsetBytes / transfer.elementBytes));
      return endpoint(builder, loc, transfer.buffer, offset,
                      bytes / transfer.elementBytes);
    };
    Value src =
        slice(source, sourceOffset, from.ddrOffsetBytes + sourceConsumed);
    Value dst = slice(target, targetOffset, to.ddrOffsetBytes + targetConsumed);
    auto token = DmaStartOp::create(
        builder, loc, DMATokenType::get(builder.getContext(), channel), src,
        dst, index(builder, loc, bytes), builder.getI32IntegerAttr(channel));
    completion = DmaWaitOp::create(builder, loc, token.getToken());
    sourceConsumed += bytes;
    targetConsumed += bytes;
    if (sourceConsumed == from.bytes) {
      ++sourcePacket;
      sourceConsumed = 0;
    }
    if (targetConsumed == to.bytes) {
      ++targetPacket;
      targetConsumed = 0;
    }
  }
  // The last wait completes every packet of the original identity copy.
  // Preserve one task there rather than duplicating its ID on packet starts;
  // predecessors and reverse logical users continue to name the same event.
  if (preserveGraph && completion)
    copyBOSCAMEGraphAttrs(copy.operation, completion);
  ModuleOp module = copy.operation->getParentOfType<ModuleOp>();
  copy.operation->erase();
  if (preserveGraph && failed(verifyCompleteComputeGraphs(module)))
    return failure();
  return success();
}

/// Return true when a vector producer's operand graph reads the same storage
/// root as the destination.  The vector transfer itself only exposes the
/// temporary source, so checking the copy operands is insufficient: a
/// vector.load or vector.transfer_read may still read the external destination
/// before the transfer is retargeted.  Be conservative for unsupported
/// memref-producing operations whose storage root cannot be recovered.
static bool vectorProducerReadsRoot(Value vector, Value destination) {
  auto destinationOrigin = spmdma::getStaticPointerOrigin(destination);
  if (!destinationOrigin)
    return true;

  llvm::SmallPtrSet<Operation *, 16> visited;
  std::function<bool(Value)> readsRoot = [&](Value value) {
    Operation *defining = value.getDefiningOp();
    if (!defining || !visited.insert(defining).second)
      return false;
    for (Value operand : defining->getOperands()) {
      if (isa<MemRefType>(operand.getType())) {
        auto origin = spmdma::getStaticPointerOrigin(operand);
        if (!origin || origin->root == destinationOrigin->root)
          return true;
      }
      if (readsRoot(operand))
        return true;
    }
    return false;
  };
  return readsRoot(vector);
}

/// Retargeting a transfer moves its write from the temporary to the external
/// destination at the transfer's original position. Operations between the
/// transfer and its provenance copy are therefore safe to cross only when
/// they are pure: a read/write could observe the old ordering, and an
/// unspeculatable operation could make moving the write observable through
/// undefined behavior or control flow.
static bool onlyPureOperationsBetween(Operation *first, Operation *last) {
  if (!first || !last || first->getBlock() != last->getBlock() ||
      !first->isBeforeInBlock(last))
    return false;
  for (Operation *between = first->getNextNode(); between && between != last;
       between = between->getNextNode())
    if (!isPure(between))
      return false;
  return true;
}

/// Preserve frontend graph facts when a provenance-bearing memref.copy is
/// normalized to an identity linalg.generic. The replacement is the same
/// memory event for planning purposes; dropping its compute metadata would
/// make the post-bufferization graph appear to have missing nodes.
static void copyBOSCAMEGraphAttrs(Operation *source, Operation *target) {
  for (NamedAttribute attr : source->getAttrs())
    if (attr.getName().getValue().starts_with("bosc_ame."))
      target->setAttr(attr.getName(), attr.getValue());
}

/// Emit the exact boundary event selected by the planner, with both original
/// endpoint descriptors. Keeping its task identity on the last DMA start
/// preserves predecessor and logical-user references without contracting or
/// renumbering the compute graph. The immediate wait completes the event at
/// the original copy position, before the next computation observes it.
static LogicalResult
emitResidentBoundaryTransfer(ModuleOp module, const TransferPlan &transfer) {
  Operation *copy = transfer.provider;
  if (!copy || transfer.firstConsumer != copy ||
      transfer.lastConsumer != copy || transfer.completionOperation != copy ||
      !transfer.source || !transfer.target || !transfer.residentBuffer ||
      transfer.packets.empty() || transfer.elementBytes <= 0 ||
      transfer.read == transfer.write ||
      transfer.buffer != (transfer.read ? transfer.source : transfer.target) ||
      transfer.residentBuffer !=
          (transfer.read ? transfer.target : transfer.source))
    return module.emitError("invalid planned resident boundary DMA transfer");

  OpBuilder builder(copy);
  Location loc = copy->getLoc();
  auto slice = [&](Value buffer, int64_t packetOffsetBytes, int64_t bytes) {
    Value offset = descriptorOffset(builder, loc, buffer);
    if (packetOffsetBytes)
      offset = builder.createOrFold<arith::AddIOp>(
          loc, offset,
          index(builder, loc, packetOffsetBytes / transfer.elementBytes));
    return endpoint(builder, loc, buffer, offset,
                    bytes / transfer.elementBytes);
  };
  DmaStartOp completion;
  for (const DmaPacket &packet : transfer.packets) {
    Value external =
        slice(transfer.buffer, packet.ddrOffsetBytes, packet.bytes);
    Value resident =
        slice(transfer.residentBuffer, packet.spmOffsetBytes, packet.bytes);
    auto start = DmaStartOp::create(
        builder, loc, DMATokenType::get(builder.getContext(), transfer.channel),
        transfer.read ? external : resident,
        transfer.read ? resident : external, index(builder, loc, packet.bytes),
        builder.getI32IntegerAttr(transfer.channel));
    DmaWaitOp::create(builder, loc, start.getToken());
    if (failed(start.verify()))
      return failure();
    completion = start;
  }
  // One logical task covers the entire row sequence. Earlier packets retain
  // no duplicate IDs; the last start is followed by its mandatory wait.
  copyBOSCAMEGraphAttrs(copy, completion);
  copy->erase();
  if (failed(verifyCompleteComputeGraphs(module)))
    return failure();
  return success();
}
} // namespace

namespace buddy::boscame {

LogicalResult emitBOSCAMEDmaCopies(ModuleOp module,
                                   const TargetInfo &targetInfo,
                                   bool preserveGraph) {
  SmallVector<SpmCopyPlan, 0> copies;
  WalkResult result = module.walk([&](Operation *op) {
    Value source, targetBuffer;
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      source = copy.getSource();
      targetBuffer = copy.getTarget();
    } else if (auto linalg = dyn_cast<linalg::LinalgOp>(op)) {
      if (!linalg.hasPureBufferSemantics() || linalg.getNumDpsInputs() != 1 ||
          linalg.getNumDpsInits() != 1 ||
          !llvm::all_of(linalg.getIndexingMapsArray(),
                        [](AffineMap map) { return map.isIdentity(); }) ||
          !llvm::all_of(linalg.getIteratorTypesArray(),
                        [](utils::IteratorType iterator) {
                          return iterator == utils::IteratorType::parallel;
                        }))
        return WalkResult::advance();
      Block *body = linalg.getBlock();
      auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
      if (!llvm::hasSingleElement(*body) || !yield ||
          yield.getValues().size() != 1 ||
          yield.getValues().front() != body->getArgument(0))
        return WalkResult::advance();
      source = linalg.getDpsInputOperand(0)->get();
      targetBuffer = linalg.getDpsInitOperand(0)->get();
    } else {
      return WalkResult::advance();
    }
    if (!spmdma::getSpmBank(source.getType()) ||
        !spmdma::getSpmBank(targetBuffer.getType()))
      return WalkResult::advance();
    if (!targetInfo.dma.supports(StorageKind::SPM, StorageKind::SPM)) {
      op->emitError("target does not support SPM-to-SPM DMA copies");
      return WalkResult::interrupt();
    }
    auto srcType = cast<MemRefType>(source.getType());
    auto dstType = cast<MemRefType>(targetBuffer.getType());
    if (srcType.getShape() != dstType.getShape() ||
        srcType.getElementType() != dstType.getElementType()) {
      op->emitError("SPM DMA copy requires matching shapes and element types");
      return WalkResult::interrupt();
    }
    auto from = analyzeBOSCAMEFootprint(source, TransferRole::Temporary, op,
                                        targetInfo);
    auto to = analyzeBOSCAMEFootprint(targetBuffer, TransferRole::Temporary, op,
                                      targetInfo);
    if (!from || !to)
      return WalkResult::interrupt();
    if (unitAxis(*from) != unitAxis(*to)) {
      op->emitError("SPM DMA copy cannot transpose physical axes");
      return WalkResult::interrupt();
    }
    SpmCopyPlan copy{op, std::move(*from), std::move(*to)};
    if (failed(verifySpmCopyRanges(copy, targetInfo)))
      return WalkResult::interrupt();
    copies.push_back(std::move(copy));
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  for (const SpmCopyPlan &copy : copies)
    if (failed(emitSpmCopy(copy, targetInfo, preserveGraph)))
      return failure();
  return success();
}

/// A vector producer followed by a provenance-marked copy is already the
/// direct-store form requested by the frontend.  Keep the vector in registers
/// and retarget its transfer to the external DDR destination.  Staging this
/// temporary through SPM would introduce a DMA round trip for a value that is
/// already in an RVV register, and would also make the destination provenance
/// depend on the compiler allocation rather than the Triton pointer.
static LogicalResult fuseDirectVectorStores(ModuleOp module, bool directDDR) {
  if (!directDDR)
    return success();

  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp copy) {
    auto memory = copy->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
    auto kind = memory ? memory.getAs<StringAttr>("kind") : StringAttr();
    auto destination =
        memory ? memory.getAs<StringAttr>("destination") : StringAttr();
    if (kind && destination && kind.getValue() == "store" &&
        destination.getValue() == "external_ddr")
      copies.push_back(copy);
  });

  for (memref::CopyOp copy : copies) {
    vector::TransferWriteOp transfer;
    // `vector.transfer_write` is side-effecting and therefore does not define
    // the destination memref SSA value. Find the unique write user of that
    // value instead of looking at its defining operation.
    for (Operation *user : copy.getSource().getUsers()) {
      auto candidate = dyn_cast<vector::TransferWriteOp>(user);
      if (!candidate || candidate.getBase() != copy.getSource())
        continue;
      if (transfer) {
        transfer = vector::TransferWriteOp();
        break;
      }
      transfer = candidate;
    }
    if (!transfer || !onlyPureOperationsBetween(transfer, copy))
      continue;
    // Retargeting moves the write from the copy's position to the transfer's
    // position. The helper above permits only pure operations between them, so
    // no intervening operation can observe the changed destination ordering.
    auto vectorType = dyn_cast<VectorType>(transfer.getVector().getType());
    if (!vectorType)
      continue;
    // Retargeting is valid only when this is the complete producer chain for
    // the temporary.  A second user or a later write could observe the old
    // buffer and make the copy's ordering observable.
    bool hasOtherUser = false;
    for (Operation *user : copy.getSource().getUsers())
      if (user != transfer && user != copy) {
        hasOtherUser = true;
        break;
      }
    if (hasOtherUser)
      continue;

    // A masked transfer or a partial source cannot be retargeted without
    // changing which destination elements survive.  Leave it for the normal
    // provenance path, whose direct-mode legality check diagnoses masks.
    auto memory = copy->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
    auto masked = memory ? memory.getAs<BoolAttr>("masked") : BoolAttr();
    auto logical = memory ? memory.getAs<DenseI64ArrayAttr>("logical_shape")
                          : DenseI64ArrayAttr();
    auto valid = memory ? memory.getAs<DenseI64ArrayAttr>("valid_shape")
                        : DenseI64ArrayAttr();
    if (!masked || masked.getValue() || !logical || !valid ||
        logical.asArrayRef() != valid.asArrayRef() || transfer.getMask() ||
        !llvm::all_of(transfer.getInBoundsValues(),
                      [](bool inBounds) { return inBounds; }) ||
        !transfer.getPermutationMap().isIdentity())
      continue;

    auto destinationType = dyn_cast<MemRefType>(copy.getTarget().getType());
    auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
    if (!destinationType || !sourceType || !destinationType.hasStaticShape() ||
        !sourceType.hasStaticShape() || destinationType.getRank() < 1 ||
        destinationType.getRank() > 2 || sourceType.getRank() < 1 ||
        sourceType.getRank() > 2 ||
        destinationType.getShape() != sourceType.getShape() ||
        destinationType.getElementType() != sourceType.getElementType() ||
        destinationType.getShape() != logical.asArrayRef() ||
        vectorType.getShape() != sourceType.getShape() ||
        vectorType.getRank() != sourceType.getRank())
      continue;

    // The temporary copy covers the whole buffer only when the transfer is
    // an identity, zero-offset write. A nonzero or dynamic index would leave
    // part of the temporary undefined (or retarget a different subregion),
    // so keep that form on the ordinary provenance path.
    bool zeroIndices = transfer.getIndices().size() ==
                       static_cast<size_t>(sourceType.getRank());
    for (Value index : transfer.getIndices()) {
      auto constant = index.getDefiningOp<arith::ConstantIndexOp>();
      zeroIndices &= constant && constant.value() == 0;
    }
    if (!zeroIndices)
      continue;

    // Keep the same address contract used by a fused linalg store.  If the
    // proof fails, leave the copy intact so the planner reports its usual
    // direct-DDR diagnostic instead of silently creating an unaligned RVV
    // transfer.  Physical row pitches are checked in bytes, independently of
    // the vector's logical minor dimension.
    auto origin = spmdma::getStaticPointerOrigin(copy.getTarget());
    auto rootArgument =
        origin ? dyn_cast<BlockArgument>(origin->root) : BlockArgument();
    if (!rootArgument || !rootArgument.getOwner()->isEntryBlock() ||
        !isa<func::FuncOp>(rootArgument.getOwner()->getParentOp()))
      continue;
    if (failed(spmdma::verifyEndpointPointerAlignment(
            copy, copy.getTarget(), "direct-DDR vector store", 64)))
      return failure();
    if (vectorProducerReadsRoot(transfer.getVector(), copy.getTarget()))
      continue;
    SmallVector<int64_t> strides;
    int64_t offset;
    auto elementBytes =
        getBOSCAMEElementBytes(destinationType.getElementType());
    if (!elementBytes ||
        failed(destinationType.getStridesAndOffset(strides, offset)) ||
        strides.size() != static_cast<size_t>(destinationType.getRank()) ||
        llvm::any_of(strides, [](int64_t stride) { return stride <= 0; }) ||
        strides.back() != 1)
      continue;
    if (destinationType.getRank() > 1) {
      int64_t rowBytes;
      if (strides.front() < destinationType.getShape().back())
        continue;
      if (llvm::MulOverflow(strides.front(), *elementBytes, rowBytes) ||
          rowBytes % spmdma::kDmaAlignmentBytes != 0)
        return copy.emitError(
            "direct-DDR vector store requires 64-byte-aligned physical row "
            "strides");
    }

    // Preserve the exact transfer map and in-bounds contract while changing
    // only the destination operand.  The transfer's vector shape is checked
    // by its verifier; the copy shape check above proves that no tail is
    // dropped by this fusion.
    transfer.getBaseMutable().assign(copy.getTarget());
    transfer->setAttr("bosc_ame.memory", memory);
    transfer->setAttr("bosc_ame.direct_ddr",
                      UnitAttr::get(module.getContext()));
    copy.erase();
  }
  return success();
}

LogicalResult normalizeBOSCAMEMemoryCopies(ModuleOp module, bool directDDR) {
  if (failed(fuseDirectVectorStores(module, directDDR)))
    return failure();
  // A uniform fill is independent of logical axis order. Traverse the same
  // column-major allocation through its physical [N,K] descriptor so the SPM
  // vectorizer can use contiguous vectors, without transposing any data.
  SmallVector<linalg::FillOp> fills;
  module.walk([&](linalg::FillOp fill) {
    if (fill.hasPureBufferSemantics())
      fills.push_back(fill);
  });
  for (linalg::FillOp fill : fills) {
    Value output = fill.getOutputs().front();
    auto type = dyn_cast<MemRefType>(output.getType());
    SmallVector<int64_t> strides;
    int64_t ignored;
    if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
        failed(type.getStridesAndOffset(strides, ignored)) || strides[0] != 1 ||
        strides[1] < type.getDimSize(0))
      continue;
    OpBuilder builder(fill);
    SmallVector<int64_t> shape{type.getDimSize(1), type.getDimSize(0)};
    SmallVector<int64_t> physicalStrides{strides[1], strides[0]};
    auto physicalType = MemRefType::get(
        shape, type.getElementType(),
        StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
                               physicalStrides),
        type.getMemorySpace());
    Value physical = memref::ReinterpretCastOp::create(
        builder, fill.getLoc(), physicalType, output,
        descriptorOffset(builder, fill.getLoc(), output),
        llvm::map_to_vector(shape,
                            [&](int64_t size) -> OpFoldResult {
                              return builder.getIndexAttr(size);
                            }),
        llvm::map_to_vector(physicalStrides,
                            [&](int64_t stride) -> OpFoldResult {
                              return builder.getIndexAttr(stride);
                            }));
    fill.getOutputsMutable().assign(physical);
  }
  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp copy) {
    if (copy->hasAttr("bosc_ame.memory"))
      copies.push_back(copy);
  });
  for (memref::CopyOp copy : copies) {
    auto provenance = copy->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
    auto kind =
        provenance ? provenance.getAs<StringAttr>("kind") : StringAttr();
    if (!kind || (kind.getValue() != "load" && kind.getValue() != "store"))
      return copy.emitError("invalid BOSCAME load/store provenance");
    auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!sourceType || !targetType || sourceType.getRank() < 1 ||
        sourceType.getRank() > 2 ||
        sourceType.getShape() != targetType.getShape() ||
        sourceType.getElementType() != targetType.getElementType())
      return copy.emitError("BOSCAME provenance copy requires matching static "
                            "rank-1 or rank-2 buffers");
    if (!sourceType.hasStaticShape()) {
      // The planner handles the one dynamic form with enough semantic
      // information to emit guarded fixed-size packets. Keep its copy and
      // dynamic subviews intact until that proof is checked.
      if (sourceType.getRank() == 1)
        continue;
      return copy.emitError(
          "BOSCAME dynamic provenance copies support rank one only");
    }
    SmallVector<int64_t> sourceStrides, targetStrides;
    int64_t ignored;
    if (failed(sourceType.getStridesAndOffset(sourceStrides, ignored)) ||
        failed(targetType.getStridesAndOffset(targetStrides, ignored)))
      return copy.emitError(
          "BOSCAME provenance copy requires physical strides");
    bool sourceColumns = sourceType.getRank() == 2 && sourceStrides[0] == 1 &&
                         sourceStrides[1] >= sourceType.getDimSize(0);
    bool targetColumns = targetType.getRank() == 2 && targetStrides[0] == 1 &&
                         targetStrides[1] >= targetType.getDimSize(0);
    if (sourceColumns != targetColumns)
      return copy.emitError("BOSCAME provenance copy cannot transpose storage; "
                            "preserve the source physical layout");
    OpBuilder builder(copy);
    Location loc = copy.getLoc();
    auto physicalView = [&](Value value, MemRefType type,
                            ArrayRef<int64_t> strides) -> Value {
      if (!sourceColumns)
        return value;
      // Both endpoints describe the same logical [K,N] coordinates with K
      // contiguous. Traverse their physical [N,K] rows without packing or
      // changing either pointer. Reinterpret replaces the descriptor offset,
      // so retain its exact original SSA arithmetic once.
      SmallVector<int64_t> shape{type.getDimSize(1), type.getDimSize(0)};
      SmallVector<int64_t> physicalStrides{strides[1], strides[0]};
      auto physicalType = MemRefType::get(
          shape, type.getElementType(),
          StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
                                 physicalStrides),
          type.getMemorySpace());
      return memref::ReinterpretCastOp::create(
          builder, loc, physicalType, value,
          descriptorOffset(builder, loc, value),
          llvm::map_to_vector(shape,
                              [&](int64_t dimension) -> OpFoldResult {
                                return builder.getIndexAttr(dimension);
                              }),
          llvm::map_to_vector(physicalStrides,
                              [&](int64_t stride) -> OpFoldResult {
                                return builder.getIndexAttr(stride);
                              }));
    };
    Value source = physicalView(copy.getSource(), sourceType, sourceStrides);
    Value target = physicalView(copy.getTarget(), targetType, targetStrides);
    unsigned rank = sourceType.getRank();
    AffineMap identity = builder.getMultiDimIdentityMap(rank);
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        builder, loc, TypeRange{}, ValueRange{source}, ValueRange{target},
        ArrayRef<AffineMap>{identity, identity}, iterators,
        [](OpBuilder &nested, Location nestedLoc, ValueRange values) {
          linalg::YieldOp::create(nested, nestedLoc, values.front());
        });
    copyBOSCAMEGraphAttrs(copy, generic);
    copy.erase();
  }
  return success();
}

LogicalResult emitBOSCAMEDmaTransfers(ModuleOp module,
                                      const WorkspacePlan &workspace) {
  for (const TransferPlan &transfer : workspace.transfers) {
    Operation *consumer = transfer.firstConsumer;
    if (!consumer)
      continue;
    if (transfer.spmBank < 0 ||
        transfer.spmBank >= workspace.target.spmBankCount ||
        transfer.channel < 0 ||
        transfer.channel >= workspace.target.dmaChannelCount)
      return consumer->emitError(
          "planned DMA bank or channel exceeds the target capability");
    bool ddrToSpm =
        transfer.guardedPrefix ? transfer.prefixLoad : transfer.read;
    bool spmToDdr =
        transfer.guardedPrefix ? !transfer.prefixLoad : transfer.write;
    if ((ddrToSpm &&
         !workspace.target.dma.supports(StorageKind::DDR, StorageKind::SPM)) ||
        (spmToDdr &&
         !workspace.target.dma.supports(StorageKind::SPM, StorageKind::DDR)))
      return consumer->emitError(
          "target does not support the planned DDR/SPM transfer direction");
    if (transfer.directExternalBoundary) {
      if (failed(emitResidentBoundaryTransfer(module, transfer)))
        return failure();
      continue;
    }
    OpBuilder builder(consumer);
    Location loc = consumer->getLoc();
    auto flatType =
        MemRefType::get({transfer.spmCapacity / transfer.elementBytes},
                        transfer.elementType, MemRefLayoutAttrInterface(),
                        SPMAttr::get(builder.getContext(), transfer.spmBank));
    Value flat = SpmViewOp::create(
        builder, loc, flatType, builder.getI32IntegerAttr(transfer.spmBank),
        builder.getI64IntegerAttr(transfer.spmOffset));
    if (transfer.guardedPrefix) {
      if (!isa<memref::CopyOp>(consumer) || !transfer.validElements ||
          !transfer.prefixPeerBuffer || transfer.packets.empty())
        return consumer->emitError(
            "invalid planned BOSCAME guarded-prefix transfer");
      const int64_t packetElements =
          workspace.target.dma.packetGranularityBytes / transfer.elementBytes;
      auto vectorType = VectorType::get({packetElements}, transfer.elementType);
      for (const DmaPacket &packet : transfer.packets) {
        int64_t endElements =
            packet.spmOffsetBytes / transfer.elementBytes + packetElements;
        Value condition = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sge, transfer.validElements,
            index(builder, loc, endElements));
        auto ifOp = scf::IfOp::create(builder, loc, condition, false);
        builder.setInsertionPointToStart(ifOp.thenBlock());
        if (transfer.prefixLoad) {
          emitPacket(builder, loc, transfer, packet, flat, /*write=*/false);
          Value loaded = vector::LoadOp::create(
              builder, loc, vectorType, flat,
              ValueRange{index(builder, loc,
                               packet.spmOffsetBytes / transfer.elementBytes)});
          vector::StoreOp::create(
              builder, loc, loaded, transfer.prefixPeerBuffer,
              ValueRange{index(builder, loc,
                               packet.spmOffsetBytes / transfer.elementBytes)});
        } else {
          TransferPlan source = transfer;
          source.buffer = transfer.prefixPeerBuffer;
          emitPacket(builder, loc, source, packet, flat, /*write=*/false);
          emitPacket(builder, loc, transfer, packet, flat, /*write=*/true);
        }
        builder.setInsertionPointAfter(ifOp);
      }
      consumer->erase();
      continue;
    }
    if (transfer.read)
      for (const DmaPacket &packet : transfer.packets)
        emitPacket(builder, loc, transfer, packet, flat, /*write=*/false);
    auto type = MemRefType::get(
        transfer.shape, transfer.elementType,
        StridedLayoutAttr::get(builder.getContext(), 0, transfer.spmStrides),
        flatType.getMemorySpace());
    Value staged = memref::ReinterpretCastOp::create(
        builder, loc, type, flat, 0, transfer.shape, transfer.spmStrides);
    SmallVector<Operation *, 4> consumers(transfer.consumerOperations.begin(),
                                          transfer.consumerOperations.end());
    if (consumers.empty())
      consumers.push_back(consumer);
    for (Operation *use : consumers)
      for (OpOperand &operand : use->getOpOperands())
        if (operand.get() == transfer.buffer)
          operand.set(staged);
    if (transfer.write) {
      builder.setInsertionPointAfter(transfer.lastConsumer);
      for (const DmaPacket &packet : transfer.packets)
        emitPacket(builder, loc, transfer, packet, flat, /*write=*/true);
    }
  }
  return success();
}

} // namespace buddy::boscame
