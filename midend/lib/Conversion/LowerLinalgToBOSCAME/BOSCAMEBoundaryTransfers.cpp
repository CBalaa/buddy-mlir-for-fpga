//===- BOSCAMEBoundaryTransfers.cpp - Resident external DMA plans
//----------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEBoundaryTransfers.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace buddy::boscame;

namespace {

bool matchIdentityCopy(Operation *operation, Value &source, Value &target) {
  if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
    source = copy.getSource();
    target = copy.getTarget();
    return true;
  }
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  if (!generic || !generic.hasPureBufferSemantics() ||
      generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }) ||
      !llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType iterator) {
                      return iterator == utils::IteratorType::parallel;
                    }))
    return false;
  Block *body = generic.getBody();
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!llvm::hasSingleElement(*body) || !yield ||
      yield.getValues().size() != 1 ||
      yield.getValues().front() != body->getArgument(0))
    return false;
  source = generic.getInputs().front();
  target = generic.getOutputs().front();
  return true;
}

int unitAxis(const TransferPlan &transfer) {
  for (int axis = transfer.strides.size() - 1; axis >= 0; --axis)
    if (transfer.strides[axis] == 1)
      return axis;
  return -1;
}

bool matchesValidShape(Value value, ArrayRef<int64_t> valid) {
  auto type = cast<MemRefType>(value.getType());
  if (type.getShape() == valid)
    return true;
  // Memory-copy normalization traverses column-major [K,N] through physical
  // [N,K] rows. Its provenance still describes the logical valid rectangle.
  auto physical = value.getDefiningOp<memref::ReinterpretCastOp>();
  if (!physical || valid.size() != 2 || type.getRank() != 2)
    return false;
  auto logicalType = cast<MemRefType>(physical.getSource().getType());
  SmallVector<int64_t> logicalStrides, physicalStrides;
  int64_t ignored;
  return logicalType.getShape() == valid && type.getDimSize(0) == valid[1] &&
         type.getDimSize(1) == valid[0] &&
         succeeded(logicalType.getStridesAndOffset(logicalStrides, ignored)) &&
         succeeded(type.getStridesAndOffset(physicalStrides, ignored)) &&
         logicalStrides[0] == 1 && physicalStrides[1] == 1 &&
         logicalStrides[1] == physicalStrides[0];
}

} // namespace

std::optional<TransferPlan>
buddy::boscame::planResidentBoundaryTransfer(Operation *operation,
                                             const TargetInfo &targetInfo) {
  Value source, target;
  if (!matchIdentityCopy(operation, source, target))
    return std::nullopt;
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto targetType = dyn_cast<MemRefType>(target.getType());
  if (!sourceType || !targetType || !sourceType.hasStaticShape() ||
      !targetType.hasStaticShape() ||
      sourceType.getShape() != targetType.getShape() ||
      sourceType.getElementType() != targetType.getElementType())
    return std::nullopt;
  bool sourceSpm = spmdma::getSpmBank(sourceType).has_value();
  bool targetSpm = spmdma::getSpmBank(targetType).has_value();
  if (sourceSpm == targetSpm ||
      !targetInfo.dma.supports(sourceSpm ? StorageKind::SPM : StorageKind::DDR,
                               targetSpm ? StorageKind::SPM : StorageKind::DDR))
    return std::nullopt;
  Value ddr = sourceSpm ? target : source;
  Value resident = sourceSpm ? source : target;
  auto ddrOrigin = spmdma::getStaticPointerOrigin(ddr);
  auto ddrArgument =
      ddrOrigin ? dyn_cast<BlockArgument>(ddrOrigin->root) : BlockArgument();
  auto function =
      ddrArgument
          ? dyn_cast<func::FuncOp>(ddrArgument.getOwner()->getParentOp())
          : func::FuncOp();
  if (!function || ddrArgument.getOwner() != &function.front())
    return std::nullopt;

  // A static sliced descriptor is the valid range itself. In particular, a
  // masked 768-of-1024 event is legal when its actual copy covers exactly the
  // static contiguous 768-element prefix. No unmasked-event assumption is
  // needed and the preceding full resident padding fill remains in place.
  if (auto memory =
          operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory")) {
    auto kind = memory.getAs<StringAttr>("kind");
    auto valid = memory.getAs<DenseI64ArrayAttr>("valid_shape");
    if (!kind || kind.getValue() != (sourceSpm ? "store" : "load") || !valid ||
        !matchesValidShape(source, valid.asArrayRef()) ||
        !matchesValidShape(target, valid.asArrayRef()))
      return std::nullopt;
  }

  // Footprint analysis emits useful diagnostics when a transfer has already
  // been selected. Here failure merely rejects a hypothetical group placement.
  ScopedDiagnosticHandler silence(operation->getContext(),
                                  [](Diagnostic &) { return success(); });
  auto external = analyzeBOSCAMEFootprint(ddr, TransferRole::Temporary,
                                          operation, targetInfo);
  auto internal = analyzeBOSCAMEFootprint(resident, TransferRole::Temporary,
                                          operation, targetInfo);
  if (!external || !internal || unitAxis(*external) != unitAxis(*internal))
    return std::nullopt;

  // Match logical rows, not the two independently packed workspaces. Either
  // endpoint can have a pitch, including a slice of a larger resident tile.
  if (external->packets.size() != 1 || internal->packets.size() != 1) {
    int fast = unitAxis(*external), row = 1 - fast;
    if (external->shape.size() != 2)
      return std::nullopt;
    int64_t rowBytes;
    if (llvm::MulOverflow(external->shape[fast], external->elementBytes,
                          rowBytes) ||
        rowBytes % targetInfo.dma.packetGranularityBytes)
      return std::nullopt;
    external->packets.clear();
    for (int64_t i = 0; i < external->shape[row]; ++i) {
      int64_t ddrOffset, spmOffset;
      if (llvm::MulOverflow(i, external->strides[row], ddrOffset) ||
          llvm::MulOverflow(ddrOffset, external->elementBytes, ddrOffset) ||
          llvm::MulOverflow(i, internal->strides[row], spmOffset) ||
          llvm::MulOverflow(spmOffset, internal->elementBytes, spmOffset) ||
          ddrOffset % targetInfo.dma.endpointAlignmentBytes ||
          spmOffset % targetInfo.dma.endpointAlignmentBytes)
        return std::nullopt;
      external->packets.push_back({ddrOffset, spmOffset, rowBytes});
    }
  }

  auto origin = spmdma::getStaticPointerOrigin(resident);
  if (!origin)
    return std::nullopt;
  auto window = origin->root.getDefiningOp<SpmViewOp>();
  if (!window)
    return std::nullopt;
  auto offset = spmdma::getStaticDescriptorOffset(resident);
  auto capacity = spmdma::getContiguousZeroOffsetByteSize(
      cast<MemRefType>(window.getView().getType()));
  if (!offset || !capacity || *offset < 0 || *capacity <= 0)
    return std::nullopt;
  const int64_t windowCapacity = *capacity;
  int64_t descriptorBytes, begin;
  if (llvm::MulOverflow(*offset, external->elementBytes, descriptorBytes) ||
      llvm::AddOverflow(origin->byteShift, descriptorBytes, begin) ||
      begin < 0 || begin > windowCapacity ||
      window.getBank() >= targetInfo.spmBankCount ||
      window.getBank() >= targetInfo.dmaChannelCount ||
      windowCapacity > targetInfo.spmBankCapacityBytes ||
      window.getOffset() >
          static_cast<uint64_t>(targetInfo.spmBankCapacityBytes -
                                windowCapacity))
    return std::nullopt;
  for (const DmaPacket &packet : external->packets)
    if (packet.spmOffsetBytes < 0 ||
        packet.spmOffsetBytes > windowCapacity - begin || packet.bytes <= 0 ||
        packet.bytes > windowCapacity - begin - packet.spmOffsetBytes)
      return std::nullopt;

  external->directExternalBoundary = true;
  external->residentBuffer = resident;
  external->source = source;
  external->target = target;
  external->provider = operation;
  external->completionOperation = operation;
  external->spmBank = window.getBank();
  external->spmOffset = window.getOffset();
  external->spmCapacity = windowCapacity;
  external->channel = window.getBank();
  external->read = !sourceSpm;
  external->write = sourceSpm;
  external->firstConsumer = operation;
  external->lastConsumer = operation;
  external->consumerOperations.push_back(operation);
  external->spmStrides = internal->strides;
  return external;
}
