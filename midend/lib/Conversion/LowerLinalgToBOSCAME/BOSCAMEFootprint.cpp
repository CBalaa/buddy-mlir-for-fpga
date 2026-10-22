//===- BOSCAMEFootprint.cpp - Physical transfer footprint -----------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFootprint.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace ::buddy::boscame;

std::optional<int64_t> buddy::boscame::getBOSCAMEElementBytes(Type type) {
  unsigned bytes = spmdma::getSpmElementByteWidth(type);
  return bytes ? std::optional<int64_t>(bytes) : std::nullopt;
}

AmeMatmulTilePlan *buddy::boscame::findAmeMatmulTilePlan(DataflowPlan &plan,
                                                         Operation *matmul) {
  for (AmeMatmulTilePlan &tilePlan : plan.ameMatmulTiles)
    if (tilePlan.matmul == matmul)
      return &tilePlan;
  return nullptr;
}

const AmeMatmulTilePlan *
buddy::boscame::findAmeMatmulTilePlan(const DataflowPlan &plan,
                                      Operation *matmul) {
  for (const AmeMatmulTilePlan &tilePlan : plan.ameMatmulTiles)
    if (tilePlan.matmul == matmul)
      return &tilePlan;
  return nullptr;
}

Value buddy::boscame::canonicalizeBOSCAMEGlobalRoot(Value root) {
  auto global = root.getDefiningOp<memref::GetGlobalOp>();
  if (!global)
    return root;
  ModuleOp module = global->getParentOfType<ModuleOp>();
  if (!module)
    return root;
  auto target = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      global, global.getNameAttr());
  if (!target)
    return root;

  Value canonicalRoot;
  module.walk([&](memref::GetGlobalOp candidate) {
    auto candidateTarget =
        SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
            candidate, candidate.getNameAttr());
    if (!canonicalRoot && candidateTarget == target)
      canonicalRoot = candidate->getResult(0);
  });
  return canonicalRoot ? canonicalRoot : root;
}

static std::optional<int64_t> getSourceOriginId(Operation *anchor,
                                                TransferRole role) {
  if (anchor->hasAttr("bosc_ame.lower_to_ame")) {
    StringRef attrName;
    switch (role) {
    case TransferRole::InputA:
      attrName = "bosc_ame.a_origin_id";
      break;
    case TransferRole::InputB:
      attrName = "bosc_ame.b_origin_id";
      break;
    case TransferRole::OutputC:
      attrName = "bosc_ame.accumulator_origin_id";
      break;
    case TransferRole::Temporary:
      break;
    }
    if (!attrName.empty())
      if (auto id = anchor->getAttrOfType<IntegerAttr>(attrName))
        return id.getInt();
  }

  auto memory = anchor->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  auto id = memory ? memory.getAs<IntegerAttr>("origin_id") : IntegerAttr();
  return id ? std::optional<int64_t>(id.getInt()) : std::nullopt;
}

std::optional<TransferPlan>
buddy::boscame::analyzeBOSCAMEFootprint(Value value, TransferRole role,
                                        Operation *anchor,
                                        const TargetInfo &target) {
  auto fail = [&](Twine message) -> std::optional<TransferPlan> {
    anchor->emitError() << message;
    return std::nullopt;
  };
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type)
    return fail("BOSCAME planner requires ranked buffer-form memrefs");
  if (type.getRank() == 0)
    return fail("scalar SPM accesses are not legal BOSCAME transfers");
  if (!type.hasStaticShape())
    return fail("BOSCAME transfer shape must be static");
  if (type.getRank() > target.dma.maxStaticRank)
    return fail("BOSCAME planner supports rank-one and rank-two rectangular "
                "transfers only");
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      llvm::any_of(strides, [](int64_t stride) {
        return ShapedType::isDynamic(stride) || stride <= 0;
      }))
    return fail("BOSCAME transfer requires static positive strides");
  auto elementBytes = getBOSCAMEElementBytes(type.getElementType());
  if (!elementBytes)
    return fail("unsupported typed DMA element type");
  if (auto known = spmdma::getStaticDescriptorOffset(value);
      known && *known < 0)
    return fail("BOSCAME transfer requires a nonnegative descriptor offset");
  if (failed(spmdma::verifyEndpointPointerAlignment(
          anchor, value, "BOSCAME DDR", target.dma.endpointAlignmentBytes)))
    return std::nullopt;

  TransferPlan result;
  result.role = role;
  result.buffer = value;
  result.sourceOriginId = getSourceOriginId(anchor, role);
  result.endpointAlignmentProven = true;
  result.ddrRoot = value;
  while (Operation *def = result.ddrRoot.getDefiningOp()) {
    if (auto sub = dyn_cast<memref::SubViewOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = sub.getSource();
    } else if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = transpose.getIn();
    } else if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = cast.getSource();
    } else if (auto cast = dyn_cast<memref::CastOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = cast.getSource();
    } else if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = cast.getSource();
    } else if (auto view = dyn_cast<memref::ViewOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = view.getSource();
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = collapse.getSrc();
    } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      result.viewChain.push_back(def);
      result.ddrRoot = expand.getSrc();
    } else {
      break;
    }
  }
  result.ddrRoot = canonicalizeBOSCAMEGlobalRoot(result.ddrRoot);
  if (auto origin = spmdma::getStaticPointerOrigin(value);
      origin && canonicalizeBOSCAMEGlobalRoot(origin->root) == result.ddrRoot)
    if (auto descriptorOffset = spmdma::getStaticDescriptorOffset(value)) {
      int64_t descriptorOffsetBytes;
      int64_t effectiveOffsetBytes;
      if (!llvm::MulOverflow(*descriptorOffset, *elementBytes,
                             descriptorOffsetBytes) &&
          !llvm::AddOverflow(origin->byteShift, descriptorOffsetBytes,
                             effectiveOffsetBytes))
        result.effectiveByteOffsetBytes = effectiveOffsetBytes;
    }
  if (!isa<BlockArgument>(result.ddrRoot) &&
      !result.ddrRoot.getDefiningOp<memref::AllocOp>() &&
      !result.ddrRoot.getDefiningOp<memref::AllocaOp>() &&
      !result.ddrRoot.getDefiningOp<memref::GetGlobalOp>() &&
      !result.ddrRoot.getDefiningOp<SpmViewOp>())
    return fail("BOSCAME transfer has no supported storage root");
  result.elementType = type.getElementType();
  result.elementBytes = *elementBytes;
  result.shape.assign(type.getShape().begin(), type.getShape().end());
  result.validShape = result.shape;
  result.strides = strides;

  int64_t elements = 1;
  for (int64_t dim : result.shape)
    if (dim <= 0 || llvm::MulOverflow(elements, dim, elements))
      return fail("BOSCAME transfer shape overflows or is empty");
  if (llvm::MulOverflow(elements, *elementBytes, result.spmCapacity))
    return fail("BOSCAME transfer byte count overflows");
  int64_t span = 1;
  for (auto [dimension, stride] : llvm::zip(result.shape, strides)) {
    int64_t extent;
    if (llvm::MulOverflow(dimension - 1, stride, extent) ||
        llvm::AddOverflow(span, extent, span))
      return fail("BOSCAME transfer physical span overflows");
  }
  if (auto known = spmdma::getStaticDescriptorOffset(value))
    if (llvm::AddOverflow(span, *known, span))
      return fail("BOSCAME transfer physical span overflows");
  int64_t spanBytes;
  if (llvm::MulOverflow(span, *elementBytes, spanBytes))
    return fail("BOSCAME transfer physical byte span overflows");
  int unitAxis = -1;
  for (int i = strides.size() - 1; i >= 0; --i)
    if (strides[i] == 1) {
      unitAxis = i;
      break;
    }
  if (unitAxis < 0)
    return fail("BOSCAME transfer has no unit-stride physical axis");
  int64_t packetBytes = result.shape[unitAxis] * *elementBytes;
  result.spmStrides.assign(type.getRank(), 1);
  bool contiguous = true;
  if (type.getRank() == 2) {
    int other = 1 - unitAxis;
    if (strides[other] < result.shape[unitAxis])
      return fail("BOSCAME transfer physical rows overlap");
    result.spmStrides[other] = result.shape[unitAxis];
    contiguous = strides[other] == result.shape[unitAxis];
  }
  if (contiguous) {
    packetBytes = result.spmCapacity;
    if (packetBytes % target.dma.packetGranularityBytes)
      return fail("every BOSCAME DMA packet must be a positive 64-byte range");
    result.packets.push_back({0, 0, packetBytes});
  } else {
    if (!target.dma.packetizesPitchedRows ||
        packetBytes % target.dma.packetGranularityBytes)
      return fail("every BOSCAME DMA packet must be a positive 64-byte range");
    int other = 1 - unitAxis;
    for (int64_t i = 0; i < result.shape[other]; ++i) {
      int64_t byteOffset;
      if (llvm::MulOverflow(i, strides[other], byteOffset) ||
          llvm::MulOverflow(byteOffset, *elementBytes, byteOffset))
        return fail("BOSCAME transfer offset overflows");
      if (byteOffset % target.dma.endpointAlignmentBytes)
        return fail("BOSCAME DMA physical row address is not 64 B aligned");
      result.packets.push_back({byteOffset, i * packetBytes, packetBytes});
    }
  }
  return result;
}
