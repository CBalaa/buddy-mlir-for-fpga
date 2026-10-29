//===- BOSCAMETargetInfo.h - BOSCAME target capability profile -*- C++ -*-===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMETARGETINFO_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMETARGETINFO_H

#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>

namespace buddy::boscame {

enum class StorageKind { DDR, SPM };

struct TransferCapability {
  int64_t endpointAlignmentBytes = 0;
  int64_t packetGranularityBytes = 0;
  int64_t maxStaticRank = 0;
  bool supportsDdrToSpm = false;
  bool supportsSpmToDdr = false;
  bool supportsSpmToSpm = false;
  bool packetizesPitchedRows = false;

  constexpr bool supports(StorageKind source, StorageKind destination) const {
    if (source == StorageKind::DDR && destination == StorageKind::SPM)
      return supportsDdrToSpm;
    if (source == StorageKind::SPM && destination == StorageKind::DDR)
      return supportsSpmToDdr;
    if (source == StorageKind::SPM && destination == StorageKind::SPM)
      return supportsSpmToSpm;
    return false;
  }
};

enum class AmeTileLayout {
  /// Logical A[M, K] with K contiguous.
  MxKWithKContiguous,
  /// Logical B[K, N] backed by physical [N, K] storage with K contiguous.
  KxNWithPhysicalNxK,
  /// Accumulator[M, N] with N contiguous.
  MxNWithNContiguous,
};

inline llvm::StringRef stringifyAmeTileLayout(AmeTileLayout layout) {
  switch (layout) {
  case AmeTileLayout::MxKWithKContiguous:
    return "mk-k-contiguous";
  case AmeTileLayout::KxNWithPhysicalNxK:
    return "kn-physical-nk-k-contiguous";
  case AmeTileLayout::MxNWithNContiguous:
    return "mn-n-contiguous";
  }
  llvm_unreachable("unknown AME tile layout");
}

inline bool hasAmeTileLayout(mlir::MemRefType type, AmeTileLayout layout) {
  if (!type || type.getRank() != 2 || !type.hasStaticShape())
    return false;
  llvm::SmallVector<int64_t, 2> strides;
  int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != 2 || strides[0] <= 0 || strides[1] <= 0)
    return false;
  switch (layout) {
  case AmeTileLayout::MxKWithKContiguous:
  case AmeTileLayout::MxNWithNContiguous:
    return strides[1] == 1 && strides[0] >= type.getDimSize(1);
  case AmeTileLayout::KxNWithPhysicalNxK:
    return strides[0] == 1 && strides[1] >= type.getDimSize(0);
  }
  llvm_unreachable("unknown AME tile layout");
}

struct AmeRoleCapability {
  unsigned elementBits = 0;
  AmeTileLayout layout = AmeTileLayout::MxKWithKContiguous;
  bool supportsSpm = false;
  bool supportsDirectDdr = false;
};

/// The marked-matmul contract implemented by the AME dialect and lowering.
/// Integer signedness is selected by the matmul's signed cast; memref i8 types
/// themselves are signless in MLIR.
struct AmeCapability {
  int64_t maxTileM = 0;
  int64_t maxTileN = 0;
  int64_t maxTileK = 0;
  bool signedIntegerMultiply = false;
  AmeRoleCapability a;
  AmeRoleCapability b;
  AmeRoleCapability accumulator;
};

/// Hardware facts known to the BOSCAME dialect boundary. Unmeasured contracts
/// such as RVV width remain unavailable instead of being inferred.
struct TargetInfo {
  llvm::StringRef name;
  int64_t spmBankCount = 0;
  int64_t spmBankCapacityBytes = 0;
  int64_t dmaChannelCount = 0;
  AmeCapability ame;
  std::optional<int64_t> rvvVectorBytes;
  TransferCapability dma;
  // Capacity supported by the current allocator/encoder, not a speculative
  // hardware register count. Unknown RVV width remains independent.
  int64_t allocatableTileRegisters = 8;
  int64_t allocatableAccumulatorRegisters = 8;
  int64_t scalarFloatRegisters = 32;
};

inline constexpr TargetInfo getNrFpgaTargetInfo() {
  return {"nr-fpga",
          spmdma::kSpmBankCount,
          spmdma::kSpmBankCapacityBytes,
          spmdma::kDmaChannelCount,
          {kAmeMaxTileDim,
           kAmeMaxTileDim,
           kAmeMaxTileDim,
           true,
           {8, AmeTileLayout::MxKWithKContiguous, true, true},
           {8, AmeTileLayout::KxNWithPhysicalNxK, true, true},
           {32, AmeTileLayout::MxNWithNContiguous, true, true}},
          std::nullopt,
          {spmdma::kDmaAlignmentBytes, spmdma::kDmaAlignmentBytes, 2, true,
           true, true, true}};
}

inline std::optional<TargetInfo> resolveTargetInfo(llvm::StringRef name) {
  if (name.empty() || name == "nr-fpga")
    return getNrFpgaTargetInfo();
  return std::nullopt;
}

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMETARGETINFO_H
