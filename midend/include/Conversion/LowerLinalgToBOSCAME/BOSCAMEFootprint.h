//===- BOSCAMEFootprint.h - Memref transfer footprint analysis -*- C++ -*-===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_FOOTPRINT_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_FOOTPRINT_H

#include "Analysis/BOSCAMELaunchFacts.h"
#include "Analysis/LinalgComputationFacts.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMETargetInfo.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class Operation;
class Type;
class Value;
} // namespace mlir

namespace buddy::boscame {

enum class TransferRole { InputA, InputB, OutputC, Temporary };

struct DmaPacket {
  int64_t ddrOffsetBytes = 0;
  int64_t spmOffsetBytes = 0;
  int64_t bytes = 0;
};

struct TransferPlan {
  TransferRole role = TransferRole::Temporary;
  /// The logical operand and its storage root. Packet DDR offsets are
  /// relative to the operand's effective address, never the root twice.
  mlir::Value buffer;
  mlir::Value ddrRoot;
  llvm::SmallVector<mlir::Operation *> viewChain;
  std::optional<int64_t> sourceOriginId;
  std::optional<int64_t> effectiveByteOffsetBytes;
  bool endpointAlignmentProven = false;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> spmStrides;
  llvm::SmallVector<int64_t> validShape;
  mlir::Type elementType;
  int64_t elementBytes = 0;
  llvm::SmallVector<DmaPacket> packets;
  /// A recognized dynamic rank-one prefix copy. Packets remain fixed size and
  /// are emitted only when `validElements` proves the whole packet is valid.
  mlir::Value validElements;
  mlir::Value prefixPeerBuffer;
  bool guardedPrefix = false;
  bool prefixLoad = false;
  int64_t spmBank = -1;
  int64_t spmOffset = 0;
  int64_t spmCapacity = 0;
  int64_t channel = 0;
  bool read = true;
  bool write = false;
  mlir::Operation *firstConsumer = nullptr;
  mlir::Operation *lastConsumer = nullptr;
  /// All straight-line consumers for a proven read-only resident value.  The
  /// legacy first/last pointers remain the emitter's insertion boundaries.
  llvm::SmallVector<mlir::Operation *, 4> consumerOperations;
  bool crossTaskResident = false;
  /// A planned external boundary copy uses an existing resident window.
  /// Packet offsets remain relative to the endpoint descriptors; spmOffset
  /// and spmCapacity describe the fixed root reservation for accounting.
  bool directExternalBoundary = false;
  mlir::Value residentBuffer;
  mlir::Value source;
  mlir::Value target;
  mlir::Operation *provider = nullptr;
  mlir::Operation *completionOperation = nullptr;
};

enum class ComputeGraphStatus { Unavailable, Incomplete, Complete };

struct PlannedTask {
  mlir::Operation *operation = nullptr;
  int64_t id = -1;
  int64_t stage = -1;
  mlir::StringAttr kind;
  llvm::SmallVector<int64_t> predecessors;
  llvm::SmallVector<int64_t> inputValues;
  llvm::SmallVector<int64_t> outputValues;
  llvm::SmallVector<int64_t> varyingProgramAxes;
  bool barrierAfter = false;
  bool rematerializable = false;
};

struct TaskDependency {
  int64_t predecessor = -1;
  int64_t consumer = -1;
  bool resolved = false;
};

/// A logical value's producer and consumers are graph facts only. They do not
/// establish a physical buffer lifetime or justify extending an SPM window.
struct LogicalValuePlan {
  int64_t id = -1;
  std::optional<int64_t> producerTask;
  llvm::SmallVector<int64_t> consumerTasks;
  bool producerIsFunctionArgument = false;
  bool lifetimeComplete = false;
};

/// Physical storage requirements for operation-local staging and the narrow
/// proven cross-task resident case. Global residency is not inferred from
/// logical value lifetimes.
struct BufferRequirement {
  size_t transferIndex = 0;
  mlir::Value buffer;
  mlir::Value storageRoot;
  /// Definite root identity only; different IDs do not prove no aliasing.
  int64_t storageRootId = -1;
  llvm::SmallVector<mlir::Operation *> viewChain;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> strides;
  std::optional<int64_t> effectiveByteOffsetBytes;
  int64_t capacityBytes = 0;
  std::optional<int64_t> spmFirstTask;
  std::optional<int64_t> spmLastTask;
  bool operationLocalLifetime = false;
  bool crossTaskResident = false;
  llvm::SmallVector<int64_t> physicalUseTasks;
  std::optional<int64_t> physicalFirstTask;
  std::optional<int64_t> physicalLastTask;
  bool physicalLifetimeKnown = false;
};

enum class BufferAccessEffect { Read, Write, Allocate, Free, Unknown };

enum class BufferAccessRoute { Unassessed, DirectDdr, ExistingSpm, Staging };

struct RootStorageAccess {
  mlir::Operation *operation = nullptr;
  unsigned position = 0;
  size_t accessIndex = 0;
};

/// Physical storage identity is independent of the contents written into it.
/// Positions are lexical preorder positions, not dynamic execution order.
/// Structured scopes use fixed windows, without content-version lifetime reuse.
struct RootStorageFact {
  mlir::Value root;
  llvm::SmallVector<mlir::Value> views;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> strides;
  int64_t capacityBytes = 0;
  llvm::SmallVector<RootStorageAccess> readers;
  llvm::SmallVector<RootStorageAccess> writers;
  /// Read traffic avoided relative to independent staging of each descriptor
  /// at each operation, before the older narrow residency search.
  uint64_t potentialReadSavingsBytes = 0;
  unsigned firstPosition = 0;
  unsigned lastPosition = 0;
  bool aliasProven = false;
  bool feedsAmeDirectRoute = false;
  bool structuredScope = false;
};

/// One write's content, including partial initialization of padded storage.
/// A physical root may hold several unrelated logical values in sequence.
struct ContentVersion {
  size_t rootIndex = 0;
  mlir::Operation *writer = nullptr;
  unsigned position = 0;
  mlir::Value writtenView;
  std::optional<int64_t> byteOffset;
  std::optional<int64_t> bytes;
  llvm::SmallVector<int64_t> validShape;
  llvm::SmallVector<mlir::Operation *> consumers;
  bool rematerializable = false;
  bool paddingInitialization = false;
};

enum class BufferPlacementKind { Spare, SpmResident, StagedFallback };
enum class BufferPlacementReason {
  Selected,
  AmeDirectRoute,
  Auxiliary,
  Unused,
  UnsupportedLayout,
  UnsupportedUse,
  ControlFlow,
  AliasUnknown,
  MissingStaging,
  Capacity,
  ComputationLegality
};

struct BufferPlacementDecision {
  size_t rootIndex = 0;
  int64_t group = -1;
  BufferPlacementKind kind = BufferPlacementKind::Spare;
  BufferPlacementReason reason = BufferPlacementReason::Unused;
  std::string detail;
  int64_t bank = -1;
  int64_t offset = 0;
  bool applied = false;
};

/// A buffer access at the current analysis point. The final plan report uses
/// surviving post-planning IR; pre-placement facts also guard residency.
/// Unknown address or alias facts never create a transfer requirement.
struct BufferAccessFact {
  mlir::Operation *operation = nullptr;
  int64_t effectIndex = -1;
  int64_t operandNumber = -1;
  BufferAccessEffect effect = BufferAccessEffect::Unknown;
  mlir::Value accessedValue;
  mlir::Value storageRoot;
  int64_t storageRootId = -1;
  bool storageRootKnown = false;
  llvm::SmallVector<mlir::Operation *> viewChain;
  mlir::Type accessedType;
  mlir::Attribute memorySpace;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> logicalShape;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> validShape;
  std::optional<int64_t> sourceOriginId;
  std::optional<int64_t> computeTaskId;
  /// Frontend logical read/write slot, only when effect arity still matches.
  std::optional<int64_t> logicalValueId;
  std::optional<int64_t> effectiveByteOffsetBytes;
  std::optional<int64_t> logicalFootprintBytes;
  std::optional<int64_t> physicalSpanBytes;
  std::optional<size_t> transferIndex;
  /// The selected route belongs to this access, independently of the buffer's
  /// logical contents and the physical placement of other operands.
  BufferAccessRoute route = BufferAccessRoute::Unassessed;
  std::string routeReason;
  std::optional<uint64_t> routeBytes;
  int64_t routeBank = -1;
  int64_t routeOffset = 0;
  int64_t routeCapacityBytes = 0;
};

/// A deduplicated descriptor observed by a post-planning memory effect. A
/// static root capacity may be reported, but this inventory does not establish
/// physical lifetime.
struct BufferInventoryFact {
  mlir::StringAttr functionName;
  mlir::Value descriptor;
  mlir::Value storageRoot;
  int64_t storageRootId = -1;
  bool storageRootKnown = false;
  llvm::SmallVector<mlir::Operation *> viewChain;
  mlir::Type descriptorType;
  mlir::Attribute memorySpace;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> strides;
  std::optional<int64_t> effectiveByteOffsetBytes;
  std::optional<int64_t> rootCapacityBytes;
  llvm::SmallVector<size_t> accessIndices;
  llvm::SmallVector<int64_t> logicalValueIds;
  llvm::SmallVector<int64_t> taskIds;
  llvm::SmallVector<size_t> transferIndices;
  bool physicalLifetimeKnown = false;
};

/// Detached provenance-bearing buffer event captured before planner rewrites.
/// It intentionally holds no Operation or Value handles that could dangle
/// when fusion removes the source event.
struct BufferSourceEventFact {
  mlir::StringAttr functionName;
  std::string operationName;
  int64_t effectIndex = -1;
  int64_t operandNumber = -1;
  BufferAccessEffect effect = BufferAccessEffect::Unknown;
  int64_t sourceOriginId = -1;
  mlir::Type accessedType;
  mlir::Attribute memorySpace;
  llvm::SmallVector<std::string> viewChain;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> logicalShape;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> validShape;
  std::optional<int64_t> effectiveByteOffsetBytes;
  std::optional<int64_t> logicalFootprintBytes;
  std::optional<int64_t> physicalSpanBytes;
};

enum class BufferAliasRelation { MayAlias, Disjoint };
enum class BufferAliasProof {
  SameRoot,
  SameRootByteIntervals,
  SameRootSymbolicByteIntervals,
  SameRootPitchedRowIntervals,
  AbiNonAliasingEntryArguments,
  DistinctGlobalSymbols,
  FreshAllocation
};

/// A reportable pairwise relation. Pairs with unresolved distinct roots are
/// omitted and remain unknown; same-root pairs are always represented.
struct BufferAliasPair {
  size_t firstAccessIndex = 0;
  size_t secondAccessIndex = 0;
  mlir::StringAttr functionName;
  BufferAliasRelation relation = BufferAliasRelation::MayAlias;
  BufferAliasProof proof = BufferAliasProof::SameRoot;
};

enum class MemoryDependencyKind { RAW, WAR, WAW, UnknownEffect, Lifetime };

/// A possible memory hazard in proven execution order. These edges are not
/// value provenance: WAR/WAW must never propagate source-load identities.
struct MemoryDependencyFact {
  size_t predecessorAccessIndex = 0;
  size_t successorAccessIndex = 0;
  mlir::StringAttr functionName;
  MemoryDependencyKind kind = MemoryDependencyKind::UnknownEffect;
};

/// Facts collected before placement. Missing alias proofs remain unknown;
/// incomplete or bounded memory ordering cannot certify a residency window.
struct KernelFacts {
  /// Explicit current-IR SSA structure. It does not replace semantic graph,
  /// alias, execution-order, or residency completeness proofs.
  LinalgComputationFacts linalg;
  llvm::SmallVector<LaunchFunctionFact, 4> launches;
  llvm::SmallVector<PlannedTask, 8> tasks;
  llvm::SmallVector<TaskDependency, 8> dependencies;
  llvm::SmallVector<LogicalValuePlan, 8> logicalValues;
  llvm::SmallVector<BufferSourceEventFact, 4> bufferSourceEvents;
  llvm::SmallVector<BufferAccessFact, 8> bufferAccesses;
  llvm::SmallVector<BufferInventoryFact, 8> bufferInventory;
  llvm::SmallVector<BufferAliasPair, 8> bufferAliasPairs;
  bool bufferAliasPairsComplete = true;
  llvm::SmallVector<MemoryDependencyFact, 8> memoryDependencies;
  bool memoryDependenciesComplete = true;
  bool memoryDependenciesBounded = false;
  ComputeGraphStatus computeGraphStatus = ComputeGraphStatus::Unavailable;
};

struct ResourceEstimate {
  uint64_t plannedTransfers = 0;
  uint64_t plannedDmaPackets = 0;
  uint64_t plannedReadBytes = 0;
  uint64_t plannedWriteBytes = 0;
  bool transferTotalsExact = true;
  uint64_t bankAllocationGroups = 0;
  uint64_t bankAllocationCandidates = 0;
  uint64_t bankAllocationRejectedCapacity = 0;
  uint64_t bankAllocationBoundedGroups = 0;
  uint64_t bankAllocationRepairGroups = 0;
  uint64_t bankAllocationRepairSearchNodes = 0;
  uint64_t bankAllocationRepairRejectedPlacements = 0;
};

enum class ResidencyDecisionReason { Selected, Capacity, NoTrafficBenefit };

/// Static traffic comparison for a proven shared read-only function argument.
/// Latency is intentionally absent until target costs are calibrated.
struct FunctionArgumentResidencyDecision {
  mlir::Value argument;
  int64_t capacityBytes = 0;
  uint64_t consumerCount = 0;
  uint64_t residentReadBytes = 0;
  uint64_t residentPackets = 0;
  uint64_t reloadReadBytes = 0;
  uint64_t reloadPackets = 0;
  ResidencyDecisionReason reason = ResidencyDecisionReason::NoTrafficBenefit;
};

/// Static traffic savings for retaining one staged producer output across its
/// proven read-only consumers. The producer writeback is common to both
/// strategies; residency removes the consumers' DDR reloads.
struct ProducerOutputResidencyDecision {
  mlir::Value output;
  mlir::Operation *producer = nullptr;
  int64_t capacityBytes = 0;
  uint64_t consumerCount = 0;
  uint64_t savedReadBytes = 0;
  uint64_t savedPackets = 0;
  bool selected = false;
};

struct CostEstimate {
  bool latencyCalibrated = false;
  std::optional<double> latencyCycles;
};

/// Effective AME tile geometry selected by the planner for one marked
/// matmul. `tail*` records the actual extent of the final tile, including a
/// full tile when the corresponding dimension divides evenly.
struct AmeMatmulTilePlan {
  mlir::Operation *matmul = nullptr;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  int64_t mTileCount = 0;
  int64_t nTileCount = 0;
  int64_t kTileCount = 0;
  int64_t tailM = 0;
  int64_t tailN = 0;
  int64_t tailK = 0;
  /// Exact static read traffic considered for the logical A/B route choice.
  /// Zero means that the corresponding product was not representable.
  uint64_t directAReadBytes = 0;
  uint64_t stagedAReadBytes = 0;
  uint64_t directBReadBytes = 0;
  uint64_t stagedBReadBytes = 0;
  bool directAByCapacity = false;
  bool directBByCapacity = false;
  bool directABySpmDisabled = false;
  bool directBBySpmDisabled = false;
  bool directAEligible = false;
  bool directBEligible = false;
};

/// Dependency-aware plan facts alongside the emitter's current transfer
/// payload. General placement and synchronization remain operation-local;
/// proven same-SSA read-only residency is represented on TransferPlan.
struct DataflowPlan : KernelFacts {
  TargetInfo target = getNrFpgaTargetInfo();
  llvm::SmallVector<AmeMatmulTilePlan, 4> ameMatmulTiles;
  llvm::SmallVector<TransferPlan, 4> transfers;
  llvm::SmallVector<BufferRequirement, 4> buffers;
  llvm::SmallVector<FunctionArgumentResidencyDecision, 4>
      functionArgumentResidency;
  llvm::SmallVector<ProducerOutputResidencyDecision, 4> producerOutputResidency;
  llvm::SmallVector<RootStorageFact, 0> rootStorageFacts;
  llvm::SmallVector<ContentVersion, 0> contentVersions;
  llvm::SmallVector<BufferPlacementDecision> bufferPlacements;
  bool functionArgumentResidencySearchBounded = false;
  bool producerOutputResidencySearchBounded = false;
  ResourceEstimate resources;
  CostEstimate cost;
  int64_t bankCapacity = getNrFpgaTargetInfo().spmBankCapacityBytes;
  int64_t usedBytes[2] = {0, 0};
};

/// Find the planner-owned AME tile geometry for a marked matmul.
AmeMatmulTilePlan *findAmeMatmulTilePlan(DataflowPlan &plan,
                                         mlir::Operation *matmul);
const AmeMatmulTilePlan *findAmeMatmulTilePlan(const DataflowPlan &plan,
                                               mlir::Operation *matmul);

using WorkspacePlan = DataflowPlan;

/// Return the byte width accepted by the typed DMA contract, or nullopt.
std::optional<int64_t> getBOSCAMEElementBytes(mlir::Type type);

/// Canonicalize a storage root that is a memref.get_global result. Separate
/// handles naming the same global symbol represent the same storage.
mlir::Value canonicalizeBOSCAMEGlobalRoot(mlir::Value root);

/// Analyze a ranked memref's physical footprint.  This is intentionally a
/// pure query: it does not insert casts, bufferization, or runtime guesses.
/// A failure diagnoses dynamic shapes, unsupported element types, negative or
/// non-positive strides, and overflowing byte ranges.
std::optional<TransferPlan> analyzeBOSCAMEFootprint(mlir::Value value,
                                                    TransferRole role,
                                                    mlir::Operation *anchor,
                                                    const TargetInfo &target);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_FOOTPRINT_H
