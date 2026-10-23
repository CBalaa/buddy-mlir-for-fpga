//===- BOSCAMEPlacement.cpp - Root storage placement ---------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlacement.h"
#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEAmeEmitter.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEBoundaryTransfers.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanner.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace ::buddy::boscame;

namespace {

struct RootRewrite {
  size_t rootIndex;
  size_t decisionIndex;
  Value root;
  SmallVector<Value> views;
  SmallVector<Operation *> deallocations;
  int64_t bank;
  int64_t offset;
};

bool isStorageView(Operation *operation) {
  return isa<memref::CastOp, memref::ReinterpretCastOp, memref::SubViewOp,
             memref::TransposeOp, memref::MemorySpaceCastOp, memref::ViewOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(operation);
}

LogicalResult collectRootRewrite(const RootStorageFact &fact,
                                 const BufferPlacementDecision &decision,
                                 RootRewrite &rewrite, std::string &reason) {
  auto reject = [&](StringRef message) {
    reason = message.str();
    return failure();
  };
  Operation *allocation = fact.root.getDefiningOp();
  auto type = dyn_cast<MemRefType>(fact.root.getType());
  if (!allocation || !isa<memref::AllocOp, memref::AllocaOp>(allocation) ||
      !type || spmdma::getSpmBank(type))
    return reject("resident root must be an ordinary alloc or alloca");
  if (!hasBOSCAMEPlacementScope(allocation))
    return reject("resident root requires a serial single-block SCF scope");
  auto bytes = spmdma::getContiguousZeroOffsetByteSize(type);
  if (!bytes || *bytes != fact.capacityBytes || *bytes <= 0 ||
      type.getRank() < 1 || type.getRank() > 2 ||
      *bytes % spmdma::kDmaAlignmentBytes)
    return reject("resident root requires a static aligned contiguous layout");
  rewrite.root = fact.root;
  rewrite.bank = decision.bank;
  rewrite.offset = decision.offset;
  rewrite.views.push_back(fact.root);
  for (size_t i = 0; i < rewrite.views.size(); ++i) {
    for (Operation *user : rewrite.views[i].getUsers()) {
      if (!hasBOSCAMEPlacementScope(user))
        return reject("resident root use escapes its serial SCF scope");
      if (isStorageView(user)) {
        if (user->getNumResults() != 1 ||
            !isa<MemRefType>(user->getResult(0).getType()))
          return reject("resident root has an unsupported storage view");
        Value view = user->getResult(0);
        if (!llvm::is_contained(rewrite.views, view))
          rewrite.views.push_back(view);
      } else if (isa<memref::DeallocOp>(user)) {
        if (user->hasAttr("bosc_ame.compute_id"))
          return reject("resident deallocation is a compute graph task");
        if (!llvm::is_contained(rewrite.deallocations, user))
          rewrite.deallocations.push_back(user);
      } else if (!isa<linalg::FillOp, linalg::GenericOp, linalg::ReduceOp,
                      linalg::MatmulOp, memref::CopyOp, memref::DimOp,
                      vector::LoadOp, vector::StoreOp>(user)) {
        return reject(
            "resident root has an unsupported scalar or escaping use");
      }
    }
  }
  return success();
}

LogicalResult collectRewrites(ModuleOp module, const DataflowPlan &plan,
                              SmallVectorImpl<RootRewrite> &rewrites,
                              std::string &reason) {
  struct Window {
    int64_t bank, begin, end;
    Operation *owner;
  };
  SmallVector<Window> windows;
  bool invalidWindow = false;
  module.walk([&](SpmViewOp view) {
    auto bytes = spmdma::getContiguousZeroOffsetByteSize(view.getType());
    if (!bytes || *bytes <= 0 || *bytes > plan.bankCapacity ||
        view.getBank() >= uint64_t(plan.target.spmBankCount) ||
        view.getOffset() > uint64_t(plan.bankCapacity - *bytes)) {
      invalidWindow = true;
      return;
    }
    Window candidate{view.getBank(), int64_t(view.getOffset()),
                     int64_t(view.getOffset()) + *bytes, nullptr};
    for (const Window &window : windows)
      if (candidate.bank == window.bank && candidate.begin < window.end &&
          window.begin < candidate.end)
        invalidWindow = true;
    windows.push_back(candidate);
  });
  if (invalidWindow) {
    reason = "explicit SPM windows are invalid or overlap";
    return failure();
  }
  DenseSet<Value> roots;
  for (auto [index, decision] : llvm::enumerate(plan.bufferPlacements)) {
    if (decision.kind != BufferPlacementKind::SpmResident || decision.applied)
      continue;
    if (decision.rootIndex >= plan.rootStorageFacts.size()) {
      reason = "resident decision refers to a missing root fact";
      return failure();
    }
    const RootStorageFact &fact = plan.rootStorageFacts[decision.rootIndex];
    if (!roots.insert(fact.root).second) {
      reason = "resident root has multiple placement decisions";
      return failure();
    }
    if (decision.bank < 0 || decision.bank >= plan.target.spmBankCount ||
        decision.offset < 0 || decision.offset % spmdma::kDmaAlignmentBytes ||
        fact.capacityBytes <= 0 || fact.capacityBytes > plan.bankCapacity ||
        decision.offset > plan.bankCapacity - fact.capacityBytes) {
      reason = "resident window exceeds its aligned SPM bank reservation";
      return failure();
    }
    Window candidate{decision.bank, decision.offset,
                     decision.offset + fact.capacityBytes,
                     fact.root.getDefiningOp()};
    for (const Window &window : windows)
      if (candidate.bank == window.bank && candidate.begin < window.end &&
          window.begin < candidate.end &&
          !haveExclusiveBOSCAMEPlacementScopes(candidate.owner, window.owner)) {
        reason = "resident window overlaps an explicit or resident SPM window";
        return failure();
      }
    RootRewrite rewrite;
    rewrite.rootIndex = decision.rootIndex;
    rewrite.decisionIndex = index;
    if (failed(collectRootRewrite(fact, decision, rewrite, reason)))
      return failure();
    windows.push_back(candidate);
    rewrites.push_back(std::move(rewrite));
  }
  return success();
}

/// A buffer substitution changes storage, not content or task identity. Move
/// the allocation's complete graph/provenance annotation to its replacement;
/// all existing view and computation annotations continue to name the same
/// logical contents and consumers. In particular, no compute ID is invented.
void updateBufferPlacementComputeGraph(Operation *allocation,
                                       Operation *replacement) {
  for (NamedAttribute attribute : allocation->getAttrs())
    if (attribute.getName().getValue().starts_with("bosc_ame."))
      replacement->setAttr(attribute.getName(), attribute.getValue());
}

Value rewriteRoot(const RootRewrite &rewrite, IRMapping *mapping = nullptr) {
  Value root = mapping ? mapping->lookup(rewrite.root) : rewrite.root;
  Operation *allocation = root.getDefiningOp();
  OpBuilder builder(allocation);
  auto type = cast<MemRefType>(root.getType());
  Attribute space = SPMAttr::get(builder.getContext(), rewrite.bank);
  auto residentType = MemRefType::get(type.getShape(), type.getElementType(),
                                      type.getLayout(), space);
  auto view = SpmViewOp::create(builder, allocation->getLoc(), residentType,
                                builder.getI32IntegerAttr(rewrite.bank),
                                builder.getI64IntegerAttr(rewrite.offset));
  updateBufferPlacementComputeGraph(allocation, view);
  for (Value original : llvm::drop_begin(rewrite.views)) {
    Value alias = mapping ? mapping->lookup(original) : original;
    auto oldType = cast<MemRefType>(alias.getType());
    alias.setType(MemRefType::get(oldType.getShape(), oldType.getElementType(),
                                  oldType.getLayout(), space));
  }
  for (Operation *deallocation : rewrite.deallocations)
    (mapping ? mapping->lookup(deallocation) : deallocation)->erase();
  root.replaceAllUsesWith(view.getView());
  if (mapping) {
    mapping->map(rewrite.root, view.getView());
    mapping->map(rewrite.root.getDefiningOp(), view.getOperation());
  }
  allocation->erase();
  return view.getView();
}

LogicalResult verifyPlacedClone(ModuleOp module) {
  if (failed(verify(module)) || failed(verifyCompleteComputeGraphs(module)) ||
      failed(verifyBOSCAMEPlanningSurface(module)))
    return failure();
  WalkResult result = module.walk([](SpmViewOp view) {
    return failed(view.verify()) ? WalkResult::interrupt()
                                 : WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult stageCloneOperand(
    const TransferPlan &transfer, const TargetInfo &target, IRMapping &mapping,
    const llvm::SmallPtrSetImpl<Operation *> &boundaries, std::string &reason) {
  SmallVector<Operation *> consumers(transfer.consumerOperations.begin(),
                                     transfer.consumerOperations.end());
  if (consumers.empty() && transfer.firstConsumer)
    consumers.push_back(transfer.firstConsumer);
  if (consumers.empty()) {
    reason = "staging requirement has no computation consumer";
    return failure();
  }
  if (transfer.elementBytes <= 0 ||
      transfer.shape.size() != transfer.spmStrides.size() ||
      transfer.spmBank < 0 || transfer.spmBank >= target.spmBankCount ||
      transfer.spmOffset < 0 || transfer.spmCapacity <= 0 ||
      transfer.spmCapacity % transfer.elementBytes ||
      transfer.spmOffset % spmdma::kDmaAlignmentBytes ||
      transfer.spmCapacity > target.spmBankCapacityBytes ||
      transfer.spmOffset > target.spmBankCapacityBytes - transfer.spmCapacity) {
    reason = "staging requirement has no valid aligned SPM reservation";
    return failure();
  }
  for (Operation *original : consumers) {
    Operation *consumer = mapping.lookupOrNull(original);
    Value operand = mapping.lookupOrNull(transfer.buffer);
    if (!consumer || !operand) {
      reason =
          "staging requirement refers to an unavailable operation or buffer";
      return failure();
    }
    if (boundaries.contains(consumer))
      continue;
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || spmdma::getSpmBank(type))
      continue;
    OpBuilder builder(consumer);
    Attribute space = SPMAttr::get(builder.getContext(), transfer.spmBank);
    auto stagedType = MemRefType::get(
        transfer.shape, transfer.elementType,
        StridedLayoutAttr::get(builder.getContext(), 0, transfer.spmStrides),
        space);
    auto flatType = MemRefType::get(
        {transfer.spmCapacity / transfer.elementBytes}, transfer.elementType,
        MemRefLayoutAttrInterface(), space);
    Value flat =
        SpmViewOp::create(builder, consumer->getLoc(), flatType,
                          builder.getI32IntegerAttr(transfer.spmBank),
                          builder.getI64IntegerAttr(transfer.spmOffset));
    Value staged = memref::ReinterpretCastOp::create(
        builder, consumer->getLoc(), stagedType, flat, 0, transfer.shape,
        transfer.spmStrides);
    for (OpOperand &use : consumer->getOpOperands())
      if (use.get() == operand)
        use.set(staged);
  }
  return success();
}

} // namespace

namespace buddy::boscame {

bool hasBOSCAMEPlacementScope(Operation *operation) {
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto function = dyn_cast<func::FuncOp>(parent))
      return function.getBody().hasOneBlock();
    if (!isa<scf::ForOp, scf::IfOp, scf::ExecuteRegionOp>(parent))
      return false;
    for (Region &region : parent->getRegions())
      if (!region.empty() && !region.hasOneBlock())
        return false;
  }
  return false;
}

bool haveExclusiveBOSCAMEPlacementScopes(Operation *first, Operation *second) {
  for (Operation *child = first; child; child = child->getParentOp()) {
    auto branch = dyn_cast_or_null<scf::IfOp>(child->getParentOp());
    if (!branch)
      continue;
    Operation *other = second;
    while (other && other->getParentOp() != branch)
      other = other->getParentOp();
    if (other && child->getParentRegion() != other->getParentRegion())
      return true;
  }
  return false;
}

LogicalResult preflightBufferPlacements(ModuleOp module,
                                        const DataflowPlan &plan,
                                        ArrayRef<TransferPlan> staging,
                                        std::string &reason) {
  reason.clear();
  SmallVector<RootRewrite> rewrites;
  if (failed(collectRewrites(module, plan, rewrites, reason)))
    return failure();
  if (rewrites.empty())
    return success();
  llvm::SmallPtrSet<Operation *, 16> computations;
  for (const RootRewrite &rewrite : rewrites)
    for (Value view : rewrite.views)
      for (Operation *user : view.getUsers())
        if (isa<linalg::LinalgOp, memref::CopyOp>(user))
          computations.insert(user);

  ScopedDiagnosticHandler capture(module.getContext(), [&](Diagnostic &diag) {
    llvm::raw_string_ostream stream(reason);
    if (!reason.empty())
      stream << "; ";
    diag.print(stream);
    return success();
  });
  IRMapping mapping;
  OwningOpRef<ModuleOp> clone(cast<ModuleOp>(module->clone(mapping)));
  for (const RootRewrite &rewrite : rewrites)
    rewriteRoot(rewrite, &mapping);
  if (failed(verifyPlacedClone(*clone)))
    return failure();

  // A proved boundary is a DMA event, so generic vector placement checks do
  // not apply to its external DDR endpoint. Prove it before staging would
  // obscure the actual source/destination relationship.
  llvm::SmallPtrSet<Operation *, 16> boundaries;
  for (Operation *original : computations) {
    Operation *operation = mapping.lookup(original);
    if (planResidentBoundaryTransfer(operation, plan.target))
      boundaries.insert(operation);
  }
  for (const TransferPlan &transfer : staging)
    if (failed(stageCloneOperand(transfer, plan.target, mapping, boundaries,
                                 reason)))
      return failure();
  for (Operation *original : computations) {
    Operation *operation = mapping.lookup(original);
    if (isa<linalg::MatmulOp>(operation)) {
      if (failed(verifyBOSCAMEPlacedMatmul(operation, plan.target)))
        return failure();
      continue;
    }
    if (!boundaries.contains(operation) &&
        failed(lowering::checkSpmComputation(operation)))
      return failure();
  }
  return verifyPlacedClone(*clone);
}

LogicalResult applyBufferPlacements(ModuleOp module, DataflowPlan &plan) {
  SmallVector<RootRewrite> rewrites;
  std::string reason;
  if (failed(collectRewrites(module, plan, rewrites, reason)))
    return module.emitError("invalid BOSCAME buffer placement: ") << reason;
  if (rewrites.empty())
    return success();
  IRMapping mapping;
  OwningOpRef<ModuleOp> clone(cast<ModuleOp>(module->clone(mapping)));
  for (const RootRewrite &rewrite : rewrites)
    rewriteRoot(rewrite, &mapping);
  if (failed(verifyPlacedClone(*clone)))
    return failure();
  // No legality decision remains after the detached rewrite verifies. The
  // live rewrite performs exactly the same substitutions and retains task
  // IDs, predecessors, input/output IDs and reverse logical users verbatim.
  for (const RootRewrite &rewrite : rewrites) {
    Value oldRoot = rewrite.root;
    Value resident = rewriteRoot(rewrite);
    RootStorageFact &fact = plan.rootStorageFacts[rewrite.rootIndex];
    fact.root = resident;
    for (Value &view : fact.views)
      if (view == oldRoot)
        view = resident;
    for (ContentVersion &version : plan.contentVersions)
      if (version.rootIndex == rewrite.rootIndex &&
          version.writtenView == oldRoot)
        version.writtenView = resident;
    plan.bufferPlacements[rewrite.decisionIndex].applied = true;
  }
  return verifyPlacedClone(module);
}

} // namespace buddy::boscame
