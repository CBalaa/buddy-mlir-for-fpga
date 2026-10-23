//===- BOSCAMEPlanner.cpp - Buffer-form Linalg planner --------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanner.h"
#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEBoundaryTransfers.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlacement.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "Dialect/BOSCAME/Transforms/BOSCAMETileLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>

using namespace mlir;
using namespace ::buddy::boscame;

static constexpr StringLiteral kDirectAOperandAttr = "bosc_ame.direct_ddr_a";
static constexpr StringLiteral kDirectBOperandAttr = "bosc_ame.direct_ddr_b";
static constexpr StringLiteral kDirectCOperandAttr = "bosc_ame.direct_ddr_c";
static constexpr StringLiteral kPrefixRangeAttr = "bosc_ame.prefix_range";
static constexpr StringLiteral kPrefixMaskAuxAttr = "bosc_ame.prefix_mask_aux";

static Value getStorageRoot(Value value);
static bool isStorageViewOperation(Operation *operation);
static bool isKnownBufferStorageRoot(Value root);

static SmallVector<int64_t> getI64Array(Operation *operation, StringRef name) {
  auto values = operation->getAttrOfType<DenseI64ArrayAttr>(name);
  return values ? SmallVector<int64_t>(values.asArrayRef())
                : SmallVector<int64_t>();
}

/// Integer task references are not checked by MLIR's SSA verifier. An
/// explicitly complete graph must be structurally valid before any rewrite
/// can erase the evidence, and must remain valid after planning rewrites.
/// Unsupported/incomplete analyses still use the ordinary fallback path.
namespace buddy::boscame {
LogicalResult verifyCompleteComputeGraphs(ModuleOp module) {
  auto version =
      module->getAttrOfType<IntegerAttr>("bosc_ame.compute_graph_version");
  if (!version || version.getInt() != 1)
    return success();
  auto claimsComplete = [](func::FuncOp function) {
    auto complete =
        function->getAttrOfType<BoolAttr>("bosc_ame.compute_graph_complete");
    return complete && complete.getValue();
  };
  DenseMap<int64_t, Operation *> tasks;
  LogicalResult status = success();
  module.walk([&](func::FuncOp function) {
    if (function.isExternal() || failed(status))
      return;
    int64_t count = 0;
    function.walk([&](Operation *operation) {
      if (failed(status) || !operation->hasAttr("bosc_ame.compute_id"))
        return;
      ++count;
      auto id = operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
      if (!id || id.getInt() < 0) {
        if (claimsComplete(function))
          status = operation->emitError("invalid BOSCAME compute task ID");
        return;
      }
      auto inserted = tasks.try_emplace(id.getInt(), operation);
      if (!inserted.second &&
          (claimsComplete(function) ||
           claimsComplete(
               inserted.first->second->getParentOfType<func::FuncOp>()))) {
        status = operation->emitError("duplicate BOSCAME compute task ID ")
                 << id.getInt();
        return;
      }
      if (!claimsComplete(function))
        return;
      if (!operation->getAttrOfType<StringAttr>("bosc_ame.compute_kind") ||
          !operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_stage") ||
          !operation->getAttrOfType<BoolAttr>(
              "bosc_ame.compute_barrier_after") ||
          !operation->getAttrOfType<BoolAttr>("bosc_ame.rematerializable")) {
        status =
            operation->emitError("incomplete BOSCAME compute task metadata");
        return;
      }
      for (StringRef name :
           {"bosc_ame.compute_predecessors", "bosc_ame.compute_input_values",
            "bosc_ame.compute_output_values"})
        if (!operation->getAttrOfType<DenseI64ArrayAttr>(name)) {
          status = operation->emitError("missing or invalid BOSCAME attribute ")
                   << name;
          return;
        }
      for (StringRef name :
           {"bosc_ame.compute_input_values", "bosc_ame.compute_output_values"})
        for (int64_t value : getI64Array(operation, name))
          if (value < 0) {
            status =
                operation->emitError("negative BOSCAME logical value ID in ")
                << name;
            return;
          }
    });
    if (failed(status) || !claimsComplete(function))
      return;
    auto expected = function->getAttrOfType<IntegerAttr>(
        "bosc_ame.compute_graph_node_count");
    if (!expected || expected.getInt() != count)
      status = function.emitError("BOSCAME compute graph node count mismatch: ")
               << "expected " << (expected ? expected.getInt() : -1)
               << ", found " << count;
  });
  if (failed(status))
    return failure();
  for (auto [id, operation] : tasks) {
    auto function = operation->getParentOfType<func::FuncOp>();
    if (!claimsComplete(function))
      continue;
    for (int64_t predecessor :
         getI64Array(operation, "bosc_ame.compute_predecessors")) {
      auto producer = tasks.find(predecessor);
      if (producer == tasks.end() ||
          producer->second->getParentOfType<func::FuncOp>() != function)
        return operation->emitError("unresolved BOSCAME compute predecessor ")
               << predecessor << " for task " << id;
      if (predecessor >= id)
        return operation->emitError(
                   "invalid BOSCAME compute predecessor order: ")
               << predecessor << " -> " << id;
    }
  }
  module.walk([&](func::FuncOp function) {
    if (failed(status) || function.isExternal() || !claimsComplete(function))
      return;
    // Bufferization may introduce implicit content-version IDs without a
    // descriptor or producer annotation. Do not invent those missing facts;
    // check the full reverse relation wherever users are explicitly present.
    DenseMap<int64_t, DenseSet<int64_t>> consumers;
    function.walk([&](Operation *operation) {
      auto task = operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
      if (!task)
        return;
      for (int64_t value :
           getI64Array(operation, "bosc_ame.compute_input_values"))
        consumers[value].insert(task.getInt());
    });
    auto verifyUsers = [&](Operation *anchor, int64_t value,
                           DenseI64ArrayAttr users) -> LogicalResult {
      DenseSet<int64_t> declared;
      for (int64_t user : users.asArrayRef()) {
        auto task = tasks.find(user);
        if (task == tasks.end() ||
            task->second->getParentOfType<func::FuncOp>() != function)
          return anchor->emitError("unresolved BOSCAME logical value user ")
                 << user << " for value " << value;
        if (!llvm::is_contained(
                getI64Array(task->second, "bosc_ame.compute_input_values"),
                value))
          return anchor->emitError("BOSCAME logical value user ")
                 << user << " does not read value " << value;
        declared.insert(user);
      }
      const DenseSet<int64_t> &actual = consumers[value];
      if (declared.size() != actual.size() ||
          llvm::any_of(actual,
                       [&](int64_t user) { return !declared.contains(user); }))
        return anchor->emitError(
                   "BOSCAME logical value users mismatch for value ")
               << value;
      return success();
    };
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      Attribute idAttribute =
          function.getArgAttr(index, "bosc_ame.logical_value_id");
      Attribute usersAttribute =
          function.getArgAttr(index, "bosc_ame.logical_value_users");
      if (!idAttribute && !usersAttribute)
        continue;
      auto id = dyn_cast_or_null<IntegerAttr>(idAttribute);
      if (!id || id.getInt() < 0) {
        status = function.emitError(
            "invalid BOSCAME function argument logical value ID");
        return;
      }
      if (!usersAttribute)
        continue;
      auto users = dyn_cast<DenseI64ArrayAttr>(usersAttribute);
      if (!users) {
        status = function.emitError(
            "invalid BOSCAME function argument logical value users");
        return;
      }
      if (failed(status = verifyUsers(function, id.getInt(), users)))
        return;
    }
    function.walk([&](Operation *operation) {
      if (failed(status))
        return;
      Attribute idsAttribute = operation->getAttr("bosc_ame.logical_value_ids");
      Attribute usersAttribute =
          operation->getAttr("bosc_ame.logical_value_users");
      if (!idsAttribute && !usersAttribute)
        return;
      auto ids = dyn_cast_or_null<DenseI64ArrayAttr>(idsAttribute);
      if (idsAttribute &&
          (!ids || llvm::any_of(ids.asArrayRef(),
                                [](int64_t value) { return value < 0; }))) {
        status = operation->emitError("invalid BOSCAME logical value IDs");
        return;
      }
      if (!usersAttribute)
        return;
      auto users = dyn_cast<ArrayAttr>(usersAttribute);
      if (!ids || !users || static_cast<size_t>(ids.size()) != users.size()) {
        status = operation->emitError(
            "BOSCAME logical_value_users must match logical_value_ids");
        return;
      }
      for (auto [value, userAttribute] : llvm::zip(ids.asArrayRef(), users)) {
        auto userIds = dyn_cast<DenseI64ArrayAttr>(userAttribute);
        if (!userIds) {
          status = operation->emitError(
              "BOSCAME logical_value_users entry must be an i64 array");
          return;
        }
        if (failed(status = verifyUsers(operation, value, userIds)))
          return;
      }
    });
  });
  return status;
}
} // namespace buddy::boscame

static bool hasUnresolvedRootMemoryEffect(Operation *operation) {
  if (hasUnknownEffects(operation))
    return true;
  auto effects = getEffectsRecursively(operation);
  if (!effects)
    return true;
  return llvm::any_of(
      *effects, [](const MemoryEffects::EffectInstance &effect) {
        if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()))
          return false;
        Value value = effect.getValue();
        return !value || !isa<MemRefType, TensorType>(value.getType());
      });
}

/// A region-branch result that mentions the tracked root but cannot be
/// resolved may let an alias escape the nested effect scan.
static bool hasNestedControlFlowAlias(Operation *operation, Value root) {
  auto branchMentionsRoot = [&](Region &region) {
    auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
    if (!yield)
      return false;
    return llvm::any_of(yield.getOperands(), [&](Value value) {
      return isa<MemRefType>(value.getType()) && getStorageRoot(value) == root;
    });
  };
  if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
    for (Value result : ifOp.getResults()) {
      if (!isa<MemRefType>(result.getType()) || getStorageRoot(result) == root)
        continue;
      if (branchMentionsRoot(ifOp.getThenRegion()) ||
          branchMentionsRoot(ifOp.getElseRegion()))
        return true;
    }
    return false;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      return false;
    for (auto [index, result] : llvm::enumerate(forOp.getResults())) {
      if (!isa<MemRefType>(result.getType()) || getStorageRoot(result) == root)
        continue;
      Value init = forOp.getInitArgs()[index];
      Value iterArg = forOp.getRegionIterArgs()[index];
      Value yielded = yield.getOperand(index);
      if (getStorageRoot(init) == root || getStorageRoot(iterArg) == root ||
          getStorageRoot(yielded) == root)
        return true;
    }
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
    auto conditionArgs = whileOp.getConditionOp().getArgs();
    auto yieldedValues = whileOp.getYieldOp().getOperands();
    auto beforeArgs = whileOp.getBeforeArguments();
    auto afterArgs = whileOp.getAfterArguments();
    for (auto [index, result] : llvm::enumerate(whileOp.getResults())) {
      if (!isa<MemRefType>(result.getType()) || getStorageRoot(result) == root)
        continue;
      if (index >= whileOp.getInits().size() || index >= beforeArgs.size() ||
          index >= conditionArgs.size() || index >= afterArgs.size() ||
          index >= yieldedValues.size())
        return true;
      if (getStorageRoot(whileOp.getInits()[index]) == root ||
          getStorageRoot(beforeArgs[index]) == root ||
          getStorageRoot(conditionArgs[index]) == root ||
          getStorageRoot(afterArgs[index]) == root ||
          getStorageRoot(yieldedValues[index]) == root)
        return true;
    }
  }
  if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(operation)) {
    for (Value result : switchOp.getResults()) {
      if (!isa<MemRefType>(result.getType()) || getStorageRoot(result) == root)
        continue;
      auto regionMentionsRoot = [&](Region &region) {
        if (region.empty())
          return true;
        auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
        if (!yield)
          return true;
        return llvm::any_of(yield.getOperands(), [&](Value value) {
          return isa<MemRefType>(value.getType()) &&
                 getStorageRoot(value) == root;
        });
      };
      if (regionMentionsRoot(switchOp.getDefaultRegion()))
        return true;
      for (Region &region : switchOp.getCaseRegions())
        if (regionMentionsRoot(region))
          return true;
    }
  }
  return false;
}

static SmallVector<Value> getComputeTaskReadValues(Operation *operation) {
  SmallVector<Value> reads;
  if (auto linalg = dyn_cast<linalg::LinalgOp>(operation)) {
    for (Value input : linalg.getDpsInputs())
      reads.push_back(input);
    for (OpOperand &output : linalg.getDpsInitsMutable())
      if (linalg.payloadUsesValueFromOperand(&output))
        reads.push_back(output.get());
  } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
    reads.push_back(copy.getSource());
  } else if (isa<memref::StoreOp, memref::LoadOp>(operation)) {
    reads.push_back(operation->getOperand(0));
  }
  llvm::erase_if(reads, [](Value read) {
    return !isa<MemRefType, TensorType>(read.getType());
  });
  return reads;
}

static void collectKernelComputeFacts(ModuleOp module, KernelFacts &plan) {
  // Rebuild after planner rewrites as well as before placement. The analysis
  // contains live IR handles, unlike the detached source-event snapshot.
  plan.linalg = analyzeLinalgComputationFacts(module);
  auto version =
      module->getAttrOfType<IntegerAttr>("bosc_ame.compute_graph_version");
  bool hasGraph = version && version.getInt() == 1;
  bool complete = hasGraph;
  DenseSet<int64_t> functionArgumentValues;

  module.walk([&](func::FuncOp function) {
    if (function.isExternal())
      return;
    auto graphComplete =
        function->getAttrOfType<BoolAttr>("bosc_ame.compute_graph_complete");
    if (!graphComplete || !graphComplete.getValue())
      complete = false;
    auto expectedNodeCount = function->getAttrOfType<IntegerAttr>(
        "bosc_ame.compute_graph_node_count");
    size_t actualNodeCount = 0;
    function.walk([&](Operation *operation) {
      actualNodeCount += operation->hasAttr("bosc_ame.compute_id");
    });
    if (!expectedNodeCount || expectedNodeCount.getInt() < 0 ||
        static_cast<uint64_t>(expectedNodeCount.getInt()) != actualNodeCount)
      complete = false;
    for (unsigned i = 0; i < function.getNumArguments(); ++i)
      if (auto valueId = function.getArgAttrOfType<IntegerAttr>(
              i, "bosc_ame.logical_value_id"))
        functionArgumentValues.insert(valueId.getInt());
  });

  DenseMap<int64_t, size_t> taskById;
  module.walk([&](Operation *operation) {
    auto id = operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
    if (!id)
      return;
    PlannedTask task;
    task.operation = operation;
    task.id = id.getInt();
    task.kind = operation->getAttrOfType<StringAttr>("bosc_ame.compute_kind");
    if (!task.kind)
      complete = false;
    if (auto stage =
            operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_stage"))
      task.stage = stage.getInt();
    else
      complete = false;
    task.predecessors = getI64Array(operation, "bosc_ame.compute_predecessors");
    task.inputValues = getI64Array(operation, "bosc_ame.compute_input_values");
    task.outputValues =
        getI64Array(operation, "bosc_ame.compute_output_values");
    task.varyingProgramAxes =
        getI64Array(operation, "bosc_ame.varying_program_axes");
    if (auto barrier = operation->getAttrOfType<BoolAttr>(
            "bosc_ame.compute_barrier_after"))
      task.barrierAfter = barrier.getValue();
    else
      complete = false;
    if (auto rematerializable =
            operation->getAttrOfType<BoolAttr>("bosc_ame.rematerializable"))
      task.rematerializable = rematerializable.getValue();
    else
      complete = false;

    auto inserted = taskById.try_emplace(task.id, plan.tasks.size());
    if (!inserted.second)
      complete = false;
    plan.tasks.push_back(std::move(task));
  });

  DenseMap<int64_t, size_t> valueById;
  auto getValuePlan = [&](int64_t id) -> LogicalValuePlan & {
    auto found = valueById.find(id);
    if (found != valueById.end())
      return plan.logicalValues[found->second];
    size_t index = plan.logicalValues.size();
    valueById.try_emplace(id, index);
    LogicalValuePlan value;
    value.id = id;
    plan.logicalValues.push_back(std::move(value));
    return plan.logicalValues.back();
  };

  for (PlannedTask &task : plan.tasks) {
    for (int64_t predecessor : task.predecessors) {
      auto producer = taskById.find(predecessor);
      bool resolved = producer != taskById.end() &&
                      plan.tasks[producer->second]
                              .operation->getParentOfType<func::FuncOp>() ==
                          task.operation->getParentOfType<func::FuncOp>();
      plan.dependencies.push_back({predecessor, task.id, resolved});
      if (!resolved || predecessor >= task.id)
        complete = false;
    }
    for (int64_t valueId : task.outputValues) {
      if (valueId < 0)
        continue;
      LogicalValuePlan &value = getValuePlan(valueId);
      if (value.producerTask && *value.producerTask != task.id)
        complete = false;
      value.producerTask = task.id;
    }
    for (int64_t valueId : task.inputValues) {
      if (valueId < 0)
        continue;
      LogicalValuePlan &value = getValuePlan(valueId);
      value.consumerTasks.push_back(task.id);
    }
  }

  auto getLogicalStorageRoot = [&](Value value) {
    SmallPtrSet<Operation *, 8> visited;
    while (Operation *def = value.getDefiningOp()) {
      if (!visited.insert(def).second)
        break;
      StringRef name = def->getName().getStringRef();
      if (isStorageViewOperation(def) || name == "bufferization.to_tensor" ||
          name == "bufferization.to_buffer") {
        if (def->getNumOperands() != 1)
          break;
        value = def->getOperand(0);
        continue;
      }
      break;
    }
    return getStorageRoot(value);
  };
  auto getTaskWriteRoots = [&](Operation *operation) {
    SmallVector<Value> writes;
    if (auto linalg = dyn_cast<linalg::LinalgOp>(operation))
      for (Value output : linalg.getDpsInits())
        writes.push_back(output);
    else if (auto copy = dyn_cast<memref::CopyOp>(operation))
      writes.push_back(copy.getTarget());
    return writes;
  };

  // Bufferization can introduce a fresh logical ID for a view or a
  // to_tensor/to_buffer bridge even though the underlying storage is still
  // produced by a known task. Recover only exact-root, same-block writers;
  // ambiguous aliases stay incomplete.
  for (LogicalValuePlan &value : plan.logicalValues) {
    if (value.producerTask || value.consumerTasks.empty())
      continue;
    std::optional<int64_t> commonWriter;
    bool commonIsArgument = false;
    bool hasCommonSource = false;
    bool unambiguous = true;
    for (int64_t consumerId : value.consumerTasks) {
      auto consumerIndex = taskById.find(consumerId);
      if (consumerIndex == taskById.end()) {
        unambiguous = false;
        break;
      }
      const PlannedTask &consumerTask = plan.tasks[consumerIndex->second];
      SmallVector<Value> reads =
          getComputeTaskReadValues(consumerTask.operation);
      bool foundSlot = false;
      for (auto [index, inputId] : llvm::enumerate(consumerTask.inputValues)) {
        if (inputId != value.id)
          continue;
        if (index >= reads.size()) {
          unambiguous = false;
          break;
        }
        foundSlot = true;
        Value cursor = reads[index];
        SmallPtrSet<Operation *, 8> visited;
        std::optional<int64_t> sourceTask;
        while (Operation *def = cursor.getDefiningOp()) {
          if (!visited.insert(def).second)
            break;
          if (auto id =
                  def->getAttrOfType<IntegerAttr>("bosc_ame.compute_id")) {
            sourceTask = id.getInt();
            break;
          }
          StringRef name = def->getName().getStringRef();
          if (isStorageViewOperation(def) ||
              name == "bufferization.to_tensor" ||
              name == "bufferization.to_buffer") {
            if (def->getNumOperands() != 1)
              break;
            cursor = def->getOperand(0);
            continue;
          }
          break;
        }

        bool sourceIsArgument = false;
        if (!sourceTask) {
          Value root = getLogicalStorageRoot(reads[index]);
          auto rootArgument = dyn_cast<BlockArgument>(root);
          sourceIsArgument =
              rootArgument && rootArgument.getOwner()->isEntryBlock() &&
              isa<func::FuncOp>(rootArgument.getOwner()->getParentOp());
          if (!sourceIsArgument) {
            Operation *writer = nullptr;
            for (const PlannedTask &candidate : plan.tasks) {
              Operation *operation = candidate.operation;
              if (operation == consumerTask.operation ||
                  operation->getParentOfType<func::FuncOp>() !=
                      consumerTask.operation->getParentOfType<func::FuncOp>() ||
                  operation->getBlock() != consumerTask.operation->getBlock() ||
                  !operation->isBeforeInBlock(consumerTask.operation))
                continue;
              bool writesRoot = llvm::any_of(
                  getTaskWriteRoots(operation), [&](Value written) {
                    return getStorageRoot(written) == root;
                  });
              if (writesRoot && (!writer || writer->isBeforeInBlock(operation)))
                writer = operation;
            }
            if (writer)
              sourceTask =
                  writer->getAttrOfType<IntegerAttr>("bosc_ame.compute_id")
                      .getInt();
          }
        }
        if (!sourceTask && !sourceIsArgument) {
          unambiguous = false;
          break;
        }
        if (hasCommonSource &&
            (commonIsArgument != sourceIsArgument ||
             (!sourceIsArgument && commonWriter != sourceTask))) {
          unambiguous = false;
          break;
        }
        hasCommonSource = true;
        commonIsArgument = sourceIsArgument;
        commonWriter = sourceTask;
      }
      if (!foundSlot) {
        unambiguous = false;
        break;
      }
    }
    if (unambiguous && hasCommonSource && commonIsArgument)
      value.producerIsFunctionArgument = true;
    else if (unambiguous && hasCommonSource && commonWriter)
      value.producerTask = *commonWriter;
  }

  for (LogicalValuePlan &value : plan.logicalValues) {
    llvm::sort(value.consumerTasks);
    value.consumerTasks.erase(
        std::unique(value.consumerTasks.begin(), value.consumerTasks.end()),
        value.consumerTasks.end());
    value.producerIsFunctionArgument =
        value.producerIsFunctionArgument ||
        (!value.producerTask && functionArgumentValues.contains(value.id));
    value.lifetimeComplete =
        value.producerTask.has_value() || value.producerIsFunctionArgument;
    if (!value.lifetimeComplete)
      complete = false;
  }
  llvm::sort(plan.logicalValues,
             [](const LogicalValuePlan &lhs, const LogicalValuePlan &rhs) {
               return lhs.id < rhs.id;
             });

  if (!hasGraph)
    plan.computeGraphStatus = ComputeGraphStatus::Unavailable;
  else
    plan.computeGraphStatus = complete ? ComputeGraphStatus::Complete
                                       : ComputeGraphStatus::Incomplete;
}

static void collectDataflowFacts(ModuleOp module, DataflowPlan &plan) {
  collectKernelComputeFacts(module, plan);

  auto addExact = [&](uint64_t amount, uint64_t &total) {
    if (amount > std::numeric_limits<uint64_t>::max() - total)
      plan.resources.transferTotalsExact = false;
    else
      total += amount;
  };
  DenseMap<Value, int64_t> storageRootIds;
  for (auto [index, transfer] : llvm::enumerate(plan.transfers)) {
    addExact(1, plan.resources.plannedTransfers);
    if (transfer.guardedPrefix)
      plan.resources.transferTotalsExact = false;
    BufferRequirement buffer;
    buffer.transferIndex = index;
    buffer.buffer = transfer.buffer;
    buffer.storageRoot = getStorageRoot(transfer.ddrRoot);
    buffer.viewChain = transfer.viewChain;
    auto rootId =
        storageRootIds.try_emplace(buffer.storageRoot, storageRootIds.size());
    buffer.storageRootId = rootId.first->second;
    buffer.shape = transfer.shape;
    buffer.strides = transfer.strides;
    buffer.effectiveByteOffsetBytes = transfer.effectiveByteOffsetBytes;
    buffer.capacityBytes = transfer.spmCapacity;
    if (transfer.firstConsumer) {
      if (auto id = transfer.firstConsumer->getAttrOfType<IntegerAttr>(
              "bosc_ame.compute_id"))
        buffer.spmFirstTask = id.getInt();
    }
    if (transfer.lastConsumer) {
      if (auto id = transfer.lastConsumer->getAttrOfType<IntegerAttr>(
              "bosc_ame.compute_id"))
        buffer.spmLastTask = id.getInt();
    }
    buffer.operationLocalLifetime =
        transfer.firstConsumer &&
        transfer.firstConsumer == transfer.lastConsumer;
    buffer.crossTaskResident = transfer.crossTaskResident;
    buffer.physicalLifetimeKnown =
        plan.computeGraphStatus == ComputeGraphStatus::Complete;
    func::FuncOp trackedFunction =
        transfer.firstConsumer
            ? transfer.firstConsumer->getParentOfType<func::FuncOp>()
            : func::FuncOp();
    module.walk([&](Operation *operation) {
      // Function bodies are visited independently. The function symbol does
      // not execute at this point and must not make every local buffer unknown.
      if (isa<func::FuncOp>(operation))
        return;
      if (trackedFunction &&
          operation->getParentOfType<func::FuncOp>() != trackedFunction)
        return;
      if (isStorageViewOperation(operation) ||
          isa<memref::DeallocOp>(operation))
        return;
      Value storageRoot = getStorageRoot(transfer.ddrRoot);
      if (hasNestedControlFlowAlias(operation, storageRoot)) {
        buffer.physicalLifetimeKnown = false;
        return;
      }
      if (isa<RegionBranchOpInterface>(operation))
        return;
      bool touchesRoot =
          llvm::any_of(operation->getOperands(), [&](Value value) {
            return isa<MemRefType>(value.getType()) &&
                   getStorageRoot(value) == storageRoot;
          });
      if (isMemoryEffectFree(operation))
        return;
      if (hasUnresolvedRootMemoryEffect(operation)) {
        buffer.physicalLifetimeKnown = false;
        return;
      }
      if (!touchesRoot)
        return;
      auto taskId =
          operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
      if (!taskId) {
        buffer.physicalLifetimeKnown = false;
        return;
      }
      buffer.physicalUseTasks.push_back(taskId.getInt());
    });
    llvm::sort(buffer.physicalUseTasks);
    buffer.physicalUseTasks.erase(std::unique(buffer.physicalUseTasks.begin(),
                                              buffer.physicalUseTasks.end()),
                                  buffer.physicalUseTasks.end());
    if (buffer.physicalLifetimeKnown && !buffer.physicalUseTasks.empty()) {
      buffer.physicalFirstTask = buffer.physicalUseTasks.front();
      buffer.physicalLastTask = buffer.physicalUseTasks.back();
    }
    plan.buffers.push_back(std::move(buffer));

    for (const DmaPacket &packet : transfer.packets) {
      if (transfer.read) {
        addExact(1, plan.resources.plannedDmaPackets);
        addExact(packet.bytes, plan.resources.plannedReadBytes);
      }
      if (transfer.write) {
        addExact(1, plan.resources.plannedDmaPackets);
        addExact(packet.bytes, plan.resources.plannedWriteBytes);
      }
    }
  }
}

static bool isSupportedAmeMatmul(linalg::MatmulOp op,
                                 const AmeCapability &ame) {
  if (!op->hasAttr(::buddy::boscame::lowering::kLowerToAme))
    return true; // Unmarked matmuls stay on ordinary RVV/memory lowering.
  if (op.getNumResults() || op.getInputs().size() != 2 ||
      op.getOutputs().size() != 1 || !ame.a.supportsSpm || !ame.b.supportsSpm ||
      !ame.accumulator.supportsSpm)
    return false;
  auto a = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto b = dyn_cast<MemRefType>(op.getInputs()[1].getType());
  auto c = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  auto aElement = a ? dyn_cast<IntegerType>(a.getElementType()) : IntegerType();
  auto bElement = b ? dyn_cast<IntegerType>(b.getElementType()) : IntegerType();
  auto cElement = c ? dyn_cast<IntegerType>(c.getElementType()) : IntegerType();
  return a && b && c && a.getRank() == 2 && b.getRank() == 2 &&
         c.getRank() == 2 && aElement && bElement && cElement &&
         aElement.getWidth() == ame.a.elementBits &&
         bElement.getWidth() == ame.b.elementBits &&
         cElement.getWidth() == ame.accumulator.elementBits &&
         op.getCast() == (ame.signedIntegerMultiply
                              ? linalg::TypeFn::cast_signed
                              : linalg::TypeFn::cast_unsigned);
}

/// Return whether a buffer-form Linalg operation has the rectangular access
/// contract required by the SPM vector island.  The planner deliberately
/// leaves other forms in ordinary DDR so a later generic-to-loops lowering can
/// implement them without staging bytes that an affine map does not cover.
/// Projected permutations include identity, transpose and broadcasts; affine
/// scaling, constants and repeated dimensions are rejected because the full
/// memref shape would over-approximate a non-rectangular valid region.
static bool isRectangularMap(AffineMap map) {
  return map.isProjectedPermutation();
}

static bool isPredicateOnlyInput(linalg::LinalgOp op, unsigned index) {
  auto generic = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!generic || index >= generic.getNumDpsInputs() ||
      index >= generic.getBody()->getNumArguments())
    return false;
  auto type = dyn_cast<MemRefType>(generic->getOperand(index).getType());
  if (!type || !type.getElementType().isInteger(1))
    return false;
  BlockArgument argument = generic.getBody()->getArgument(index);
  return !argument.use_empty() &&
         llvm::all_of(argument.getUses(), [](OpOperand &use) {
           auto select = dyn_cast<arith::SelectOp>(use.getOwner());
           return select && use.getOperandNumber() == 0;
         });
}

static bool isPredicateOnlyOutput(linalg::LinalgOp op, unsigned index) {
  auto generic = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!generic || index < generic.getNumDpsInputs() ||
      index >= generic->getNumOperands())
    return false;
  auto type = dyn_cast<MemRefType>(generic->getOperand(index).getType());
  if (!type || !type.getElementType().isInteger(1))
    return false;
  auto memory = generic->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  auto kind = memory ? memory.getAs<StringAttr>("kind") : StringAttr();
  auto destination =
      memory ? memory.getAs<StringAttr>("destination") : StringAttr();
  return !(kind && destination && kind.getValue() == "store" &&
           destination.getValue() == "external_ddr");
}

static bool hasEffectFreePayload(linalg::LinalgOp op) {
  bool effectFree = true;
  op->walk([&](Operation *nested) {
    if (nested != op.getOperation() && !isMemoryEffectFree(nested))
      effectFree = false;
  });
  return effectFree;
}

static bool isStageableBufferLinalg(linalg::LinalgOp op) {
  if (op->hasAttr(kPrefixRangeAttr) || op->hasAttr(kPrefixMaskAuxAttr) ||
      !hasEffectFreePayload(op))
    return false;

  if (auto reduce = dyn_cast<linalg::ReduceOp>(op.getOperation())) {
    // lowerSpmComputations currently has an ordered f32 rank-1 reduction
    // implementation only.  Other reductions remain ordinary arithmetic.
    if (reduce.getInputs().empty() || reduce.getInits().empty())
      return false;
    auto input = dyn_cast<MemRefType>(reduce.getInputs().front().getType());
    auto output = dyn_cast<MemRefType>(reduce.getInits().front().getType());
    return input && output && input.getRank() == 1 &&
           input.getElementType().isF32() && output.getRank() == 0;
  }

  if (auto generic = dyn_cast<linalg::GenericOp>(op.getOperation())) {
    auto maps = generic.getIndexingMapsArray();
    if (!llvm::all_of(maps, isRectangularMap))
      return false;
    // Predicate producers are ordinary RVV computation, not DMA-backed SPM
    // values. Their i1 result may feed a later generic's select condition.
    if (llvm::any_of(generic.getOutputs(), [](Value output) {
          auto type = dyn_cast<MemRefType>(output.getType());
          return type && type.getElementType().isInteger(1);
        }))
      return false;
    // Vector computation lowering supports f32 and native integer elements.
    // Keep f16/bf16/f64 on the ordinary fallback.  Integer widths remain in
    // the staged path so the transfer analyzer can reject widths that the
    // typed DMA contract cannot represent (for example i1/i4), rather than
    // silently changing the requested BOSCAME lowering into a CPU fallback.
    for (Value operand : generic.getOperands()) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || type.getRank() == 0)
        continue;
      Type element = type.getElementType();
      if (!element.isF32() && !element.isInteger())
        return false;
    }
  }
  return true;
}

struct ResidentCandidate {
  Value value;
  /// A non-null producer means the value is written by the first operation
  /// and then read by the listed consumers.  The producer's staged output is
  /// kept live until the final consumer, with its DDR write emitted there.
  Operation *producer = nullptr;
  SmallVector<Operation *, 4> consumers;
};

struct BufferAliasContext {
  DenseSet<Operation *> abiEntryFunctions;
};

static bool isResidencyWindowSafe(const ResidentCandidate &candidate,
                                  const KernelFacts &facts,
                                  const BufferAliasContext &aliases);

static bool isFunctionEntryArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner()->isEntryBlock() &&
         isa<func::FuncOp>(argument.getOwner()->getParentOp());
}

/// Find the deliberately narrow residency case that the existing emitter can
/// materialize safely.  The value is either a function argument read by
/// multiple staged Linalg operations, or one staged Linalg output read by one
/// or more later operations. Root-touching operations between the first and
/// last use invalidate the candidate. Cross-root effects are checked against
/// complete access and alias facts below, so proven-disjoint operations need
/// not block residency. Same-root subranges remain conservatively unsupported.
static DenseMap<Value, ResidentCandidate>
findResidentCandidates(ModuleOp module, const KernelFacts &facts,
                       const BufferAliasContext &aliases) {
  DenseMap<Value, SmallVector<Operation *, 4>> uses;
  if (facts.computeGraphStatus != ComputeGraphStatus::Complete ||
      !facts.memoryDependenciesComplete || facts.memoryDependenciesBounded)
    return {};

  module.walk([&](linalg::LinalgOp op) {
    if (!op.hasPureBufferSemantics() || !isStageableBufferLinalg(op))
      return;
    for (Value value : op.getDpsInputs()) {
      auto type = dyn_cast<MemRefType>(value.getType());
      if (!type || type.getRank() == 0 || spmdma::getSpmBank(type))
        continue;
      uses[value].push_back(op.getOperation());
    }
  });

  DenseMap<Value, ResidentCandidate> candidates;
  for (auto &entry : uses) {
    Value value = entry.first;
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() == 0 || spmdma::getSpmBank(type))
      continue;
    SmallVector<Operation *, 4> consumers;
    for (Operation *consumer : entry.second)
      if (consumers.empty() || consumers.back() != consumer)
        consumers.push_back(consumer);

    Operation *producer = nullptr;
    unsigned producerOutputUses = 0;

    // Every use of the exact SSA value must be one of the read-only DPS input
    // uses collected above, or the one output use that establishes a resident
    // value. A view or unknown use needs a separate alias proof.
    bool valid = llvm::all_of(value.getUses(), [&](OpOperand &use) {
      auto linalg = dyn_cast<linalg::LinalgOp>(use.getOwner());
      if (!linalg)
        return false;
      if (use.getOperandNumber() < linalg.getNumDpsInputs())
        return llvm::is_contained(consumers, use.getOwner());
      if (!isStageableBufferLinalg(linalg))
        return false;
      if (producer && producer != use.getOwner())
        return false;
      producer = use.getOwner();
      ++producerOutputUses;
      return true;
    });
    // A producer output is only safe to keep in SPM when it is a single
    // destination use.  In particular, reject an in-place producer that also
    // consumes this exact SSA value.
    if (!valid || producerOutputUses > 1 ||
        (producer && llvm::is_contained(consumers, producer)))
      continue;
    if ((!producer && consumers.size() < 2) || (producer && consumers.empty()))
      continue;

    Operation *first = producer ? producer : consumers.front();
    Operation *last = consumers.back();
    Block *block = first->getBlock();
    auto function = first->getParentOfType<func::FuncOp>();
    if (!block || !function || last->getBlock() != block ||
        last->getParentOfType<func::FuncOp>() != function ||
        !first->isBeforeInBlock(last))
      continue;
    if (producer && !producer->isBeforeInBlock(consumers.front()))
      continue;

    Value root = getStorageRoot(value);
    auto touchesRoot = [&](Operation *operation) {
      if (isStorageViewOperation(operation) ||
          isa<memref::DeallocOp>(operation))
        return false;
      return llvm::any_of(operation->getOperands(), [&](Value operand) {
        return isa<MemRefType>(operand.getType()) &&
               getStorageRoot(operand) == root;
      });
    };

    // A consumer may read the root, but it may not write an output alias of
    // that root. This rejects in-place Linalg updates conservatively.
    for (Operation *consumer : consumers) {
      auto linalg = cast<linalg::LinalgOp>(consumer);
      for (OpOperand &operand : linalg->getOpOperands())
        if (operand.getOperandNumber() >= linalg.getNumDpsInputs() &&
            isa<MemRefType>(operand.get().getType()) &&
            getStorageRoot(operand.get()) == root)
          valid = false;
      auto barrier =
          consumer->getAttrOfType<BoolAttr>("bosc_ame.compute_barrier_after");
      auto varying = consumer->getAttrOfType<DenseI64ArrayAttr>(
          "bosc_ame.varying_program_axes");
      if (!barrier || barrier.getValue() || !varying ||
          !varying.asArrayRef().empty())
        valid = false;
    }
    if (producer) {
      auto linalg = cast<linalg::LinalgOp>(producer);
      // The resident output itself is the only root write permitted at the
      // beginning of the interval.  A producer input alias would require
      // preserving an old value and is outside this bounded optimization.
      for (OpOperand &operand : linalg->getOpOperands()) {
        bool output = operand.getOperandNumber() >= linalg.getNumDpsInputs();
        if (!output && isa<MemRefType>(operand.get().getType()) &&
            getStorageRoot(operand.get()) == root)
          valid = false;
        if (output && operand.get() != value &&
            isa<MemRefType>(operand.get().getType()) &&
            getStorageRoot(operand.get()) == root)
          valid = false;
      }
      auto barrier =
          producer->getAttrOfType<BoolAttr>("bosc_ame.compute_barrier_after");
      auto varying = producer->getAttrOfType<DenseI64ArrayAttr>(
          "bosc_ame.varying_program_axes");
      if (!barrier || barrier.getValue() || !varying ||
          !varying.asArrayRef().empty())
        valid = false;
    }
    if (!valid)
      continue;

    for (Operation *operation = first; operation;
         operation = operation == last ? nullptr : operation->getNextNode()) {
      bool knownConsumer = llvm::is_contained(consumers, operation);
      if (operation != first && operation != last && !knownConsumer &&
          touchesRoot(operation)) {
        valid = false;
        break;
      }
      if (operation != first && operation != last) {
        if (operation->hasAttr("bosc_ame.compute_barrier_after")) {
          auto barrier = operation->getAttrOfType<BoolAttr>(
              "bosc_ame.compute_barrier_after");
          if (!barrier || barrier.getValue()) {
            valid = false;
            break;
          }
        }
      }
    }
    if (valid &&
        isResidencyWindowSafe({value, producer, consumers}, facts, aliases))
      candidates.try_emplace(value,
                             ResidentCandidate{value, producer, consumers});
  }
  return candidates;
}

/// A frontend-preserved store event names an external DDR destination
/// explicitly.  Under the direct-ddr policy such an event must remain on the
/// ordinary memory path; staging its destination here would silently turn the
/// requested direct store back into an SPM/DMA round trip.  Keep the check
/// conservative until the metadata is present: legacy buffer-form Linalg with
/// only `kind = "store"` continues to use its established staging behavior.
static bool isDirectExternalStore(linalg::LinalgOp op,
                                  const LinalgToBOSCAMEConfig &config) {
  if (config.storePolicy != "direct-ddr")
    return false;
  auto memory = op->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  if (!memory)
    return false;
  auto kind = memory.getAs<StringAttr>("kind");
  auto destination = memory.getAs<StringAttr>("destination");
  return kind && destination && kind.getValue() == "store" &&
         destination.getValue() == "external_ddr";
}

static bool hasLogicalPlacement(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  auto placement = module
                       ? module->getAttrOfType<StringAttr>("bosc_ame.placement")
                       : StringAttr();
  return placement && placement.getValue() == "logical";
}

/// Direct vector stores currently lower a complete rectangular tile.  A
/// frontend mask or a shape mismatch would make the generated stores overwrite
/// elements outside the valid region, so reject that form in direct mode
/// instead of silently staging it through the compatibility DMA path.
static LogicalResult verifyDirectStoreMetadata(Operation *op,
                                               Value destination) {
  auto memory = op->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  if (!memory)
    return success();
  auto masked = memory.getAs<BoolAttr>("masked");
  auto logical = memory.getAs<DenseI64ArrayAttr>("logical_shape");
  auto valid = memory.getAs<DenseI64ArrayAttr>("valid_shape");
  if (!masked || masked.getValue())
    return op->emitError("direct-ddr store does not support masked stores; "
                         "select store-policy=spm-dma");
  if (!logical || !valid || logical.asArrayRef() != valid.asArrayRef())
    return op->emitError("direct-ddr store requires equal static logical and "
                         "valid shapes");
  auto type = dyn_cast<MemRefType>(destination.getType());
  if (!type || (type.getRank() != 1 && type.getRank() != 2) ||
      type.getRank() != static_cast<int64_t>(logical.size()) ||
      type.getShape() != logical.asArrayRef() ||
      llvm::any_of(logical.asArrayRef(),
                   [](int64_t extent) { return extent <= 0; }))
    return op->emitError("direct-ddr store requires a positive static shape");
  // Direct vector stores advance one physical row at a time.  The target
  // instruction stream has no byte-granular prologue for a row, so every row
  // address must remain cache-line aligned just like a DMA packet.
  SmallVector<int64_t> strides;
  int64_t offset;
  int64_t elementBytes =
      getBOSCAMEElementBytes(type.getElementType()).value_or(0);
  if (!elementBytes || failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != static_cast<size_t>(type.getRank()) ||
      llvm::any_of(strides, [](int64_t stride) { return stride <= 0; }) ||
      strides.back() != 1)
    return op->emitError(
        "direct-ddr store requires a positive contiguous final axis");
  if (type.getRank() == 2) {
    int64_t rowStrideBytes;
    if (strides.front() < type.getShape().back() ||
        llvm::MulOverflow(strides.front(), elementBytes, rowStrideBytes) ||
        rowStrideBytes % spmdma::kDmaAlignmentBytes != 0)
      return op->emitError(
          "direct-ddr store requires 64-byte-aligned physical row strides");
  }
  if (failed(spmdma::verifyEndpointPointerAlignment(
          op, destination, "direct-DDR store", spmdma::kDmaAlignmentBytes)))
    return failure();
  return success();
}

/// Function block arguments are the external DDR pointers at the
/// buffer-form boundary.  An allocation or alloca is compiler-managed storage
/// and must retain the SPM/DMA path even when direct-ddr is the global policy.
static bool isExternalDdrBuffer(Value value) {
  if (auto type = dyn_cast<MemRefType>(value.getType());
      type && spmdma::getSpmBank(type))
    return false;
  while (Operation *def = value.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
      value = transpose.getIn();
      continue;
    }
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      value = view.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      value = expand.getSrc();
      continue;
    }
    break;
  }
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner()->isEntryBlock() &&
         isa<func::FuncOp>(argument.getOwner()->getParentOp()) &&
         !spmdma::getSpmBank(argument.getType());
}

/// Follow only control-flow edges that preserve the exact memref SSA value.
/// Distinct branch values may share a root but can describe different views.
static Value getExactControlFlowPassthrough(Value value,
                                            DenseSet<Value> &visited,
                                            Operation *assumedWhile = nullptr) {
  if (!visited.insert(value).second)
    return {};
  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    DenseSet<Value> trueVisited(visited);
    DenseSet<Value> falseVisited(visited);
    Value trueValue = getExactControlFlowPassthrough(select.getTrueValue(),
                                                     trueVisited, assumedWhile);
    Value falseValue = getExactControlFlowPassthrough(
        select.getFalseValue(), falseVisited, assumedWhile);
    return trueValue && trueValue == falseValue ? trueValue : Value();
  }
  auto ifOp = value.getDefiningOp<scf::IfOp>();
  if (ifOp) {
    unsigned resultNumber = cast<OpResult>(value).getResultNumber();
    Value common;
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
      if (!yield || resultNumber >= yield.getNumOperands())
        return {};
      DenseSet<Value> branchVisited(visited);
      Value branchValue = getExactControlFlowPassthrough(
          yield.getOperand(resultNumber), branchVisited, assumedWhile);
      if (!branchValue || (common && common != branchValue))
        return {};
      common = branchValue;
    }
    return common;
  }
  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    unsigned resultNumber = cast<OpResult>(value).getResultNumber();
    if (resultNumber >= forOp.getInitArgs().size() ||
        resultNumber >= forOp.getRegionIterArgs().size())
      return {};
    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield || resultNumber >= yield.getNumOperands())
      return {};
    Value init = forOp.getInitArgs()[resultNumber];
    Value iterArg = forOp.getRegionIterArgs()[resultNumber];
    DenseSet<Value> yieldVisited(visited);
    Value yielded = getExactControlFlowPassthrough(
        yield.getOperand(resultNumber), yieldVisited, assumedWhile);
    if (!yielded || (yielded != iterArg && yielded != init))
      return {};
    return init;
  }
  if (auto switchOp = value.getDefiningOp<scf::IndexSwitchOp>()) {
    unsigned resultNumber = cast<OpResult>(value).getResultNumber();
    Value common;
    auto visitRegion = [&](Region &region) -> Value {
      if (region.empty())
        return {};
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (!yield || resultNumber >= yield.getNumOperands())
        return {};
      DenseSet<Value> branchVisited(visited);
      return getExactControlFlowPassthrough(yield.getOperand(resultNumber),
                                            branchVisited, assumedWhile);
    };
    common = visitRegion(switchOp.getDefaultRegion());
    if (!common)
      return {};
    for (Region &region : switchOp.getCaseRegions()) {
      Value branchValue = visitRegion(region);
      if (!branchValue || branchValue != common)
        return {};
    }
    return common;
  }
  if (auto whileOp = value.getDefiningOp<scf::WhileOp>()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return {};
    unsigned resultNumber = result.getResultNumber();
    auto conditionArgs = whileOp.getConditionOp().getArgs();
    auto yieldedValues = whileOp.getYieldOp().getOperands();
    auto beforeArgs = whileOp.getBeforeArguments();
    auto afterArgs = whileOp.getAfterArguments();
    if (resultNumber >= whileOp.getInits().size() ||
        resultNumber >= beforeArgs.size() ||
        resultNumber >= conditionArgs.size() ||
        resultNumber >= afterArgs.size() ||
        resultNumber >= yieldedValues.size())
      return {};

    // A while result is the value forwarded by the before-region condition
    // on the iteration that exits. Prove that value is the same exact
    // storage as the init and that every after-region backedge preserves it.
    DenseSet<Value> initVisited(visited);
    Value common = getExactControlFlowPassthrough(
        whileOp.getInits()[resultNumber], initVisited, assumedWhile);
    if (!common)
      return {};
    DenseSet<Value> conditionVisited(visited);
    Value conditionSource = getExactControlFlowPassthrough(
        conditionArgs[resultNumber], conditionVisited, whileOp);
    if (!conditionSource || conditionSource != common)
      return {};
    DenseSet<Value> yieldedVisited(visited);
    Value yieldedSource = getExactControlFlowPassthrough(
        yieldedValues[resultNumber], yieldedVisited, whileOp);
    if (!yieldedSource || yieldedSource != common)
      return {};
    return common;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto parent = argument.getOwner()->getParentOp();
    auto whileOp = dyn_cast<scf::WhileOp>(parent);
    if (!whileOp)
      return value;
    unsigned index = argument.getArgNumber();
    if (assumedWhile != whileOp) {
      if (index >= whileOp.getNumResults())
        return {};
      DenseSet<Value> resultVisited;
      Value source = getExactControlFlowPassthrough(
          whileOp.getResult(index), resultVisited, assumedWhile);
      if (!source)
        return {};
    }
    if (argument.getOwner() == whileOp.getBeforeBody()) {
      if (index >= whileOp.getInits().size())
        return {};
      return getExactControlFlowPassthrough(whileOp.getInits()[index], visited,
                                            assumedWhile);
    }
    if (argument.getOwner() == whileOp.getAfterBody()) {
      auto conditionArgs = whileOp.getConditionOp().getArgs();
      if (index >= conditionArgs.size())
        return {};
      return getExactControlFlowPassthrough(conditionArgs[index], visited,
                                            assumedWhile);
    }
  }
  return value;
}

static Value getForIterArgInit(BlockArgument argument) {
  auto forOp = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
  if (!forOp || argument.getOwner() != forOp.getBody())
    return {};
  auto iterArgs = forOp.getRegionIterArgs();
  for (auto [index, iterArg] : llvm::enumerate(iterArgs)) {
    if (iterArg != argument || index >= forOp.getInitArgs().size())
      continue;
    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield || index >= yield.getNumOperands())
      return {};
    Value init = forOp.getInitArgs()[index];
    DenseSet<Value> visited;
    Value yielded =
        getExactControlFlowPassthrough(yield.getOperand(index), visited);
    return yielded == argument || yielded == init ? init : Value();
  }
  return {};
}

/// Strip storage-preserving views and follow exact memref passthrough through
/// scf.if, scf.for, and exact scf.while forwarding. Unknown or changed
/// control-flow values remain opaque.
static Value getStorageRoot(Value value) {
  DenseSet<Value> active;
  std::function<Value(Value)> resolve = [&](Value current) -> Value {
    if (!current || !active.insert(current).second)
      return {};
    auto finish = [&](Value root) {
      active.erase(current);
      return root;
    };
    while (Operation *def = current.getDefiningOp()) {
      if (auto cast = dyn_cast<memref::CastOp>(def)) {
        current = cast.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
        current = cast.getSource();
        continue;
      }
      if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
        current = subview.getSource();
        continue;
      }
      if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
        current = transpose.getIn();
        continue;
      }
      if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
        current = cast.getSource();
        continue;
      }
      if (auto view = dyn_cast<memref::ViewOp>(def)) {
        current = view.getSource();
        continue;
      }
      if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
        current = collapse.getSrc();
        continue;
      }
      if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
        current = expand.getSrc();
        continue;
      }
      if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
        current = reshape.getSource();
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(def)) {
        if (select.getTrueValue() == select.getFalseValue())
          return finish(resolve(select.getTrueValue()));
        Value trueRoot = resolve(select.getTrueValue());
        Value falseRoot = resolve(select.getFalseValue());
        if (!trueRoot || trueRoot != falseRoot)
          break;
        return finish(trueRoot);
      }
      if (isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::IndexSwitchOp>(def)) {
        DenseSet<Value> visited;
        Value source = getExactControlFlowPassthrough(current, visited);
        if (!source)
          break;
        return finish(resolve(source));
      }
      break;
    }
    if (auto argument = dyn_cast<BlockArgument>(current))
      if (Value init = getForIterArgInit(argument))
        return finish(resolve(init));
    if (isa<BlockArgument>(current)) {
      DenseSet<Value> visited;
      Value source = getExactControlFlowPassthrough(current, visited);
      if (source && source != current)
        return finish(resolve(source));
    }
    return finish(canonicalizeBOSCAMEGlobalRoot(current));
  };
  Value root = resolve(value);
  return root ? root : canonicalizeBOSCAMEGlobalRoot(value);
}

static bool isStorageViewOperation(Operation *operation) {
  return isa<memref::CastOp, memref::ReinterpretCastOp, memref::SubViewOp,
             memref::TransposeOp, memref::MemorySpaceCastOp, memref::ViewOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(operation);
}

static std::optional<std::pair<int64_t, int64_t>>
getCheckedByteInterval(std::optional<int64_t> offsetBytes, int64_t spanBytes) {
  if (!offsetBytes || *offsetBytes < 0 || spanBytes < 0)
    return std::nullopt;
  int64_t endBytes;
  if (llvm::AddOverflow(*offsetBytes, spanBytes, endBytes))
    return std::nullopt;
  return std::pair<int64_t, int64_t>{*offsetBytes, endBytes};
}

static bool areByteIntervalsDisjoint(const std::pair<int64_t, int64_t> &lhs,
                                     const std::pair<int64_t, int64_t> &rhs) {
  return lhs.first == lhs.second || rhs.first == rhs.second ||
         lhs.second <= rhs.first || rhs.second <= lhs.first;
}

struct SymbolicByteInterval {
  struct DynamicOffsetTerm {
    Value value;
    int64_t coefficientBytes;
  };

  Value root;
  SmallVector<DynamicOffsetTerm> dynamicOffsets;
  int64_t startBytes = 0;
  int64_t endBytes = 0;
};

struct SymbolicByteRows {
  SymbolicByteInterval interval;
  int64_t rowCount = 0;
  int64_t rowStrideBytes = 0;
  int64_t rowBytes = 0;
};

/// Summarize a statically shaped positive-stride view's bounding interval
/// relative to its root descriptor. The root's own runtime descriptor offset
/// is shared by all views and cancels in pairwise comparisons; unsupported
/// view semantics remain unknown.
static std::optional<SymbolicByteInterval>
getSymbolicByteInterval(Value value) {
  // A select may preserve storage identity while choosing different view
  // descriptors.  Do not treat the select result as an offset-zero view;
  // retain interval proofs only when both arms forward the exact same SSA
  // value.
  if (value.getDefiningOp<arith::SelectOp>()) {
    DenseSet<Value> visited;
    Value source = getExactControlFlowPassthrough(value, visited);
    if (!source)
      return std::nullopt;
    value = source;
  }

  // A control-flow result carries no descriptor metadata of its own. Trace
  // only an exact passthrough before walking view offsets; divergent or
  // changed paths must remain unknown rather than being treated as offset 0.
  if (value.getDefiningOp<scf::IfOp>() || value.getDefiningOp<scf::ForOp>() ||
      value.getDefiningOp<scf::WhileOp>() ||
      value.getDefiningOp<scf::IndexSwitchOp>()) {
    DenseSet<Value> visited;
    Value source = getExactControlFlowPassthrough(value, visited);
    if (!source)
      return std::nullopt;
    value = source;
  }
  auto getPositiveStrides =
      [](MemRefType type) -> std::optional<SmallVector<int64_t>> {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!type || failed(type.getStridesAndOffset(strides, offset)) ||
        type.getRank() < 1 ||
        strides.size() != static_cast<size_t>(type.getRank()) ||
        llvm::any_of(strides, [](int64_t stride) { return stride <= 0; }))
      return std::nullopt;
    return strides;
  };
  auto getPositiveStrideStaticType = [](Value candidate) -> MemRefType {
    auto type = dyn_cast<MemRefType>(candidate.getType());
    if (!type || type.getRank() < 1 || !type.hasStaticShape() ||
        llvm::any_of(type.getShape(),
                     [](int64_t dimension) { return dimension <= 0; }))
      return {};
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(strides, offset)) ||
        strides.size() != static_cast<size_t>(type.getRank()) ||
        llvm::any_of(strides, [](int64_t stride) { return stride <= 0; }))
      return {};
    return type;
  };
  auto hasCompatibleReassociationLayout = [&](MemRefType sourceType,
                                              MemRefType resultType,
                                              ArrayRef<ReassociationIndices>
                                                  reassociation,
                                              bool collapsing) {
    auto sourceStrides = getPositiveStrides(sourceType);
    auto resultStrides = getPositiveStrides(resultType);
    int64_t sourceOffset, resultOffset;
    SmallVector<int64_t> ignoredStrides;
    if (!sourceType || !resultType || !sourceStrides || !resultStrides ||
        sourceType.getElementType() != resultType.getElementType() ||
        failed(sourceType.getStridesAndOffset(ignoredStrides, sourceOffset)) ||
        failed(resultType.getStridesAndOffset(ignoredStrides, resultOffset)) ||
        sourceOffset != resultOffset)
      return false;

    int64_t nextDimension = 0;
    if (collapsing) {
      if (reassociation.size() != static_cast<size_t>(resultType.getRank()))
        return false;
      for (auto [resultDimension, group] : llvm::enumerate(reassociation)) {
        if (group.empty())
          return false;
        int64_t groupSize = 1;
        for (int64_t dimension : group) {
          if (dimension != nextDimension || dimension >= sourceType.getRank() ||
              llvm::MulOverflow(groupSize, sourceType.getShape()[dimension],
                                groupSize))
            return false;
          ++nextDimension;
        }
        if (groupSize != resultType.getShape()[resultDimension])
          return false;

        int64_t strideDimension = group.back();
        for (size_t index = group.size() - 1;
             index > 0 && sourceType.getShape()[strideDimension] == 1; --index)
          strideDimension = group[index - 1];
        if ((*resultStrides)[resultDimension] !=
            (*sourceStrides)[strideDimension])
          return false;

        int64_t expectedStride = (*sourceStrides)[strideDimension];
        for (int64_t dimension = group.back(); dimension > group.front();
             --dimension) {
          if (llvm::MulOverflow(expectedStride,
                                sourceType.getShape()[dimension],
                                expectedStride))
            return false;
          if (sourceType.getShape()[dimension - 1] != 1 &&
              (*sourceStrides)[dimension - 1] != expectedStride)
            return false;
        }
      }
      return nextDimension == sourceType.getRank();
    }

    if (reassociation.size() != static_cast<size_t>(sourceType.getRank()))
      return false;
    for (auto [sourceDimension, group] : llvm::enumerate(reassociation)) {
      if (group.empty())
        return false;
      int64_t groupSize = 1;
      for (int64_t dimension : group) {
        if (dimension != nextDimension || dimension >= resultType.getRank() ||
            llvm::MulOverflow(groupSize, resultType.getShape()[dimension],
                              groupSize))
          return false;
        ++nextDimension;
      }
      if (groupSize != sourceType.getShape()[sourceDimension])
        return false;

      int64_t expectedStride = (*sourceStrides)[sourceDimension];
      for (size_t index = group.size(); index > 0; --index) {
        int64_t dimension = group[index - 1];
        if ((*resultStrides)[dimension] != expectedStride)
          return false;
        if (index > 1 &&
            llvm::MulOverflow(expectedStride, resultType.getShape()[dimension],
                              expectedStride))
          return false;
      }
    }
    return nextDimension == resultType.getRank();
  };

  MemRefType valueType = getPositiveStrideStaticType(value);
  if (!valueType)
    return std::nullopt;
  auto elementBytes = getBOSCAMEElementBytes(valueType.getElementType());
  if (!elementBytes)
    return std::nullopt;

  int64_t relativeOffsetElements = 0;
  SmallVector<SymbolicByteInterval::DynamicOffsetTerm> dynamicOffsets;
  auto accumulateDynamicOffset = [&](Value offset, int64_t coefficientBytes) {
    auto found = llvm::find_if(
        dynamicOffsets, [&](const auto &term) { return term.value == offset; });
    if (found == dynamicOffsets.end()) {
      dynamicOffsets.push_back({offset, coefficientBytes});
      return true;
    }
    int64_t combinedCoefficientBytes;
    if (llvm::AddOverflow(found->coefficientBytes, coefficientBytes,
                          combinedCoefficientBytes))
      return false;
    found->coefficientBytes = combinedCoefficientBytes;
    return true;
  };
  bool hasReinterpretBase = false;
  Value cursor = value;
  while (Operation *definition = cursor.getDefiningOp()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(definition)) {
      Value source = subview.getSource();
      MemRefType sourceType = getPositiveStrideStaticType(source);
      MemRefType resultType = getPositiveStrideStaticType(cursor);
      SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = subview.getMixedSizes();
      SmallVector<OpFoldResult> strides = subview.getMixedStrides();
      auto sourceStrides = getPositiveStrides(sourceType);
      auto resultStrides = getPositiveStrides(resultType);
      if (!sourceType || !resultType ||
          sourceType.getElementType() != valueType.getElementType() ||
          resultType.getElementType() != valueType.getElementType() ||
          sourceType.getRank() != resultType.getRank() || !sourceStrides ||
          !resultStrides ||
          offsets.size() != static_cast<size_t>(sourceType.getRank()) ||
          sizes.size() != static_cast<size_t>(sourceType.getRank()) ||
          strides.size() != static_cast<size_t>(sourceType.getRank()))
        return std::nullopt;

      for (int64_t dimension = 0; dimension < sourceType.getRank();
           ++dimension) {
        OpFoldResult offset = offsets[dimension];
        OpFoldResult size = sizes[dimension];
        OpFoldResult stride = strides[dimension];
        auto sizeValue = getConstantIntValue(size);
        auto subviewStride = getConstantIntValue(stride);
        int64_t composedStride;
        if (!sizeValue || *sizeValue <= 0 ||
            *sizeValue != resultType.getShape()[dimension] || !subviewStride ||
            *subviewStride <= 0 ||
            llvm::MulOverflow((*sourceStrides)[dimension], *subviewStride,
                              composedStride) ||
            (*resultStrides)[dimension] != composedStride)
          return std::nullopt;

        if (auto offsetValue = getConstantIntValue(offset)) {
          int64_t physicalOffset;
          if (*offsetValue < 0 ||
              llvm::MulOverflow(*offsetValue, (*sourceStrides)[dimension],
                                physicalOffset) ||
              llvm::AddOverflow(relativeOffsetElements, physicalOffset,
                                relativeOffsetElements))
            return std::nullopt;
        } else {
          Value dynamicValue = dyn_cast<Value>(offset);
          int64_t coefficientBytes;
          if (!dynamicValue ||
              llvm::MulOverflow((*sourceStrides)[dimension], *elementBytes,
                                coefficientBytes) ||
              !accumulateDynamicOffset(dynamicValue, coefficientBytes))
            return std::nullopt;
        }
      }
      cursor = source;
      continue;
    }

    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(definition)) {
      Value source = collapse.getSrc();
      MemRefType sourceType = getPositiveStrideStaticType(source);
      MemRefType resultType = getPositiveStrideStaticType(cursor);
      if (!sourceType || !resultType ||
          !hasCompatibleReassociationLayout(sourceType, resultType,
                                            collapse.getReassociationIndices(),
                                            /*collapsing=*/true))
        return std::nullopt;
      cursor = source;
      continue;
    }

    if (auto expand = dyn_cast<memref::ExpandShapeOp>(definition)) {
      Value source = expand.getSrc();
      MemRefType sourceType = getPositiveStrideStaticType(source);
      MemRefType resultType = getPositiveStrideStaticType(cursor);
      if (!sourceType || !resultType ||
          !hasCompatibleReassociationLayout(sourceType, resultType,
                                            expand.getReassociationIndices(),
                                            /*collapsing=*/false))
        return std::nullopt;
      cursor = source;
      continue;
    }

    if (auto cast = dyn_cast<memref::CastOp>(definition)) {
      Value source = cast.getSource();
      MemRefType sourceType = getPositiveStrideStaticType(source);
      MemRefType resultType = getPositiveStrideStaticType(cursor);
      auto sourceStrides = getPositiveStrides(sourceType);
      auto resultStrides = getPositiveStrides(resultType);
      if (!sourceType || !resultType ||
          sourceType.getElementType() != valueType.getElementType() ||
          resultType.getElementType() != valueType.getElementType() ||
          sourceType.getRank() != resultType.getRank() || !sourceStrides ||
          !resultStrides || sourceType.getShape() != resultType.getShape() ||
          *sourceStrides != *resultStrides)
        return std::nullopt;
      cursor = source;
      continue;
    }

    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(definition)) {
      Value source = cast.getSource();
      MemRefType sourceType = getPositiveStrideStaticType(source);
      MemRefType resultType = getPositiveStrideStaticType(cursor);
      auto sourceStrides = getPositiveStrides(sourceType);
      auto resultStrides = getPositiveStrides(resultType);
      int64_t sourceOffset, resultOffset;
      SmallVector<int64_t> ignoredStrides;
      if (!sourceType || !resultType || !sourceStrides || !resultStrides ||
          sourceType.getElementType() != valueType.getElementType() ||
          resultType.getElementType() != valueType.getElementType() ||
          sourceType.getShape() != resultType.getShape() ||
          *sourceStrides != *resultStrides ||
          failed(
              sourceType.getStridesAndOffset(ignoredStrides, sourceOffset)) ||
          failed(
              resultType.getStridesAndOffset(ignoredStrides, resultOffset)) ||
          sourceOffset != resultOffset)
        return std::nullopt;
      cursor = source;
      continue;
    }

    if (auto transpose = dyn_cast<memref::TransposeOp>(definition)) {
      Value source = transpose.getIn();
      MemRefType sourceType = getPositiveStrideStaticType(source);
      MemRefType resultType = getPositiveStrideStaticType(cursor);
      auto sourceStrides = getPositiveStrides(sourceType);
      auto resultStrides = getPositiveStrides(resultType);
      AffineMap permutation = transpose.getPermutation();
      if (!sourceType || !resultType || !sourceStrides || !resultStrides ||
          sourceType.getElementType() != valueType.getElementType() ||
          resultType.getElementType() != valueType.getElementType() ||
          sourceType.getRank() != resultType.getRank() ||
          permutation.getNumDims() != sourceType.getRank() ||
          permutation.getNumResults() != resultType.getRank())
        return std::nullopt;
      SmallVector<bool> seen(sourceType.getRank(), false);
      for (int64_t resultDimension = 0; resultDimension < resultType.getRank();
           ++resultDimension) {
        auto expression =
            dyn_cast<AffineDimExpr>(permutation.getResult(resultDimension));
        if (!expression || expression.getPosition() >= sourceType.getRank() ||
            seen[expression.getPosition()] ||
            sourceType.getShape()[expression.getPosition()] !=
                resultType.getShape()[resultDimension] ||
            (*sourceStrides)[expression.getPosition()] !=
                (*resultStrides)[resultDimension])
          return std::nullopt;
        seen[expression.getPosition()] = true;
      }
      if (llvm::any_of(seen, [](bool present) { return !present; }))
        return std::nullopt;
      cursor = source;
      continue;
    }

    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(definition)) {
      MemRefType resultType = getPositiveStrideStaticType(cursor);
      Value source = reinterpret.getSource();
      SmallVector<OpFoldResult> offsets = reinterpret.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = reinterpret.getMixedSizes();
      SmallVector<OpFoldResult> strides = reinterpret.getMixedStrides();
      auto resultStrides = getPositiveStrides(resultType);
      if (!resultType || !resultStrides ||
          resultType.getElementType() != valueType.getElementType() ||
          offsets.size() != 1 ||
          sizes.size() != static_cast<size_t>(resultType.getRank()) ||
          strides.size() != static_cast<size_t>(resultType.getRank()))
        return std::nullopt;

      auto reinterpretOffset = getConstantIntValue(offsets.front());
      if (reinterpretOffset && *reinterpretOffset < 0)
        return std::nullopt;
      if (!reinterpretOffset) {
        Value dynamicValue = dyn_cast<Value>(offsets.front());
        if (!dynamicValue ||
            !accumulateDynamicOffset(dynamicValue, *elementBytes))
          return std::nullopt;
      }
      for (auto [dimension, parts] :
           llvm::enumerate(llvm::zip(sizes, strides))) {
        auto sizeValue = getConstantIntValue(std::get<0>(parts));
        auto strideValue = getConstantIntValue(std::get<1>(parts));
        if (!sizeValue || *sizeValue <= 0 ||
            *sizeValue != resultType.getShape()[dimension] || !strideValue ||
            *strideValue <= 0 || (*resultStrides)[dimension] != *strideValue)
          return std::nullopt;
      }

      int64_t resultTypeOffset;
      SmallVector<int64_t> ignoredStrides;
      if (failed(
              resultType.getStridesAndOffset(ignoredStrides, resultTypeOffset)))
        return std::nullopt;
      if (reinterpretOffset) {
        if (resultTypeOffset != ShapedType::kDynamic &&
            resultTypeOffset != *reinterpretOffset)
          return std::nullopt;
      } else if (resultTypeOffset != ShapedType::kDynamic) {
        return std::nullopt;
      }

      Value root = getStorageRoot(value);
      auto rootType = dyn_cast<MemRefType>(root.getType());
      int64_t rootOffset;
      SmallVector<int64_t> rootStrides;
      // Reinterpret offsets and statically normalized root views share the
      // aligned-pointer-relative frame used by pairwise comparisons.
      if (!rootType || !isKnownBufferStorageRoot(root) ||
          getStorageRoot(source) != root ||
          rootType.getElementType() != valueType.getElementType() ||
          failed(rootType.getStridesAndOffset(rootStrides, rootOffset)) ||
          rootOffset == ShapedType::kDynamic || rootOffset < 0 ||
          (reinterpretOffset &&
           llvm::AddOverflow(relativeOffsetElements, *reinterpretOffset,
                             relativeOffsetElements)))
        return std::nullopt;

      // Only these view operations preserve the source aligned pointer.
      // In particular, memref.view can shift it independently of its offset.
      Value sourceBase = source;
      while (Operation *sourceDefinition = sourceBase.getDefiningOp()) {
        if (!isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
                 memref::TransposeOp>(sourceDefinition))
          break;
        sourceBase = sourceDefinition->getOperand(0);
      }
      if (canonicalizeBOSCAMEGlobalRoot(sourceBase) != root)
        return std::nullopt;

      // Reinterpret offsets are based on the shared aligned pointer, so the
      // source descriptor's offset and view chain do not contribute.
      hasReinterpretBase = true;
      break;
    }

    if (isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::IndexSwitchOp>(
            definition)) {
      DenseSet<Value> visited;
      Value source = getExactControlFlowPassthrough(cursor, visited);
      if (!source)
        return std::nullopt;
      cursor = source;
      continue;
    }

    // Unrealized casts carry no storage identity or descriptor semantics.
    if (isa<UnrealizedConversionCastOp>(definition))
      return std::nullopt;

    // Other storage views can alter pointers or layouts and are deliberately
    // not summarized.
    if (isStorageViewOperation(definition))
      return std::nullopt;
    break;
  }

  if (auto argument = dyn_cast<BlockArgument>(cursor)) {
    auto forOp = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (forOp && argument.getOwner() == forOp.getBody()) {
      Value init = getForIterArgInit(argument);
      if (!init)
        return std::nullopt;
      cursor = init;
    }
    if (isa<scf::WhileOp>(argument.getOwner()->getParentOp())) {
      DenseSet<Value> visited;
      Value source = getExactControlFlowPassthrough(cursor, visited);
      if (!source || source == cursor)
        return std::nullopt;
      cursor = source;
    }
  }

  Value root = getStorageRoot(value);
  auto valueStrides = getPositiveStrides(valueType);
  if (!valueStrides || getStorageRoot(cursor) != root)
    return std::nullopt;
  MemRefType ordinaryRootType;
  int64_t ordinaryRootOffset = ShapedType::kDynamic;
  if (!hasReinterpretBase) {
    ordinaryRootType = getPositiveStrideStaticType(cursor);
    if (!ordinaryRootType ||
        ordinaryRootType.getElementType() != valueType.getElementType())
      return std::nullopt;
    SmallVector<int64_t> rootStrides;
    if (failed(ordinaryRootType.getStridesAndOffset(rootStrides,
                                                    ordinaryRootOffset)))
      return std::nullopt;
  }

  int64_t startBytes;
  int64_t spanBytes;
  int64_t endBytes;
  int64_t spanElements;
  if (llvm::MulOverflow(relativeOffsetElements, *elementBytes, startBytes))
    return std::nullopt;
  if (ordinaryRootType && ordinaryRootOffset != ShapedType::kDynamic) {
    // Normalize statically known descriptor offsets into the same
    // aligned-pointer frame used by reinterpret_cast. A dynamic root offset
    // remains omitted because it cancels only between ordinary root views.
    auto rootElementBytes =
        getBOSCAMEElementBytes(ordinaryRootType.getElementType());
    int64_t rootOffsetBytes;
    if (!rootElementBytes ||
        llvm::MulOverflow(ordinaryRootOffset, *rootElementBytes,
                          rootOffsetBytes) ||
        llvm::AddOverflow(startBytes, rootOffsetBytes, startBytes))
      return std::nullopt;
  }
  spanElements = 1;
  for (auto [dimension, stride] :
       llvm::zip(valueType.getShape(), *valueStrides)) {
    int64_t strideExtent;
    if (llvm::MulOverflow(dimension - 1, stride, strideExtent) ||
        llvm::AddOverflow(spanElements, strideExtent, spanElements))
      return std::nullopt;
  }
  if (llvm::MulOverflow(spanElements, *elementBytes, spanBytes) ||
      llvm::AddOverflow(startBytes, spanBytes, endBytes))
    return std::nullopt;

  return SymbolicByteInterval{root, std::move(dynamicOffsets), startBytes,
                              endBytes};
}

static bool areSymbolicByteIntervalsDisjoint(const SymbolicByteInterval &lhs,
                                             const SymbolicByteInterval &rhs) {
  if (lhs.root != rhs.root ||
      lhs.dynamicOffsets.size() != rhs.dynamicOffsets.size())
    return false;
  for (const auto &lhsTerm : lhs.dynamicOffsets) {
    auto rhsTerm = llvm::find_if(rhs.dynamicOffsets, [&](const auto &term) {
      return term.value == lhsTerm.value;
    });
    if (rhsTerm == rhs.dynamicOffsets.end() ||
        rhsTerm->coefficientBytes != lhsTerm.coefficientBytes)
      return false;
  }
  return areByteIntervalsDisjoint({lhs.startBytes, lhs.endBytes},
                                  {rhs.startBytes, rhs.endBytes});
}

/// Keep exact row ranges for common pitched matrix views. Other layouts and
/// large row counts retain the bounding-interval fallback.
static std::optional<SymbolicByteRows> getSymbolicByteRows(Value value) {
  constexpr int64_t kMaxRows = 4096;
  auto interval = getSymbolicByteInterval(value);
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!interval || !type || type.getRank() != 2 || !type.hasStaticShape() ||
      type.getShape()[0] <= 0 || type.getShape()[0] > kMaxRows ||
      type.getShape()[1] <= 0)
    return std::nullopt;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != 2 || strides[0] <= 0 || strides[1] != 1)
    return std::nullopt;
  auto elementBytes = getBOSCAMEElementBytes(type.getElementType());
  if (!elementBytes)
    return std::nullopt;

  int64_t rowStrideBytes;
  int64_t rowBytes;
  int64_t finalRowOffset;
  int64_t finalRowEnd;
  if (llvm::MulOverflow(strides[0], *elementBytes, rowStrideBytes) ||
      llvm::MulOverflow(type.getShape()[1], *elementBytes, rowBytes) ||
      rowStrideBytes < rowBytes ||
      llvm::MulOverflow(type.getShape()[0] - 1, rowStrideBytes,
                        finalRowOffset) ||
      llvm::AddOverflow(interval->startBytes, finalRowOffset, finalRowOffset) ||
      llvm::AddOverflow(finalRowOffset, rowBytes, finalRowEnd) ||
      interval->startBytes < 0 || finalRowEnd > interval->endBytes)
    return std::nullopt;

  return SymbolicByteRows{std::move(*interval), type.getShape()[0],
                          rowStrideBytes, rowBytes};
}

static bool haveMatchingDynamicOffsets(const SymbolicByteInterval &lhs,
                                       const SymbolicByteInterval &rhs) {
  if (lhs.root != rhs.root ||
      lhs.dynamicOffsets.size() != rhs.dynamicOffsets.size())
    return false;
  return llvm::all_of(lhs.dynamicOffsets, [&](const auto &lhsTerm) {
    auto rhsTerm = llvm::find_if(rhs.dynamicOffsets, [&](const auto &term) {
      return term.value == lhsTerm.value;
    });
    return rhsTerm != rhs.dynamicOffsets.end() &&
           rhsTerm->coefficientBytes == lhsTerm.coefficientBytes;
  });
}

static bool areSymbolicByteRowsDisjoint(const SymbolicByteRows &lhs,
                                        const SymbolicByteRows &rhs) {
  if (!haveMatchingDynamicOffsets(lhs.interval, rhs.interval))
    return false;

  int64_t lhsRow = 0;
  int64_t rhsRow = 0;
  while (lhsRow < lhs.rowCount && rhsRow < rhs.rowCount) {
    int64_t lhsStart, rhsStart, lhsEnd, rhsEnd;
    if (llvm::MulOverflow(lhsRow, lhs.rowStrideBytes, lhsStart) ||
        llvm::AddOverflow(lhs.interval.startBytes, lhsStart, lhsStart) ||
        llvm::AddOverflow(lhsStart, lhs.rowBytes, lhsEnd) ||
        llvm::MulOverflow(rhsRow, rhs.rowStrideBytes, rhsStart) ||
        llvm::AddOverflow(rhs.interval.startBytes, rhsStart, rhsStart) ||
        llvm::AddOverflow(rhsStart, rhs.rowBytes, rhsEnd))
      return false;
    if (lhsEnd <= rhsStart) {
      ++lhsRow;
      continue;
    }
    if (rhsEnd <= lhsStart) {
      ++rhsRow;
      continue;
    }
    return false;
  }
  return true;
}

static std::optional<std::pair<int64_t, int64_t>>
getTransferByteInterval(const TransferPlan &transfer) {
  if (!transfer.effectiveByteOffsetBytes ||
      *transfer.effectiveByteOffsetBytes < 0 || transfer.elementBytes <= 0 ||
      transfer.shape.empty() ||
      transfer.shape.size() != transfer.strides.size())
    return std::nullopt;

  int64_t spanElements = 1;
  for (auto [dimension, stride] : llvm::zip(transfer.shape, transfer.strides)) {
    int64_t extent;
    if (dimension <= 0 || stride <= 0 ||
        llvm::MulOverflow(dimension - 1, stride, extent) ||
        llvm::AddOverflow(spanElements, extent, spanElements))
      return std::nullopt;
  }
  int64_t spanBytes;
  if (llvm::MulOverflow(spanElements, transfer.elementBytes, spanBytes))
    return std::nullopt;
  return getCheckedByteInterval(transfer.effectiveByteOffsetBytes, spanBytes);
}

static std::optional<std::pair<int64_t, int64_t>>
getBufferAccessByteInterval(const BufferAccessFact &access) {
  if (!access.physicalSpanBytes)
    return std::nullopt;
  return getCheckedByteInterval(access.effectiveByteOffsetBytes,
                                *access.physicalSpanBytes);
}

static bool areTransferByteIntervalsDisjoint(const TransferPlan &lhs,
                                             const TransferPlan &rhs) {
  auto lhsInterval = getTransferByteInterval(lhs);
  auto rhsInterval = getTransferByteInterval(rhs);
  if (lhsInterval && rhsInterval &&
      areByteIntervalsDisjoint(*lhsInterval, *rhsInterval))
    return true;
  auto lhsRows = getSymbolicByteRows(lhs.buffer);
  auto rhsRows = getSymbolicByteRows(rhs.buffer);
  if (lhsRows && rhsRows && areSymbolicByteRowsDisjoint(*lhsRows, *rhsRows))
    return true;
  auto lhsSymbolic = getSymbolicByteInterval(lhs.buffer);
  auto rhsSymbolic = getSymbolicByteInterval(rhs.buffer);
  return lhsSymbolic && rhsSymbolic && lhs.ddrRoot == rhs.ddrRoot &&
         areSymbolicByteIntervalsDisjoint(*lhsSymbolic, *rhsSymbolic);
}

static BufferAccessEffect getBufferAccessEffect(MemoryEffects::Effect *effect) {
  if (isa<MemoryEffects::Read>(effect))
    return BufferAccessEffect::Read;
  if (isa<MemoryEffects::Write>(effect))
    return BufferAccessEffect::Write;
  if (isa<MemoryEffects::Allocate>(effect))
    return BufferAccessEffect::Allocate;
  if (isa<MemoryEffects::Free>(effect))
    return BufferAccessEffect::Free;
  return BufferAccessEffect::Unknown;
}

static bool isKnownBufferStorageRoot(Value root) {
  if (!root)
    return false;
  if (auto argument = dyn_cast<BlockArgument>(root))
    return argument.getOwner()->isEntryBlock() &&
           isa<func::FuncOp>(argument.getOwner()->getParentOp());
  Operation *definition = root.getDefiningOp();
  return definition && (isa<memref::AllocOp, memref::AllocaOp,
                            memref::GetGlobalOp, SpmViewOp>(definition));
}

static std::optional<int64_t>
getBufferAccessOriginId(Operation *operation, BufferAccessEffect effect,
                        int64_t operandNumber,
                        const LinalgComputationFacts &linalgFacts) {
  if (operation->hasAttr("bosc_ame.lower_to_ame")) {
    StringRef originName;
    if (effect == BufferAccessEffect::Read && operandNumber == 0)
      originName = "bosc_ame.a_origin_id";
    else if (effect == BufferAccessEffect::Read && operandNumber == 1)
      originName = "bosc_ame.b_origin_id";
    else if ((effect == BufferAccessEffect::Read ||
              effect == BufferAccessEffect::Write) &&
             operandNumber == 2)
      originName = "bosc_ame.accumulator_origin_id";
    if (!originName.empty())
      if (auto origin = operation->getAttrOfType<IntegerAttr>(originName))
        return origin.getInt();
  }

  auto operationIndex = linalgFacts.operationIndices.find(operation);
  if (operationIndex != linalgFacts.operationIndices.end() &&
      operationIndex->second < linalgFacts.operations.size()) {
    const LinalgOperationFact &operationFact =
        linalgFacts.operations[operationIndex->second];
    if (operationFact.memoryEvent &&
        operationFact.memoryEvent->kind == LinalgMemoryEventKind::Copy) {
      const LinalgMemoryEventFact &event = *operationFact.memoryEvent;
      bool isLoad = event.role == LinalgMemoryEventRole::Load &&
                    event.effect == LinalgMemoryEventEffect::Read &&
                    effect == BufferAccessEffect::Read;
      bool isStore = event.role == LinalgMemoryEventRole::Store &&
                     event.effect == LinalgMemoryEventEffect::Write &&
                     effect == BufferAccessEffect::Write;
      if (!isLoad && !isStore)
        return std::nullopt;
      if (operandNumber < 0 || static_cast<size_t>(operandNumber) >=
                                   operationFact.operandValueIndices.size())
        return std::nullopt;
      size_t accessedValueIndex =
          operationFact.operandValueIndices[operandNumber];
      const auto &eventValues =
          isLoad ? event.inputValueIndices : event.outputValueIndices;
      if (!llvm::is_contained(eventValues, accessedValueIndex))
        return std::nullopt;
      return event.originId;
    }
  }

  auto memory = operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  auto kind = memory ? memory.getAs<StringAttr>("kind") : StringAttr();
  bool matchingEffect =
      kind &&
      ((kind.getValue() == "load" && effect == BufferAccessEffect::Read) ||
       (kind.getValue() == "store" && effect == BufferAccessEffect::Write));
  auto origin =
      matchingEffect ? memory.getAs<IntegerAttr>("origin_id") : IntegerAttr();
  return origin ? std::optional<int64_t>(origin.getInt()) : std::nullopt;
}

static std::optional<int64_t> getStaticElementCount(ArrayRef<int64_t> shape) {
  int64_t elements = 1;
  for (int64_t dimension : shape) {
    if (ShapedType::isDynamic(dimension) || dimension < 0 ||
        llvm::MulOverflow(elements, dimension, elements))
      return std::nullopt;
  }
  return elements;
}

static void populateBufferAccessLayout(BufferAccessFact &fact) {
  if (!fact.accessedValue)
    return;

  fact.accessedType = fact.accessedValue.getType();
  if (auto memref = dyn_cast<MemRefType>(fact.accessedType)) {
    fact.memorySpace = memref.getMemorySpace();
    fact.shape.assign(memref.getShape().begin(), memref.getShape().end());
    int64_t typeOffset;
    SmallVector<int64_t> strides;
    if (succeeded(memref.getStridesAndOffset(strides, typeOffset)))
      fact.strides = std::move(strides);
    else
      fact.strides.assign(memref.getRank(), ShapedType::kDynamic);

    fact.storageRoot = getStorageRoot(fact.accessedValue);
    fact.storageRootKnown = isKnownBufferStorageRoot(fact.storageRoot);
    if (fact.storageRootKnown) {
      Value cursor = fact.accessedValue;
      while (Operation *definition = cursor.getDefiningOp()) {
        if (!isStorageViewOperation(definition) &&
            !isa<memref::ReshapeOp>(definition))
          break;
        fact.viewChain.push_back(definition);
        if (definition->getNumOperands() == 0)
          break;
        cursor = definition->getOperand(0);
      }
    }

    if (auto origin = spmdma::getStaticPointerOrigin(fact.accessedValue);
        origin &&
        canonicalizeBOSCAMEGlobalRoot(origin->root) == fact.storageRoot)
      if (auto descriptorOffset =
              spmdma::getStaticDescriptorOffset(fact.accessedValue))
        if (auto elementBytes =
                getBOSCAMEElementBytes(memref.getElementType())) {
          int64_t descriptorOffsetBytes;
          int64_t effectiveOffsetBytes;
          if (!llvm::MulOverflow(*descriptorOffset, *elementBytes,
                                 descriptorOffsetBytes) &&
              !llvm::AddOverflow(origin->byteShift, descriptorOffsetBytes,
                                 effectiveOffsetBytes))
            fact.effectiveByteOffsetBytes = effectiveOffsetBytes;
        }

    if (auto elementBytes = getBOSCAMEElementBytes(memref.getElementType())) {
      fact.logicalShape = fact.shape;
      if (auto memory = fact.operation->getAttrOfType<DictionaryAttr>(
              "bosc_ame.memory")) {
        auto kind = memory.getAs<StringAttr>("kind");
        bool matchesEvent =
            kind && ((kind.getValue() == "load" &&
                      fact.effect == BufferAccessEffect::Read) ||
                     (kind.getValue() == "store" &&
                      fact.effect == BufferAccessEffect::Write));
        if (matchesEvent) {
          if (auto logicalShape =
                  memory.getAs<DenseI64ArrayAttr>("logical_shape"))
            fact.logicalShape.assign(logicalShape.asArrayRef().begin(),
                                     logicalShape.asArrayRef().end());
          if (auto validShape = memory.getAs<DenseI64ArrayAttr>("valid_shape"))
            fact.validShape.assign(validShape.asArrayRef().begin(),
                                   validShape.asArrayRef().end());
        }
      }
      if (fact.logicalShape.empty() && memref.getRank() != 0)
        fact.logicalShape = fact.shape;
      if (fact.validShape.empty())
        fact.validShape = fact.logicalShape;

      if (auto elements = getStaticElementCount(fact.validShape)) {
        int64_t bytes;
        if (!llvm::MulOverflow(*elements, *elementBytes, bytes))
          fact.logicalFootprintBytes = bytes;
      }

      auto elements = getStaticElementCount(fact.shape);
      if (elements && *elements == 0) {
        fact.physicalSpanBytes = 0;
      } else if (elements && fact.shape.size() == fact.strides.size() &&
                 llvm::all_of(fact.strides, [](int64_t stride) {
                   return !ShapedType::isDynamic(stride) && stride > 0;
                 })) {
        int64_t spanElements = 1;
        bool validSpan = true;
        for (auto [dimension, stride] : llvm::zip(fact.shape, fact.strides)) {
          int64_t extent;
          if (dimension <= 0 ||
              llvm::MulOverflow(dimension - 1, stride, extent) ||
              llvm::AddOverflow(spanElements, extent, spanElements)) {
            validSpan = false;
            break;
          }
        }
        int64_t spanBytes;
        if (validSpan &&
            !llvm::MulOverflow(spanElements, *elementBytes, spanBytes))
          fact.physicalSpanBytes = spanBytes;
      }
    }
  } else if (auto shaped = dyn_cast<ShapedType>(fact.accessedType)) {
    fact.shape.assign(shaped.getShape().begin(), shaped.getShape().end());
    fact.logicalShape = fact.shape;
    fact.validShape = fact.shape;
  }

  if (fact.validShape.empty() && fact.shape.empty())
    fact.validShape = fact.shape;
}

static void collectBufferAccessFacts(ModuleOp module, KernelFacts &plan,
                                     ArrayRef<TransferPlan> transfers = {}) {
  DenseMap<Value, int64_t> storageRootIds;
  auto append = [&](Operation *operation, int64_t effectIndex,
                    BufferAccessEffect effect, Value value,
                    int64_t operandNumber) {
    BufferAccessFact fact;
    fact.operation = operation;
    fact.effectIndex = effectIndex;
    fact.operandNumber = operandNumber;
    fact.effect = effect;
    fact.accessedValue = value;
    fact.computeTaskId =
        operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id")
            ? std::optional<int64_t>(
                  operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id")
                      .getInt())
            : std::nullopt;
    fact.sourceOriginId = getBufferAccessOriginId(operation, fact.effect,
                                                  operandNumber, plan.linalg);
    populateBufferAccessLayout(fact);

    if (fact.storageRootKnown) {
      auto root =
          storageRootIds.try_emplace(fact.storageRoot, storageRootIds.size());
      fact.storageRootId = root.first->second;
    }
    for (auto [index, transfer] : llvm::enumerate(transfers)) {
      bool selectedConsumer =
          transfer.firstConsumer == operation ||
          llvm::is_contained(transfer.consumerOperations, operation);
      if (selectedConsumer && transfer.buffer == value) {
        fact.transferIndex = index;
        break;
      }
    }
    plan.bufferAccesses.push_back(std::move(fact));
  };

  module.walk([&](Operation *operation) {
    if (isa<ModuleOp, func::FuncOp>(operation))
      return;

    if (auto linalg = dyn_cast<linalg::LinalgOp>(operation)) {
      int64_t effectIndex = 0;
      for (unsigned index = 0; index < linalg.getNumDpsInputs(); ++index) {
        OpOperand &input = operation->getOpOperand(index);
        append(operation, effectIndex++, BufferAccessEffect::Read, input.get(),
               input.getOperandNumber());
      }
      for (OpOperand &output : linalg.getDpsInitsMutable()) {
        if (linalg.payloadUsesValueFromOperand(&output))
          append(operation, effectIndex++, BufferAccessEffect::Read,
                 output.get(), output.getOperandNumber());
        append(operation, effectIndex++, BufferAccessEffect::Write,
               output.get(), output.getOperandNumber());
      }
      return;
    }

    // Recursive region effects are represented by their nested operations.
    if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return;

    if (auto effectsInterface = dyn_cast<MemoryEffectOpInterface>(operation)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      effectsInterface.getEffects(effects);
      for (auto [effectIndex, effect] : llvm::enumerate(effects)) {
        Value value = effect.getValue();
        OpOperand *operand = effect.getEffectValue<OpOperand *>();
        append(operation, effectIndex,
               getBufferAccessEffect(effect.getEffect()), value,
               operand ? static_cast<int64_t>(operand->getOperandNumber())
                       : -1);
      }
      return;
    }
    if (!isMemoryEffectFree(operation))
      append(operation, 0, BufferAccessEffect::Unknown, Value(), -1);
  });

  // Metadata refers to logical read/write slots, whereas local effects may
  // also include scalar operands and allocate/free effects. Bind only exact
  // logical effect arities; rewrites or unsupported contracts stay unknown.
  DenseMap<Operation *, SmallVector<size_t, 4>> reads, writes;
  for (auto [index, access] : llvm::enumerate(plan.bufferAccesses)) {
    if (!access.accessedValue ||
        !isa<MemRefType, TensorType>(access.accessedValue.getType()))
      continue;
    if (access.effect == BufferAccessEffect::Read)
      reads[access.operation].push_back(index);
    else if (access.effect == BufferAccessEffect::Write)
      writes[access.operation].push_back(index);
  }
  auto bind = [&](const auto &effects, StringRef attribute) {
    for (const auto &entry : effects) {
      auto ids =
          entry.first->template getAttrOfType<DenseI64ArrayAttr>(attribute);
      if (!ids || static_cast<size_t>(ids.size()) != entry.second.size())
        continue;
      for (auto [accessIndex, id] : llvm::zip(entry.second, ids.asArrayRef()))
        if (id >= 0)
          plan.bufferAccesses[accessIndex].logicalValueId = id;
    }
  };
  bind(reads, "bosc_ame.compute_input_values");
  bind(writes, "bosc_ame.compute_output_values");
}

static std::optional<int64_t> getStaticRootCapacityBytes(Value root) {
  if (!root ||
      !isa_and_nonnull<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(
          root.getDefiningOp()))
    return std::nullopt;

  auto type = dyn_cast<MemRefType>(root.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity())
    return std::nullopt;

  auto elementBytes = getBOSCAMEElementBytes(type.getElementType());
  auto elements = getStaticElementCount(type.getShape());
  int64_t capacityBytes;
  if (!elementBytes || !elements ||
      llvm::MulOverflow(*elements, *elementBytes, capacityBytes))
    return std::nullopt;
  return capacityBytes;
}

static void collectBufferInventoryFacts(DataflowPlan &plan) {
  DenseMap<Value, size_t> inventoryByValue;
  for (auto [accessIndex, access] : llvm::enumerate(plan.bufferAccesses)) {
    Value descriptor = access.accessedValue;
    if (!descriptor || !isa<BaseMemRefType>(descriptor.getType()))
      continue;

    auto [found, inserted] =
        inventoryByValue.try_emplace(descriptor, plan.bufferInventory.size());
    if (inserted) {
      BufferInventoryFact fact;
      fact.descriptor = descriptor;
      fact.storageRoot = access.storageRoot;
      fact.storageRootId = access.storageRootId;
      fact.storageRootKnown = access.storageRootKnown;
      fact.viewChain = access.viewChain;
      fact.descriptorType = access.accessedType;
      fact.memorySpace = access.memorySpace;
      fact.shape = access.shape;
      fact.strides = access.strides;
      fact.effectiveByteOffsetBytes = access.effectiveByteOffsetBytes;
      if (fact.storageRootKnown)
        fact.rootCapacityBytes = getStaticRootCapacityBytes(fact.storageRoot);
      if (access.operation)
        if (auto function = access.operation->getParentOfType<func::FuncOp>())
          fact.functionName = function.getNameAttr();
      plan.bufferInventory.push_back(std::move(fact));
    }

    BufferInventoryFact &fact = plan.bufferInventory[found->second];
    fact.accessIndices.push_back(accessIndex);
    if (access.logicalValueId)
      fact.logicalValueIds.push_back(*access.logicalValueId);
    if (access.computeTaskId)
      fact.taskIds.push_back(*access.computeTaskId);
    if (access.transferIndex)
      fact.transferIndices.push_back(*access.transferIndex);

    if (fact.storageRootKnown != access.storageRootKnown ||
        (fact.storageRootKnown &&
         (fact.storageRoot != access.storageRoot ||
          fact.storageRootId != access.storageRootId))) {
      fact.storageRoot = {};
      fact.storageRootId = -1;
      fact.storageRootKnown = false;
      fact.rootCapacityBytes.reset();
    }
    if (fact.effectiveByteOffsetBytes != access.effectiveByteOffsetBytes)
      fact.effectiveByteOffsetBytes.reset();
    if (fact.storageRootKnown &&
        fact.rootCapacityBytes !=
            getStaticRootCapacityBytes(access.storageRoot))
      fact.rootCapacityBytes.reset();
  }

  for (BufferInventoryFact &fact : plan.bufferInventory) {
    llvm::sort(fact.logicalValueIds);
    fact.logicalValueIds.erase(
        std::unique(fact.logicalValueIds.begin(), fact.logicalValueIds.end()),
        fact.logicalValueIds.end());
    llvm::sort(fact.taskIds);
    fact.taskIds.erase(std::unique(fact.taskIds.begin(), fact.taskIds.end()),
                       fact.taskIds.end());
    llvm::sort(fact.transferIndices);
    fact.transferIndices.erase(
        std::unique(fact.transferIndices.begin(), fact.transferIndices.end()),
        fact.transferIndices.end());
  }
}

static void collectBufferSourceEvents(ModuleOp module, DataflowPlan &plan) {
  DataflowPlan snapshot;
  snapshot.linalg = analyzeLinalgComputationFacts(module);
  collectBufferAccessFacts(module, snapshot);
  for (const BufferAccessFact &access : snapshot.bufferAccesses) {
    if (!access.sourceOriginId || !access.operation)
      continue;
    BufferSourceEventFact event;
    if (auto function = access.operation->getParentOfType<func::FuncOp>())
      event.functionName = function.getNameAttr();
    event.operationName = access.operation->getName().getStringRef().str();
    event.effectIndex = access.effectIndex;
    event.operandNumber = access.operandNumber;
    event.effect = access.effect;
    event.sourceOriginId = *access.sourceOriginId;
    event.accessedType = access.accessedType;
    event.memorySpace = access.memorySpace;
    for (Operation *view : access.viewChain)
      event.viewChain.push_back(view->getName().getStringRef().str());
    event.shape = access.shape;
    event.logicalShape = access.logicalShape;
    event.strides = access.strides;
    event.validShape = access.validShape;
    event.effectiveByteOffsetBytes = access.effectiveByteOffsetBytes;
    event.logicalFootprintBytes = access.logicalFootprintBytes;
    event.physicalSpanBytes = access.physicalSpanBytes;
    plan.bufferSourceEvents.push_back(std::move(event));
  }
}

static BufferAliasContext getBufferAliasContext(ModuleOp module,
                                                StringRef abi) {
  BufferAliasContext context;
  if (abi == spmdma::kDmaAbiAlignedNonAliasing)
    module.walk([&](func::FuncOp function) {
      if (function.isExternal() || !function.isPublic())
        return;
      auto symbolUses = SymbolTable::getSymbolUses(function, module);
      if (symbolUses && symbolUses->empty())
        context.abiEntryFunctions.insert(function.getOperation());
    });
  return context;
}

static std::optional<BufferAliasProof>
getDistinctRootProof(Value lhs, Value rhs, func::FuncOp function,
                     const BufferAliasContext &context) {
  if (!lhs || !rhs || lhs == rhs || !isKnownBufferStorageRoot(lhs) ||
      !isKnownBufferStorageRoot(rhs))
    return std::nullopt;
  auto lhsType = dyn_cast<BaseMemRefType>(lhs.getType());
  auto rhsType = dyn_cast<BaseMemRefType>(rhs.getType());
  auto lhsArgument = dyn_cast<BlockArgument>(lhs);
  auto rhsArgument = dyn_cast<BlockArgument>(rhs);
  if (context.abiEntryFunctions.contains(function.getOperation()) && lhsType &&
      rhsType && !lhsType.getMemorySpace() && !rhsType.getMemorySpace() &&
      lhsArgument && rhsArgument &&
      lhsArgument.getOwner() == rhsArgument.getOwner() &&
      lhsArgument.getOwner()->isEntryBlock() &&
      lhsArgument.getOwner()->getParentOp() == function)
    return BufferAliasProof::AbiNonAliasingEntryArguments;

  auto lhsGlobal = lhs.getDefiningOp<memref::GetGlobalOp>();
  auto rhsGlobal = rhs.getDefiningOp<memref::GetGlobalOp>();
  if (lhsGlobal && rhsGlobal &&
      lhsGlobal.getNameAttr() != rhsGlobal.getNameAttr())
    return BufferAliasProof::DistinctGlobalSymbols;

  // A fresh local allocation cannot alias another recognized DDR root. An
  // unresolved root may itself be an escaped alias of the allocation, and
  // custom memory-space allocation semantics are deliberately not assumed.
  if (lhsType && rhsType && !lhsType.getMemorySpace() &&
      !rhsType.getMemorySpace() &&
      (lhs.getDefiningOp<memref::AllocOp>() ||
       lhs.getDefiningOp<memref::AllocaOp>() ||
       rhs.getDefiningOp<memref::AllocOp>() ||
       rhs.getDefiningOp<memref::AllocaOp>()))
    return BufferAliasProof::FreshAllocation;
  return std::nullopt;
}

static std::optional<BufferAliasProof>
getDisjointAccessProof(const BufferAccessFact &lhs, const BufferAccessFact &rhs,
                       const BufferAliasContext &context) {
  if (lhs.effect == BufferAccessEffect::Unknown ||
      rhs.effect == BufferAccessEffect::Unknown || !lhs.storageRootKnown ||
      !rhs.storageRootKnown)
    return std::nullopt;
  if (lhs.storageRoot != rhs.storageRoot) {
    auto function = lhs.operation->getParentOfType<func::FuncOp>();
    if (!function || rhs.operation->getParentOfType<func::FuncOp>() != function)
      return std::nullopt;
    return getDistinctRootProof(lhs.storageRoot, rhs.storageRoot, function,
                                context);
  }

  // Lifetime ordering follows storage identity, even for an empty payload.
  if (lhs.effect == BufferAccessEffect::Allocate ||
      lhs.effect == BufferAccessEffect::Free ||
      rhs.effect == BufferAccessEffect::Allocate ||
      rhs.effect == BufferAccessEffect::Free)
    return std::nullopt;

  auto lhsInterval = getBufferAccessByteInterval(lhs);
  auto rhsInterval = getBufferAccessByteInterval(rhs);
  if (lhsInterval && rhsInterval) {
    if (areByteIntervalsDisjoint(*lhsInterval, *rhsInterval))
      return BufferAliasProof::SameRootByteIntervals;
  }
  auto lhsSymbolic = getSymbolicByteInterval(lhs.accessedValue);
  auto rhsSymbolic = getSymbolicByteInterval(rhs.accessedValue);
  if (lhsSymbolic && rhsSymbolic &&
      areSymbolicByteIntervalsDisjoint(*lhsSymbolic, *rhsSymbolic))
    return BufferAliasProof::SameRootSymbolicByteIntervals;
  auto lhsRows = getSymbolicByteRows(lhs.accessedValue);
  auto rhsRows = getSymbolicByteRows(rhs.accessedValue);
  if (lhsRows && rhsRows && areSymbolicByteRowsDisjoint(*lhsRows, *rhsRows))
    return BufferAliasProof::SameRootPitchedRowIntervals;
  return std::nullopt;
}

/// Compare known storage roots within each function. Same-root intervals use
/// checked bounding ranges, while cross-root disjointness needs a named ABI or
/// global-symbol contract. Other cross-root pairs remain absent (unknown).
static void collectBufferAliasPairs(ModuleOp module, KernelFacts &plan,
                                    const BufferAliasContext &context) {
  constexpr size_t kMaxBufferAliasPairs = 4096;
  struct RootAccessGroup {
    Value root;
    SmallVector<size_t, 4> accessIndices;
  };

  bool complete = true;

  auto appendPair = [&](size_t firstIndex, size_t secondIndex,
                        StringAttr functionName, BufferAliasRelation relation,
                        BufferAliasProof proof) {
    if (plan.bufferAliasPairs.size() == kMaxBufferAliasPairs) {
      complete = false;
      return false;
    }
    if (firstIndex > secondIndex)
      std::swap(firstIndex, secondIndex);
    plan.bufferAliasPairs.push_back(
        {firstIndex, secondIndex, functionName, relation, proof});
    return true;
  };

  module.walk([&](func::FuncOp function) {
    if (!complete)
      return;

    SmallVector<RootAccessGroup, 8> groups;
    DenseMap<Value, size_t> groupIndices;
    for (auto [accessIndex, access] : llvm::enumerate(plan.bufferAccesses)) {
      if (!access.operation ||
          access.operation->getParentOfType<func::FuncOp>() != function ||
          !access.storageRootKnown ||
          access.effect == BufferAccessEffect::Allocate ||
          access.effect == BufferAccessEffect::Free)
        continue;
      auto [iterator, inserted] =
          groupIndices.try_emplace(access.storageRoot, groups.size());
      if (inserted)
        groups.push_back({access.storageRoot, {}});
      groups[iterator->second].accessIndices.push_back(accessIndex);
    }

    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
      const RootAccessGroup &group = groups[groupIndex];
      for (size_t i = 0; i < group.accessIndices.size(); ++i) {
        for (size_t j = i + 1; j < group.accessIndices.size(); ++j) {
          const BufferAccessFact &lhs =
              plan.bufferAccesses[group.accessIndices[i]];
          const BufferAccessFact &rhs =
              plan.bufferAccesses[group.accessIndices[j]];
          auto proof = getDisjointAccessProof(lhs, rhs, context);
          if (!appendPair(group.accessIndices[i], group.accessIndices[j],
                          function.getNameAttr(),
                          proof ? BufferAliasRelation::Disjoint
                                : BufferAliasRelation::MayAlias,
                          proof.value_or(BufferAliasProof::SameRoot)))
            return;
        }
      }

      for (size_t otherIndex = groupIndex + 1; otherIndex < groups.size();
           ++otherIndex) {
        const RootAccessGroup &other = groups[otherIndex];
        std::optional<BufferAliasProof> proof =
            getDistinctRootProof(group.root, other.root, function, context);
        if (!proof)
          continue;
        for (size_t accessIndex : group.accessIndices)
          for (size_t otherAccessIndex : other.accessIndices) {
            auto accessProof = getDisjointAccessProof(
                plan.bufferAccesses[accessIndex],
                plan.bufferAccesses[otherAccessIndex], context);
            if (!accessProof)
              continue;
            if (!appendPair(accessIndex, otherAccessIndex,
                            function.getNameAttr(),
                            BufferAliasRelation::Disjoint, *accessProof))
              return;
          }
      }
      if (!complete)
        return;
    }
  });

  llvm::stable_sort(plan.bufferAliasPairs,
                    [](const auto &lhs, const auto &rhs) {
                      if (lhs.firstAccessIndex != rhs.firstAccessIndex)
                        return lhs.firstAccessIndex < rhs.firstAccessIndex;
                      return lhs.secondAccessIndex < rhs.secondAccessIndex;
                    });
  plan.bufferAliasPairsComplete = complete;
}

static bool isPhysicalMemoryAccess(const BufferAccessFact &access) {
  if (!access.accessedValue || access.effect == BufferAccessEffect::Unknown)
    return true;
  if (isa<TensorType>(access.accessedValue.getType()))
    return false;
  // Linalg records scalar DPS inputs alongside its buffer inputs; those SSA
  // values are not memory targets. Other effect interfaces may name pointer
  // or dialect-specific address values: retain those as unresolved storage.
  return !isa<linalg::LinalgOp>(access.operation) ||
         isa<MemRefType, UnrankedMemRefType>(access.accessedValue.getType());
}

static bool isLifetimeEffect(BufferAccessEffect effect) {
  return effect == BufferAccessEffect::Allocate ||
         effect == BufferAccessEffect::Free;
}

struct MemoryAccessPosition {
  Operation *anchor = nullptr;
  Operation *nestedOperation = nullptr;
  bool supported = false;
};

static MemoryAccessPosition getMemoryAccessPosition(Operation *operation,
                                                    func::FuncOp function) {
  if (!operation)
    return {};

  if (operation->getParentRegion() == &function.getBody())
    return {operation, nullptr, true};

  Region *parentRegion = operation->getParentRegion();
  auto executeRegion =
      parentRegion ? dyn_cast<scf::ExecuteRegionOp>(parentRegion->getParentOp())
                   : scf::ExecuteRegionOp();
  if (!executeRegion || &executeRegion.getRegion() != parentRegion ||
      executeRegion->getParentRegion() != &function.getBody() ||
      !executeRegion.getRegion().hasOneBlock())
    return {};

  Block &regionBlock = executeRegion.getRegion().front();
  if (operation->getBlock() != &regionBlock ||
      !isa<scf::YieldOp>(regionBlock.getTerminator()))
    return {};
  return {executeRegion, operation, true};
}

static std::optional<bool>
isMemoryAccessBefore(const MemoryAccessPosition &lhs,
                     const MemoryAccessPosition &rhs) {
  if (!lhs.supported || !rhs.supported)
    return std::nullopt;
  if (lhs.anchor == rhs.anchor) {
    if (!lhs.nestedOperation || !rhs.nestedOperation)
      return std::nullopt;
    return lhs.nestedOperation->isBeforeInBlock(rhs.nestedOperation);
  }
  if (lhs.anchor->getBlock() != rhs.anchor->getBlock())
    return std::nullopt;
  return lhs.anchor->isBeforeInBlock(rhs.anchor);
}

static void collectMemoryDependencies(KernelFacts &facts,
                                      const BufferAliasContext &aliases) {
  constexpr size_t kMaxComparisons = 65536;
  constexpr size_t kMaxDependencies = 8192;
  DenseMap<Operation *, SmallVector<size_t, 8>> functionAccesses;
  DenseMap<Operation *, MemoryAccessPosition> accessPositions;
  SmallVector<Operation *> functions;
  for (auto [index, access] : llvm::enumerate(facts.bufferAccesses)) {
    if (!access.operation || !isPhysicalMemoryAccess(access))
      continue;
    auto function = access.operation->getParentOfType<func::FuncOp>();
    if (!function)
      continue;
    if (!functionAccesses.contains(function))
      functions.push_back(function);
    functionAccesses[function].push_back(index);
    if (function.getBody().getBlocks().size() != 1)
      facts.memoryDependenciesComplete = false;
    auto position = getMemoryAccessPosition(access.operation, function);
    accessPositions.try_emplace(access.operation, position);
    // Only direct function-body operations and one-block execute_region
    // bodies have a proven lexical order. Other regions may be conditional,
    // loop-carried, nested or CFG-driven.
    if (!position.supported)
      facts.memoryDependenciesComplete = false;
  }

  size_t comparisons = 0;
  for (Operation *functionOperation : functions) {
    auto function = cast<func::FuncOp>(functionOperation);
    const auto &indices = functionAccesses[functionOperation];
    for (size_t i = 0; i < indices.size(); ++i)
      for (size_t j = i + 1; j < indices.size(); ++j) {
        if (++comparisons > kMaxComparisons ||
            facts.memoryDependencies.size() == kMaxDependencies) {
          facts.memoryDependenciesComplete = false;
          facts.memoryDependenciesBounded = true;
          return;
        }
        size_t predecessor = indices[i], successor = indices[j];
        const auto &lhs = facts.bufferAccesses[predecessor];
        const auto &rhs = facts.bufferAccesses[successor];
        if (lhs.operation == rhs.operation ||
            (lhs.effect == BufferAccessEffect::Read &&
             rhs.effect == BufferAccessEffect::Read))
          continue;
        if (getDisjointAccessProof(lhs, rhs, aliases))
          continue;
        auto order = isMemoryAccessBefore(accessPositions[lhs.operation],
                                          accessPositions[rhs.operation]);
        if (!order) {
          facts.memoryDependenciesComplete = false;
          continue;
        }
        if (!*order)
          std::swap(predecessor, successor);
        const auto &before = facts.bufferAccesses[predecessor];
        const auto &after = facts.bufferAccesses[successor];
        MemoryDependencyKind kind;
        if (before.effect == BufferAccessEffect::Unknown ||
            after.effect == BufferAccessEffect::Unknown)
          kind = MemoryDependencyKind::UnknownEffect;
        else if (isLifetimeEffect(before.effect) ||
                 isLifetimeEffect(after.effect))
          kind = MemoryDependencyKind::Lifetime;
        else if (before.effect == BufferAccessEffect::Write &&
                 after.effect == BufferAccessEffect::Read)
          kind = MemoryDependencyKind::RAW;
        else if (before.effect == BufferAccessEffect::Read &&
                 after.effect == BufferAccessEffect::Write)
          kind = MemoryDependencyKind::WAR;
        else
          kind = MemoryDependencyKind::WAW;
        facts.memoryDependencies.push_back(
            {predecessor, successor, function.getNameAttr(), kind});
      }
  }
}

static bool isResidencyWindowSafe(const ResidentCandidate &candidate,
                                  const KernelFacts &facts,
                                  const BufferAliasContext &aliases) {
  if (!facts.memoryDependenciesComplete || facts.memoryDependenciesBounded ||
      candidate.consumers.empty())
    return false;
  Operation *first =
      candidate.producer ? candidate.producer : candidate.consumers.front();
  Operation *last = candidate.consumers.back();
  BufferAccessFact materialization;
  materialization.operation = first;
  materialization.effect = BufferAccessEffect::Read;
  materialization.accessedValue = candidate.value;
  populateBufferAccessLayout(materialization);
  auto function = first->getParentOfType<func::FuncOp>();
  for (const BufferAccessFact &access : facts.bufferAccesses) {
    Operation *operation = access.operation;
    if (!operation || !isPhysicalMemoryAccess(access) ||
        operation->getParentOfType<func::FuncOp>() != function)
      continue;
    if (operation->getBlock() != first->getBlock())
      return false;
    if (operation->isBeforeInBlock(first) || last->isBeforeInBlock(operation))
      continue;
    bool designatedRead = access.effect == BufferAccessEffect::Read &&
                          access.accessedValue == candidate.value &&
                          llvm::is_contained(candidate.consumers, operation);
    bool designatedWrite = operation == candidate.producer &&
                           access.accessedValue == candidate.value &&
                           (access.effect == BufferAccessEffect::Read ||
                            access.effect == BufferAccessEffect::Write);
    if (designatedRead || designatedWrite)
      continue;
    if (getDisjointAccessProof(materialization, access, aliases))
      continue;
    // An input cache must observe writes before its next read. A producer's
    // delayed DDR write must also remain visible to every other reader.
    if (candidate.producer || access.effect != BufferAccessEffect::Read)
      return false;
  }
  return true;
}

static void refinePhysicalLifetimes(DataflowPlan &plan,
                                    const BufferAliasContext &aliases) {
  for (BufferRequirement &buffer : plan.buffers) {
    if (!buffer.physicalLifetimeKnown)
      continue;
    if (!plan.memoryDependenciesComplete || plan.memoryDependenciesBounded) {
      buffer.physicalLifetimeKnown = false;
    } else {
      BufferAccessFact tracked;
      tracked.operation = plan.transfers[buffer.transferIndex].firstConsumer;
      tracked.effect = BufferAccessEffect::Read;
      tracked.accessedValue = buffer.buffer;
      populateBufferAccessLayout(tracked);
      for (const auto &access : plan.bufferAccesses) {
        if (!isPhysicalMemoryAccess(access) || !access.operation ||
            access.operation->getParentOfType<func::FuncOp>() !=
                tracked.operation->getParentOfType<func::FuncOp>() ||
            getDisjointAccessProof(tracked, access, aliases))
          continue;
        if (!access.storageRootKnown ||
            access.storageRoot != buffer.storageRoot ||
            access.effect == BufferAccessEffect::Unknown ||
            (!access.computeTaskId && !isLifetimeEffect(access.effect))) {
          buffer.physicalLifetimeKnown = false;
          break;
        }
      }
    }
    if (!buffer.physicalLifetimeKnown) {
      buffer.physicalFirstTask.reset();
      buffer.physicalLastTask.reset();
    }
  }
}

/// A logical A/B load can be materialized directly only when the storage root
/// has no observable consumer other than the selected AME operand.  Looking at
/// marked matmuls alone is insufficient: a vector op, copy, call, or unknown
/// side effect may read or write the same root between the producer and the
/// dot.  View construction and deallocation do not consume the logical value
/// and are therefore ignored here.
static bool hasOnlySelectedAmeUse(linalg::MatmulOp current, Value operand,
                                  Value root) {
  (void)operand;
  auto module = current->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  bool valid = true;
  module.walk([&](Operation *operation) {
    if (!valid || operation == current.getOperation())
      return;
    if (isStorageViewOperation(operation) || isa<memref::DeallocOp>(operation))
      return;
    for (OpOperand &use : operation->getOpOperands()) {
      if (getStorageRoot(use.get()) != root)
        continue;
      // The selected matmul is handled above.  Any remaining operation that
      // names the root is an observable use, including another marked dot.
      valid = false;
      return;
    }
  });
  return valid;
}

/// A direct logical operand must have one AME consumer.  Counting by storage
/// root keeps a value shared by two marked matmuls on the conservative SPM
/// path, where liveness analysis can provide an actual reusable window.
static bool isSingleLogicalAmeInput(linalg::MatmulOp current, Value operand,
                                    unsigned inputIndex,
                                    int64_t alignmentBytes) {
  if (!hasLogicalPlacement(current) ||
      !current->hasAttr(::buddy::boscame::lowering::kLowerToAme) ||
      !isExternalDdrBuffer(operand))
    return false;
  auto type = dyn_cast<MemRefType>(operand.getType());
  TileRole role = inputIndex == 0 ? TileRole::A : TileRole::B;
  auto layout = type ? computeTileStorageLayout(type, role)
                     : std::optional<TileStorageLayout>();
  if (!layout || alignmentBytes <= 0 ||
      layout->rowStrideBytes % alignmentBytes != 0)
    return false;
  unsigned width =
      type ? spmdma::getSpmElementByteWidth(type.getElementType()) : 0;
  if (!width ||
      !spmdma::isDescriptorOffsetAligned(
          operand, alignmentBytes /
                       std::gcd(alignmentBytes, static_cast<int64_t>(width))))
    return false;
  if (!spmdma::isEndpointPointerAlignmentProven(operand, alignmentBytes))
    return false;
  Value root = getStorageRoot(operand);
  if (!hasOnlySelectedAmeUse(current, operand, root))
    return false;
  unsigned consumers = 0;
  auto module = current->getParentOfType<ModuleOp>();
  module.walk([&](linalg::MatmulOp candidate) {
    if (!candidate->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      return;
    for (Value input : candidate.getInputs())
      if (getStorageRoot(input) == root) {
        ++consumers;
        break;
      }
  });
  return consumers == 1;
}

static int64_t getAmeTileCount(int64_t dimension, int64_t tile) {
  return 1 + (dimension - 1) / tile;
}

static int64_t getAmeTailSize(int64_t dimension, int64_t tile) {
  int64_t remainder = dimension % tile;
  return remainder ? remainder : tile;
}

static void
appendAmeMatmulTilePlan(linalg::MatmulOp matmul, int64_t tileM, int64_t tileN,
                        int64_t tileK,
                        llvm::SmallVectorImpl<AmeMatmulTilePlan> &plans) {
  AmeMatmulTilePlan plan;
  plan.matmul = matmul.getOperation();
  plan.tileM = tileM;
  plan.tileN = tileN;
  plan.tileK = tileK;

  if (matmul.getInputs().size() < 2 || matmul.getOutputs().empty()) {
    plans.push_back(plan);
    return;
  }
  auto a = dyn_cast<MemRefType>(matmul.getInputs()[0].getType());
  auto c = dyn_cast<MemRefType>(matmul.getOutputs()[0].getType());
  if (a && c && a.getRank() == 2 && c.getRank() == 2 && a.hasStaticShape() &&
      c.hasStaticShape()) {
    plan.m = c.getDimSize(0);
    plan.n = c.getDimSize(1);
    plan.k = a.getDimSize(1);
    if (plan.m > 0 && plan.n > 0 && plan.k > 0) {
      plan.mTileCount = getAmeTileCount(plan.m, tileM);
      plan.nTileCount = getAmeTileCount(plan.n, tileN);
      plan.kTileCount = getAmeTileCount(plan.k, tileK);
      plan.tailM = getAmeTailSize(plan.m, tileM);
      plan.tailN = getAmeTailSize(plan.n, tileN);
      plan.tailK = getAmeTailSize(plan.k, tileK);
    }
  }
  plans.push_back(plan);
}

static bool directAmeInputTilesLegal(linalg::MatmulOp matmul,
                                     const AmeMatmulTilePlan *tilePlan,
                                     const TargetInfo &target,
                                     unsigned inputIndex) {
  auto input = dyn_cast<MemRefType>(matmul.getInputs()[inputIndex].getType());
  const AmeRoleCapability &role = inputIndex == 0 ? target.ame.a : target.ame.b;
  if (!tilePlan || tilePlan->k <= 0 || tilePlan->tileK <= 0 || !input ||
      !input.hasStaticShape() || !role.supportsDirectDdr ||
      !hasAmeTileLayout(input, role.layout))
    return false;
  int64_t blockK = tilePlan->tileK;
  int64_t alignment = target.dma.endpointAlignmentBytes;
  int64_t elementBytes = role.elementBits / 8;
  if (alignment <= 0 || elementBytes <= 0 || blockK > target.ame.maxTileK)
    return false;
  int64_t kGranularity = alignment / elementBytes;
  int64_t kDimension =
      inputIndex == 0 ? input.getDimSize(1) : input.getDimSize(0);
  if (kGranularity <= 0 || kDimension % kGranularity != 0)
    return false;
  return blockK > 0 && blockK % kGranularity == 0;
}

static std::optional<uint64_t>
checkedAmeInputBytes(int64_t first, int64_t second, int64_t elementBytes) {
  int64_t elements;
  int64_t bytes;
  if (first <= 0 || second <= 0 || elementBytes <= 0 ||
      llvm::MulOverflow(first, second, elements) ||
      llvm::MulOverflow(elements, elementBytes, bytes) || bytes <= 0)
    return std::nullopt;
  return static_cast<uint64_t>(bytes);
}

static std::optional<uint64_t> checkedAmeDirectInputBytes(uint64_t stagedBytes,
                                                          int64_t tileCount) {
  if (!stagedBytes || tileCount <= 0 ||
      stagedBytes > std::numeric_limits<uint64_t>::max() /
                        static_cast<uint64_t>(tileCount))
    return std::nullopt;
  return stagedBytes * static_cast<uint64_t>(tileCount);
}

static bool directAmeOutputTilesLegal(linalg::MatmulOp matmul,
                                      const AmeMatmulTilePlan *tilePlan,
                                      const TargetInfo &target) {
  auto c = dyn_cast<MemRefType>(matmul.getOutputs()[0].getType());
  if (!tilePlan || tilePlan->n <= 0 || tilePlan->tileN <= 0 || !c ||
      !c.hasStaticShape() || !target.ame.accumulator.supportsDirectDdr ||
      !hasAmeTileLayout(c, target.ame.accumulator.layout))
    return false;
  auto layout = computeTileStorageLayout(c, TileRole::Accumulator);
  int64_t alignment = target.dma.endpointAlignmentBytes;
  int64_t elementBytes = target.ame.accumulator.elementBits / 8;
  if (!layout || alignment <= 0 || elementBytes <= 0 ||
      layout->rowStrideBytes % alignment != 0 ||
      tilePlan->tileN > target.ame.maxTileN)
    return false;
  int64_t blockN = tilePlan->tileN;
  auto cElement = dyn_cast<IntegerType>(c.getElementType());
  if (!cElement || cElement.getWidth() != target.ame.accumulator.elementBits)
    return false;
  int64_t nGranularity = alignment / elementBytes;
  if (nGranularity <= 0 || c.getDimSize(1) % nGranularity != 0)
    return false;
  return blockN > 0 && blockN % nGranularity == 0;
}

static bool directAmeAddressLegal(Value value, int64_t alignmentBytes) {
  auto type = dyn_cast<MemRefType>(value.getType());
  unsigned width =
      type ? spmdma::getSpmElementByteWidth(type.getElementType()) : 0;
  return alignmentBytes > 0 && width &&
         spmdma::isDescriptorOffsetAligned(
             value, alignmentBytes / std::gcd(alignmentBytes,
                                              static_cast<int64_t>(width))) &&
         spmdma::isEndpointPointerAlignmentProven(value, alignmentBytes);
}

static bool directAmeAddressKnownMisaligned(Value value,
                                            int64_t alignmentBytes) {
  auto type = dyn_cast<MemRefType>(value.getType());
  unsigned width =
      type ? spmdma::getSpmElementByteWidth(type.getElementType()) : 0;
  auto offset = spmdma::getStaticDescriptorOffset(value);
  if (alignmentBytes <= 0 || !width || !offset)
    return false;
  int64_t factor =
      alignmentBytes / std::gcd(alignmentBytes, static_cast<int64_t>(width));
  if (*offset % factor != 0)
    return true;
  // Keep a statically misaligned `memref.view` on the direct path long enough
  // for the tile verifier to issue its direct-DDR diagnostic.  A dynamic
  // shift has no static origin and therefore remains an ordinary fallback
  // candidate (the DMA analyzer will apply its own endpoint proof).
  auto origin = spmdma::getStaticPointerOrigin(value);
  return origin && origin->byteShift % alignmentBytes != 0;
}

static void markLogicalAmeInputs(linalg::MatmulOp matmul,
                                 const LinalgToBOSCAMEConfig &config,
                                 AmeMatmulTilePlan *tilePlan,
                                 const TargetInfo &target) {
  matmul->removeAttr(kDirectAOperandAttr);
  matmul->removeAttr(kDirectBOperandAttr);
  matmul->removeAttr(kDirectCOperandAttr);
  if (!matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme) ||
      matmul.getInputs().size() != 2)
    return;
  // The accumulator is a separate placement decision from logical A/B
  // inputs.  Mark it for every proven direct-DDR AME output, including the
  // buffer-form compatibility path where the module has no logical marker.
  // An unprovable or narrow output is diagnosed by verifyDirectAmeOutput below
  // in the default direct-ddr mode. Explicit spm-dma remains the compatibility
  // path for such output layouts.
  auto output = matmul.getOutputs().front();
  if (config.storePolicy == "direct-ddr" &&
      directAmeOutputTilesLegal(matmul, tilePlan, target) &&
      isExternalDdrBuffer(output) &&
      (directAmeAddressLegal(output, target.dma.endpointAlignmentBytes) ||
       directAmeAddressKnownMisaligned(output,
                                       target.dma.endpointAlignmentBytes)))
    matmul->setAttr(kDirectCOperandAttr, UnitAttr::get(matmul.getContext()));
  if (!hasLogicalPlacement(matmul))
    return;
  // A and B must describe disjoint storage.  The non-alias ABI covers
  // distinct function arguments, but two views of one argument are still a
  // definite alias that would make direct TR reuse observe a changed input.
  if (getStorageRoot(matmul.getInputs()[0]) ==
      getStorageRoot(matmul.getInputs()[1]))
    return;

  if (!tilePlan || tilePlan->m <= 0 || tilePlan->n <= 0 || tilePlan->k <= 0)
    return;
  tilePlan->stagedAReadBytes =
      checkedAmeInputBytes(tilePlan->m, tilePlan->k,
                           target.ame.a.elementBits / 8)
          .value_or(0);
  tilePlan->stagedBReadBytes =
      checkedAmeInputBytes(tilePlan->k, tilePlan->n,
                           target.ame.b.elementBits / 8)
          .value_or(0);
  if (auto direct = checkedAmeDirectInputBytes(tilePlan->stagedAReadBytes,
                                               tilePlan->nTileCount))
    tilePlan->directAReadBytes = *direct;
  if (auto direct = checkedAmeDirectInputBytes(tilePlan->stagedBReadBytes,
                                               tilePlan->mTileCount))
    tilePlan->directBReadBytes = *direct;

  auto noteDirectCandidate = [&](unsigned inputIndex, uint64_t directBytes,
                                 uint64_t stagedBytes) {
    if (!directBytes || !stagedBytes ||
        !directAmeInputTilesLegal(matmul, tilePlan, target, inputIndex) ||
        !isSingleLogicalAmeInput(matmul, matmul.getInputs()[inputIndex],
                                 inputIndex, target.dma.endpointAlignmentBytes))
      return;
    if (inputIndex == 0)
      tilePlan->directAEligible = true;
    else
      tilePlan->directBEligible = true;
  };
  noteDirectCandidate(0, tilePlan->directAReadBytes,
                      tilePlan->stagedAReadBytes);
  noteDirectCandidate(1, tilePlan->directBReadBytes,
                      tilePlan->stagedBReadBytes);
  if (!config.enableSpmDma) {
    if (tilePlan->directAEligible) {
      matmul->setAttr(kDirectAOperandAttr, UnitAttr::get(matmul.getContext()));
      tilePlan->directABySpmDisabled = true;
    }
    if (tilePlan->directBEligible) {
      matmul->setAttr(kDirectBOperandAttr, UnitAttr::get(matmul.getContext()));
      tilePlan->directBBySpmDisabled = true;
    }
  }
}

/// The default direct-ddr policy must not silently turn an external AME C
/// output into an SPM/DMA round trip. A direct C tile is legal only when every
/// emitted N tile starts at a proven 64-byte address and its physical row
/// stride is hardware-compatible. Keep this check at the planner boundary so
/// dynamic descriptor offsets and non-cache-line tile widths fail before any
/// fallback workspace is allocated.
static LogicalResult verifyDirectAmeOutput(linalg::MatmulOp matmul,
                                           const LinalgToBOSCAMEConfig &config,
                                           const AmeMatmulTilePlan *tilePlan,
                                           const TargetInfo &target) {
  if (config.storePolicy != "direct-ddr" ||
      !matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme) ||
      matmul.getOutputs().size() != 1 ||
      !isSupportedAmeMatmul(matmul, target.ame))
    return success();
  Value output = matmul.getOutputs().front();
  if (auto type = dyn_cast<MemRefType>(output.getType());
      type && spmdma::getSpmBank(type))
    return success();
  if (!isExternalDdrBuffer(output))
    return success();
  auto cType = dyn_cast<MemRefType>(output.getType());
  auto cLayout = cType ? computeTileStorageLayout(cType, TileRole::Accumulator)
                       : std::optional<TileStorageLayout>();
  if (cLayout &&
      cLayout->rowStrideBytes % target.dma.endpointAlignmentBytes != 0)
    return matmul.emitError(
        "BOSCAME direct-DDR operand requires a 64-byte-aligned physical row "
        "stride");
  if (!directAmeOutputTilesLegal(matmul, tilePlan, target))
    return matmul.emitError(
        "direct-ddr AME C output cannot be tiled with 64-byte-aligned "
        "accumulator rows; choose store-policy=spm-dma for compatibility "
        "staging");
  if (!directAmeAddressLegal(output, target.dma.endpointAlignmentBytes)) {
    if (directAmeAddressKnownMisaligned(output,
                                        target.dma.endpointAlignmentBytes))
      return matmul.emitError(
          "BOSCAME direct-DDR operand requires a 64-byte-aligned descriptor "
          "offset");
    return matmul.emitError(
        "direct-ddr AME C output address must be provably 64-byte aligned; "
        "dynamic or misaligned per-tile offsets require store-policy=spm-dma");
  }
  return success();
}

// The immediate-wait schedule reuses fixed bank ranges and DMA channels.
// Only region operations with a known serial execution contract may enclose
// one of its consumers; unknown or concurrent regions need a different plan.
static LogicalResult verifySerialExecution(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<func::FuncOp>(parent))
      return success();
    if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ExecuteRegionOp,
             scf::IndexSwitchOp>(parent))
      return op->emitError("BOSCAME fixed SPM workspace requires serial "
                           "execution inside func.func; unsupported ancestor ")
             << parent->getName();
  }
  return op->emitError("BOSCAME buffer computations must be inside func.func");
}

struct DynamicPrefixSubview {
  Value base;
  Value extent;
};

static FailureOr<DynamicPrefixSubview>
getDynamicPrefixSubview(memref::CopyOp copy, Value view) {
  auto fail = [&](Twine message) -> FailureOr<DynamicPrefixSubview> {
    copy.emitError() << message;
    return failure();
  };
  auto subview = view.getDefiningOp<memref::SubViewOp>();
  if (!subview || subview.getMixedOffsets().size() != 1 ||
      subview.getMixedSizes().size() != 1 ||
      subview.getMixedStrides().size() != 1 ||
      getConstantIntValue(subview.getMixedOffsets().front()) != 0 ||
      getConstantIntValue(subview.getMixedStrides().front()) != 1)
    return fail("dynamic BOSCAME prefix requires a zero-offset, unit-stride "
                "rank-one subview");
  Value extent = dyn_cast<Value>(subview.getMixedSizes().front());
  if (!extent)
    return fail("dynamic BOSCAME prefix requires an SSA extent");
  auto resultType = dyn_cast<MemRefType>(view.getType());
  auto baseType = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!resultType || !baseType || resultType.getRank() != 1 ||
      !ShapedType::isDynamic(resultType.getDimSize(0)) ||
      !baseType.hasStaticShape() || baseType.getRank() != 1)
    return fail("dynamic BOSCAME prefix requires a static rank-one maximum "
                "buffer");
  return DynamicPrefixSubview{subview.getSource(), extent};
}

static bool
areDynamicPrefixEndpointsDisjoint(Value externalBase, Value peerBase,
                                  func::FuncOp function,
                                  const BufferAliasContext &aliasContext) {
  Value externalRoot = getStorageRoot(externalBase);
  Value peerRoot = getStorageRoot(peerBase);
  if (!externalRoot || !peerRoot || externalRoot == peerRoot)
    return false;

  if (spmdma::getSpmBank(peerBase.getType()))
    return true;
  return function &&
         getDistinctRootProof(externalRoot, peerRoot, function, aliasContext)
             .has_value();
}

static bool isClampedPrefixExtent(Value extent, int64_t maximum) {
  auto operandOtherThanConstant = [](Value lhs, Value rhs,
                                     int64_t constant) -> Value {
    if (getConstantIntValue(lhs) == constant)
      return rhs;
    if (getConstantIntValue(rhs) == constant)
      return lhs;
    return {};
  };

  std::function<bool(Value)> isClampedValue = [&](Value value) {
    if (getConstantIntValue(value) == 0)
      return true;

    if (auto select = value.getDefiningOp<arith::SelectOp>())
      return isClampedValue(select.getTrueValue()) &&
             isClampedValue(select.getFalseValue());

    if (auto lowerThenUpper = value.getDefiningOp<arith::MaxSIOp>()) {
      Value upper = operandOtherThanConstant(lowerThenUpper.getLhs(),
                                             lowerThenUpper.getRhs(), 0);
      auto upperClamp = upper ? upper.getDefiningOp<arith::MinSIOp>() : nullptr;
      return upperClamp &&
             operandOtherThanConstant(upperClamp.getLhs(), upperClamp.getRhs(),
                                      maximum);
    }
    if (auto upperThenLower = value.getDefiningOp<arith::MinSIOp>()) {
      Value lower = operandOtherThanConstant(upperThenLower.getLhs(),
                                             upperThenLower.getRhs(), maximum);
      auto lowerClamp = lower ? lower.getDefiningOp<arith::MaxSIOp>() : nullptr;
      return lowerClamp && operandOtherThanConstant(lowerClamp.getLhs(),
                                                    lowerClamp.getRhs(), 0);
    }
    return false;
  };

  return isClampedValue(extent);
}

static FailureOr<TransferPlan> analyzeDynamicPrefixCopy(
    memref::CopyOp copy, const LinalgToBOSCAMEConfig &config,
    const TargetInfo &targetInfo, const BufferAliasContext &aliasContext) {
  auto fail = [&](Twine message) -> FailureOr<TransferPlan> {
    copy.emitError() << message;
    return failure();
  };
  auto provenance = copy->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  auto kind = provenance ? provenance.getAs<StringAttr>("kind") : StringAttr();
  if (!kind || (kind.getValue() != "load" && kind.getValue() != "store"))
    return fail("dynamic BOSCAME copy requires load/store provenance");
  if (kind.getValue() == "store" && config.storePolicy != "spm-dma")
    return fail("dynamic prefix stores require store-policy=spm-dma");

  FailureOr<DynamicPrefixSubview> source =
      getDynamicPrefixSubview(copy, copy.getSource());
  if (failed(source))
    return failure();
  FailureOr<DynamicPrefixSubview> targetSubview =
      getDynamicPrefixSubview(copy, copy.getTarget());
  if (failed(targetSubview))
    return failure();
  if (source->extent != targetSubview->extent)
    return fail("dynamic BOSCAME prefix endpoints must use the same extent");

  auto sourceType = cast<MemRefType>(source->base.getType());
  auto targetType = cast<MemRefType>(targetSubview->base.getType());
  if (sourceType.getShape() != targetType.getShape() ||
      sourceType.getElementType() != targetType.getElementType())
    return fail("dynamic BOSCAME prefix maximum buffers must match");
  int64_t maximum = sourceType.getDimSize(0);
  auto elementBytes = getBOSCAMEElementBytes(sourceType.getElementType());
  if (!elementBytes || *elementBytes <= 0 ||
      *elementBytes > targetInfo.dma.packetGranularityBytes)
    return fail("dynamic BOSCAME prefix has an unsupported element type");
  int64_t packetElements =
      targetInfo.dma.packetGranularityBytes / *elementBytes;
  if (maximum > targetInfo.spmBankCapacityBytes / *elementBytes)
    return fail("BOSCAME SPM workspace capacity exceeded");
  if (maximum <= 0 || maximum % packetElements != 0 ||
      !isClampedPrefixExtent(source->extent, maximum) ||
      !spmdma::isProvablyMultipleOf(source->extent, packetElements))
    return fail("dynamic BOSCAME prefix extent must be clamped to the static "
                "maximum and provably divisible by a DMA packet");

  bool isLoad = kind.getValue() == "load";
  Value external = isLoad ? copy.getSource() : copy.getTarget();
  Value externalBase = isLoad ? source->base : targetSubview->base;
  Value peer = isLoad ? copy.getTarget() : copy.getSource();
  Value peerBase = isLoad ? targetSubview->base : source->base;
  SmallVector<int64_t> peerStrides;
  int64_t peerOffset;
  auto peerType = dyn_cast<MemRefType>(peerBase.getType());
  if (!peerType ||
      failed(peerType.getStridesAndOffset(peerStrides, peerOffset)) ||
      peerStrides.size() != 1 || peerStrides.front() != 1)
    return fail("dynamic BOSCAME prefix peer buffer must be contiguous");
  if (!isExternalDdrBuffer(external))
    return fail("dynamic BOSCAME prefix requires a distinct external DDR "
                "endpoint and local snapshot");
  auto function = copy->getParentOfType<func::FuncOp>();
  if (!areDynamicPrefixEndpointsDisjoint(externalBase, peerBase, function,
                                         aliasContext))
    return fail("dynamic BOSCAME prefix peer buffer may alias the external "
                "endpoint");

  TransferRole role = isLoad ? TransferRole::InputA : TransferRole::OutputC;
  auto analyzed = analyzeBOSCAMEFootprint(externalBase, role, copy, targetInfo);
  if (!analyzed)
    return failure();
  if (!isLoad && failed(spmdma::verifyEndpointPointerAlignment(
                     copy, peerBase, "dynamic prefix store source",
                     targetInfo.dma.endpointAlignmentBytes)))
    return failure();

  TransferPlan transfer = std::move(*analyzed);
  transfer.buffer = external;
  transfer.validElements = source->extent;
  transfer.prefixPeerBuffer = peer;
  transfer.guardedPrefix = true;
  transfer.prefixLoad = isLoad;
  transfer.packets.clear();
  for (int64_t element = 0; element < maximum; element += packetElements) {
    int64_t offsetBytes;
    if (llvm::MulOverflow(element, *elementBytes, offsetBytes))
      return fail("dynamic BOSCAME prefix packet offset overflows");
    transfer.packets.push_back(
        {offsetBytes, offsetBytes, targetInfo.dma.packetGranularityBytes});
  }
  transfer.read = true;
  transfer.write = !isLoad;
  transfer.firstConsumer = copy;
  transfer.lastConsumer = copy;
  transfer.consumerOperations.assign(1, copy);
  return transfer;
}

/// Return the endpoints of a provenance-preserving identity copy.  The
/// frontend emits `memref.copy` after bufferization, while small hand-written
/// tests use an identity linalg.generic.  Keep both forms behind one matcher
/// so direct accumulator fusion can reason about the memory event rather than
/// the particular bufferization spelling.
static bool matchIdentityCopy(Operation *operation, Value &source,
                              Value &target) {
  if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
    source = copy.getSource();
    target = copy.getTarget();
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    auto targetType = dyn_cast<MemRefType>(target.getType());
    return sourceType && targetType && sourceType.hasStaticShape() &&
           targetType.hasStaticShape() && sourceType.getRank() == 2 &&
           targetType.getRank() == 2 &&
           sourceType.getShape() == targetType.getShape() &&
           sourceType.getElementType() == targetType.getElementType();
  }

  auto generic = dyn_cast<linalg::GenericOp>(operation);
  if (!generic || !generic.hasPureBufferSemantics() ||
      generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
    return false;
  source = generic.getInputs().front();
  target = generic.getOutputs().front();
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto targetType = dyn_cast<MemRefType>(target.getType());
  if (!sourceType || !targetType || sourceType.getRank() != 2 ||
      targetType.getRank() != 2 ||
      sourceType.getShape() != targetType.getShape() ||
      sourceType.getElementType() != targetType.getElementType())
    return false;
  if (generic.getIndexingMapsArray().size() != 2 ||
      llvm::any_of(generic.getIndexingMapsArray(),
                   [](AffineMap map) { return !map.isIdentity(); }) ||
      !llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType iterator) {
                      return iterator == utils::IteratorType::parallel;
                    }))
    return false;
  Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 1 || body.getNumArguments() != 2)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  return yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == body.getArgument(0);
}

static bool matchI32Add(linalg::GenericOp candidate, Value expected,
                        Value &other, Value &destination) {
  if (!candidate.hasPureBufferSemantics() ||
      candidate.getInputs().size() != 2 || candidate.getOutputs().size() != 1)
    return false;
  Value lhs = candidate.getInputs()[0];
  Value rhs = candidate.getInputs()[1];
  if (lhs != expected && rhs != expected)
    return false;
  other = lhs == expected ? rhs : lhs;
  destination = candidate.getOutputs().front();
  auto expectedType = dyn_cast<MemRefType>(expected.getType());
  auto otherType = dyn_cast<MemRefType>(other.getType());
  auto destinationType = dyn_cast<MemRefType>(destination.getType());
  if (!expectedType || !otherType || !destinationType ||
      expectedType.getRank() != 2 || !expectedType.hasStaticShape() ||
      !otherType.hasStaticShape() || !destinationType.hasStaticShape() ||
      expectedType.getShape() != otherType.getShape() ||
      expectedType.getShape() != destinationType.getShape() ||
      expectedType.getElementType() != otherType.getElementType() ||
      expectedType.getElementType() != destinationType.getElementType() ||
      !expectedType.getElementType().isInteger(32))
    return false;
  if (candidate.getIndexingMapsArray().size() != 3 ||
      llvm::any_of(candidate.getIndexingMapsArray(),
                   [](AffineMap map) { return !map.isIdentity(); }) ||
      !llvm::all_of(candidate.getIteratorTypesArray(),
                    [](utils::IteratorType iterator) {
                      return iterator == utils::IteratorType::parallel;
                    }))
    return false;
  Block &body = candidate.getRegion().front();
  if (body.empty() || !isa<linalg::YieldOp>(body.back()) ||
      body.getOperations().size() != 2)
    return false;
  arith::AddIOp add;
  for (Operation &nested : body.without_terminator())
    if (auto addOp = dyn_cast<arith::AddIOp>(&nested))
      add = addOp;
  auto yield = cast<linalg::YieldOp>(body.back());
  if (!add || add.getType() != expectedType.getElementType() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != add.getResult())
    return false;
  return (add.getLhs() == body.getArgument(0) &&
          add.getRhs() == body.getArgument(1)) ||
         (add.getLhs() == body.getArgument(1) &&
          add.getRhs() == body.getArgument(0));
}

static bool isUnmaskedMemoryEvent(Operation *operation, StringRef kind,
                                  Value sourceOrTarget) {
  if (!operation || !sourceOrTarget)
    return false;
  auto memory = operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  auto actualKind = memory ? memory.getAs<StringAttr>("kind") : StringAttr();
  auto masked = memory ? memory.getAs<BoolAttr>("masked") : BoolAttr();
  auto logical = memory ? memory.getAs<DenseI64ArrayAttr>("logical_shape")
                        : DenseI64ArrayAttr();
  auto valid = memory ? memory.getAs<DenseI64ArrayAttr>("valid_shape")
                      : DenseI64ArrayAttr();
  auto type = dyn_cast<MemRefType>(sourceOrTarget.getType());
  return actualKind && actualKind.getValue() == kind && masked &&
         !masked.getValue() && logical && valid &&
         logical.asArrayRef() == valid.asArrayRef() && type &&
         type.hasStaticShape() && logical.asArrayRef() == type.getShape();
}

/// Remove dead compiler-owned buffer chains after direct fusion.  Buffer form
/// represents initialization as writes to an allocated memref, so simply
/// erasing the consuming matmul can leave a fill/copy island that the ordinary
/// planner would still stage.  This helper only drops writes to an internal
/// value and follows view/alloc definitions; it never touches an external
/// function argument.
static void eraseDeadBuffer(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || !isa<memref::AllocOp, memref::AllocaOp, memref::ReinterpretCastOp,
                   memref::CastOp, memref::SubViewOp>(def))
    return;
  SmallVector<Operation *> users(value.getUsers());
  for (Operation *user : users) {
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (copy.getTarget() == value)
        copy.erase();
      continue;
    }
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (fill.getOutputs().size() == 1 && fill.getOutputs().front() == value)
        fill.erase();
    }
  }
  if (!value.use_empty())
    return;
  Value source;
  if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def))
    source = cast.getSource();
  else if (auto cast = dyn_cast<memref::CastOp>(def))
    source = cast.getSource();
  else if (auto subview = dyn_cast<memref::SubViewOp>(def))
    source = subview.getSource();
  def->erase();
  if (source)
    eraseDeadBuffer(source);
}

/// Move a view and its pure arithmetic descriptor operands before a loop. The
/// frontend computes program-dependent C offsets immediately before the
/// post-loop snapshot, so moving only the reinterpret_cast would create an
/// SSA dominance violation when the matmul is retargeted to that view.
static bool moveViewBeforeLoop(Value view, scf::ForOp loop) {
  Operation *root = view.getDefiningOp();
  if (!root || root->getBlock() != loop->getBlock())
    return false;
  SmallVector<Operation *> order;
  llvm::SmallPtrSet<Operation *, 16> visited;
  auto collect = [&](auto &&self, Operation *operation) -> bool {
    if (!operation || operation->getBlock() != loop->getBlock())
      return false;
    if (operation->isBeforeInBlock(loop.getOperation()))
      return true;
    if (!visited.insert(operation).second)
      return true;
    if (!isa<arith::ConstantOp, arith::AddIOp, arith::MulIOp,
             arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
             memref::ReinterpretCastOp, memref::SubViewOp, memref::CastOp>(
            operation))
      return false;
    for (Value operand : operation->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (!self(self, def))
          return false;
    order.push_back(operation);
    return true;
  };
  if (!collect(collect, root))
    return false;
  // `order` is in dependency-first order. Moving it in that same order keeps
  // each defining value above the operation that consumes it.
  for (Operation *operation : order)
    operation->moveBefore(loop);
  return true;
}

/// A fusion temporary must be an isolated compiler-owned allocation. Check
/// every level of the view chain, since looking only at the selected view's
/// users would miss a consumer of another alias of the same allocation.
static bool isPrivateFusionBuffer(Value value,
                                  ArrayRef<Operation *> permittedUsers) {
  SmallVector<Operation *> allowed(permittedUsers);
  while (true) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || !type.hasStaticShape() || spmdma::getSpmBank(type) ||
        llvm::any_of(value.getUsers(), [&](Operation *user) {
          return !llvm::is_contained(allowed, user) &&
                 !isa<memref::DeallocOp>(user);
        }))
      return false;
    Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    if (isa<memref::AllocOp, memref::AllocaOp>(def))
      return true;
    Value source;
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def))
      source = cast.getSource();
    else if (auto cast = dyn_cast<memref::CastOp>(def))
      source = cast.getSource();
    else if (auto subview = dyn_cast<memref::SubViewOp>(def))
      source = subview.getSource();
    else
      return false;
    allowed.assign(1, def);
    value = source;
  }
}

/// Moving a memory event across this interval may cross only allocations
/// and operations without memory effects. Check block order before walking.
static bool hasHarmlessFusionGap(Operation *from, Operation *to) {
  if (from->getBlock() != to->getBlock() || !from->isBeforeInBlock(to))
    return false;
  for (Operation *between = from->getNextNode(); between != to;
       between = between->getNextNode())
    if (!isa<memref::AllocOp, memref::AllocaOp>(between) &&
        !isMemoryEffectFree(between))
      return false;
  return true;
}

/// Fuse the bufferized frontend form where the C snapshot is outside the K
/// loop.  Each marked matmul currently writes a zero-initialized temporary and
/// adds it into a loop-carried accumulation buffer; after the loop the
/// snapshot is added once and copied back to C.  Retargeting every matmul to C
/// makes the AME emitter perform the equivalent read/accumulate/write sequence
/// for each K iteration, so the temporary SPM accumulator and all provenance
/// copies disappear.
static bool fuseLoopedAmeAccumulator(linalg::MatmulOp matmul,
                                     const LinalgToBOSCAMEConfig &config,
                                     const DataflowPlan &plan) {
  auto loop = matmul->getParentOfType<scf::ForOp>();
  if (!loop || matmul->getBlock() != loop.getBody()) {
    return false;
  }

  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower) {
    return false;
  }
  if (llvm::any_of(loop.getBody()->getOperations(), [&](Operation &operation) {
        return &operation != matmul.getOperation() &&
               isa<scf::ForOp, scf::IfOp, scf::WhileOp, func::CallOp>(
                   &operation);
      }))
    return false;
  // The rewrite removes the single post-loop accumulation chain. If another
  // marked dot contributes to the same loop, retargeting only this one would
  // silently drop that contribution after the chain is erased.
  unsigned markedMatmuls = 0;
  for (Operation &operation : loop.getBody()->getOperations())
    if (auto candidate = dyn_cast<linalg::MatmulOp>(&operation))
      markedMatmuls +=
          candidate->hasAttr(::buddy::boscame::lowering::kLowerToAme);
  if (markedMatmuls != 1)
    return false;

  Value product = matmul.getOutputs().front();
  auto zeroCopy = dyn_cast_or_null<memref::CopyOp>(matmul->getPrevNode());
  if (!zeroCopy || zeroCopy.getTarget() != product ||
      zeroCopy->getNextNode() != matmul.getOperation()) {
    return false;
  }

  auto sourceType = dyn_cast<MemRefType>(zeroCopy.getSource().getType());
  Value zeroSource = zeroCopy.getSource();
  auto productType = dyn_cast<MemRefType>(product.getType());
  if (!sourceType || !productType || sourceType.getRank() != 2 ||
      productType.getRank() != 2 || !sourceType.hasStaticShape() ||
      !productType.hasStaticShape() ||
      sourceType.getShape() != productType.getShape() ||
      sourceType.getElementType() != productType.getElementType()) {
    return false;
  }

  // The template copied into every K iteration must have been filled with
  // zero. A fill after the loop or an additional write to the template would
  // make this rewrite order-dependent, so require a unique fill user and a
  // dominating fill in the enclosing block.
  linalg::FillOp zeroFill;
  for (Operation *user : zeroCopy.getSource().getUsers()) {
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (fill.getOutputs().size() == 1 &&
          fill.getOutputs().front() == zeroCopy.getSource()) {
        if (!matchPattern(fill.getInputs().front(), m_Zero()) || zeroFill)
          return false;
        zeroFill = fill;
      }
      continue;
    }
    if (user == zeroCopy.getOperation() || isa<memref::DeallocOp>(user))
      continue;
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (copy.getSource() == zeroSource &&
          !isExternalDdrBuffer(copy.getTarget()))
        continue;
    }
    // Another read is harmless only when it is itself a dead initialization
    // copy; conservatively reject all other users to preserve provenance.
    return false;
  }
  if (!zeroFill || zeroFill->getBlock() != loop->getBlock() ||
      !zeroFill->isBeforeInBlock(loop)) {
    return false;
  }

  linalg::GenericOp innerAdd;
  Value accumulation;
  for (Operation *user : product.getUsers()) {
    auto candidate = dyn_cast<linalg::GenericOp>(user);
    if (!candidate || candidate->getBlock() != loop.getBody() ||
        candidate->getPrevNode() != matmul.getOperation() || innerAdd)
      continue;
    Value other, destination;
    if (!matchI32Add(candidate, product, other, destination) ||
        other == product || destination != other)
      continue;
    innerAdd = candidate;
    accumulation = destination;
  }
  if (!innerAdd) {
    return false;
  }
  // Moving the C read before the loop and the write after it is valid only
  // when the loop body has no other observable memory operation.  The
  // matched zero copy and elementwise add are removed by this rewrite; all
  // remaining operations must be pure address arithmetic or view creation.
  for (Operation &operation : loop.getBody()->getOperations()) {
    if (&operation == matmul.getOperation() ||
        &operation == zeroCopy.getOperation() ||
        &operation == innerAdd.getOperation() ||
        operation.hasTrait<OpTrait::IsTerminator>() ||
        isa<memref::AllocOp, memref::AllocaOp>(&operation) ||
        isMemoryEffectFree(&operation))
      continue;
    return false;
  }
  // The zero template may be copied only into the per-iteration product and
  // the loop accumulator that this rewrite removes. Any other destination
  // could observe the template after its fill is erased.
  if (llvm::any_of(zeroSource.getUsers(), [&](Operation *user) {
        auto copy = dyn_cast<memref::CopyOp>(user);
        return copy && copy.getSource() == zeroSource &&
               copy.getTarget() != product && copy.getTarget() != accumulation;
      }))
    return false;

  // Identify the C snapshot load and the final store around the loop. The
  // event metadata is mandatory: without it an ordinary generic add/copy
  // must stay on the compatibility path.
  Operation *load = nullptr;
  Value snapshot;
  Operation *outerAdd = nullptr;

  // Find the unique identity i32 add that combines the loop accumulation with
  // a provenance-marked C snapshot. It must execute after the loop and feed
  // the external store immediately.
  Value cDestination;
  for (Operation *candidate : accumulation.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(candidate);
    if (!generic || generic->getBlock() == loop.getBody())
      continue;
    Value other, destination;
    if (!matchI32Add(generic, accumulation, other, destination))
      continue;
    if (destination != accumulation)
      continue;
    // The snapshot's load event is discovered from its defining users below;
    // reject values that have no such event.
    Operation *candidateLoad = nullptr;
    for (Operation *user : other.getUsers()) {
      Value source, target;
      if (!matchIdentityCopy(user, source, target) || target != other ||
          !isUnmaskedMemoryEvent(user, "load", source))
        continue;
      if (candidateLoad) {
        candidateLoad = nullptr;
        break;
      }
      candidateLoad = user;
    }
    if (!candidateLoad || outerAdd)
      continue;
    outerAdd = generic;
    load = candidateLoad;
    snapshot = other;
    cDestination = destination;
  }
  if (!outerAdd || !load || !snapshot) {
    return false;
  }

  Operation *store = nullptr;
  Value destination;
  for (Operation *user : cDestination.getUsers()) {
    Value source, target;
    if (!matchIdentityCopy(user, source, target) || source != cDestination ||
        !isUnmaskedMemoryEvent(user, "store", target))
      continue;
    auto memory = user->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
    auto external =
        memory ? memory.getAs<StringAttr>("destination") : StringAttr();
    if (!external || external.getValue() != "external_ddr" || store)
      continue;
    store = user;
    destination = target;
  }
  if (!store || cDestination == Value() ||
      load->getBlock() != loop->getBlock() ||
      outerAdd->getBlock() != loop->getBlock() ||
      store->getBlock() != loop->getBlock() || !loop->isBeforeInBlock(load) ||
      !load->isBeforeInBlock(outerAdd) || outerAdd->getNextNode() != store) {
    return false;
  }
  if (!hasHarmlessFusionGap(loop, load) ||
      !hasHarmlessFusionGap(load, outerAdd))
    return false;

  Value cSource;
  Value ignored;
  if (!matchIdentityCopy(load, cSource, ignored) ||
      !isExternalDdrBuffer(cSource) || !isExternalDdrBuffer(destination) ||
      cSource != destination ||
      failed(verifyDirectStoreMetadata(store, destination))) {
    return false;
  }
  // Direct C read/modify/write changes the point at which C is observed. If
  // an A/B operand aliases that storage, updating C during the K loop would
  // change a later matmul input; retain the conservative staged path.
  if (getStorageRoot(cSource) == getStorageRoot(matmul.getInputs()[0]) ||
      getStorageRoot(cSource) == getStorageRoot(matmul.getInputs()[1]))
    return false;
  auto productMemref = dyn_cast<MemRefType>(product.getType());
  auto destinationMemref = dyn_cast<MemRefType>(destination.getType());
  if (!productMemref || !destinationMemref ||
      productMemref.getShape() != destinationMemref.getShape() ||
      productMemref.getElementType() != destinationMemref.getElementType()) {
    return false;
  }
  if (!directAmeOutputTilesLegal(
          matmul, findAmeMatmulTilePlan(plan, matmul.getOperation()),
          plan.target) ||
      !directAmeAddressLegal(destination,
                             plan.target.dma.endpointAlignmentBytes))
    return false;

  // The loop accumulator must start at the same proven zero template.
  // Accept only one dominating initialization copy and no other writers.
  memref::CopyOp initialCopy;
  for (Operation *user : accumulation.getUsers()) {
    auto copy = dyn_cast<memref::CopyOp>(user);
    if (!copy || copy.getTarget() != accumulation)
      continue;
    if (initialCopy || copy.getSource() != zeroSource ||
        copy->hasAttr("bosc_ame.memory") ||
        copy->getBlock() != loop->getBlock() ||
        !zeroFill->isBeforeInBlock(copy) || !copy->isBeforeInBlock(loop))
      return false;
    initialCopy = copy;
  }
  if (!initialCopy || zeroCopy->hasAttr("bosc_ame.memory") ||
      !isPrivateFusionBuffer(product, {matmul, zeroCopy, innerAdd}) ||
      !isPrivateFusionBuffer(snapshot, {load, outerAdd}) ||
      !isPrivateFusionBuffer(accumulation,
                             {initialCopy, innerAdd, outerAdd, store}) ||
      !isPrivateFusionBuffer(zeroSource, {zeroFill, zeroCopy, initialCopy}))
    return false;

  // The external C view is often created after the K loop by
  // StructuredToMemref. Move that view before the loop before retargeting the
  // nested matmul, or the direct tile operation would violate SSA dominance.
  // Its dynamic offset is commonly computed immediately before the view, so
  // move the whole same-block defining chain together.  Moving only the view
  // leaves an offset operand below the loop and produces invalid SSA.
  if (!moveViewBeforeLoop(destination, loop))
    return false;

  matmul.getOutputsMutable()[0].set(destination);
  matmul->setAttr("bosc_ame.loop_accumulator",
                  UnitAttr::get(matmul.getContext()));
  zeroCopy.erase();
  innerAdd.erase();
  load->erase();
  outerAdd->erase();
  store->erase();
  eraseDeadBuffer(product);
  eraseDeadBuffer(snapshot);
  eraseDeadBuffer(accumulation);
  eraseDeadBuffer(zeroSource);
  return true;
}

/// Contract the matched straight-line tasks together with the IR rewrite.
/// C's incoming storage ID and its new contents need distinct logical IDs:
/// reusing the store's destination ID would make the matmul consume its own
/// result. Keep source-origin IDs and the surviving task ID unchanged.
static bool updateDirectAccumulatorComputeGraph(
    linalg::MatmulOp matmul, Operation *load, linalg::FillOp fill,
    linalg::GenericOp add, Operation *store, ArrayRef<Operation *> copies,
    Value destination) {
  SmallVector<Operation *> fused{load, fill, matmul, add, store};
  fused.append(copies.begin(), copies.end());
  bool hasMetadata = llvm::any_of(fused, [](Operation *operation) {
    return operation->hasAttr("bosc_ame.compute_id");
  });
  if (!hasMetadata)
    return true;

  auto id = matmul->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
  SmallVector<int64_t> inputs =
      getI64Array(matmul, "bosc_ame.compute_input_values");
  SmallVector<int64_t> cInputs =
      getI64Array(load, "bosc_ame.compute_input_values");
  SmallVector<int64_t> outputs =
      getI64Array(store, "bosc_ame.compute_output_values");
  // Partial annotations are allowed on incomplete graphs. Retain the staged
  // form when they cannot describe the fused operation's boundary values.
  if (!id || inputs.size() != 3 || cInputs.size() != 1 || outputs.size() != 1)
    return false;

  auto function = matmul->getParentOfType<func::FuncOp>();
  auto module = function->getParentOfType<ModuleOp>();
  MLIRContext *context = matmul.getContext();
  Builder builder(context);
  DenseSet<int64_t> fusedIds;
  SmallVector<int64_t> predecessors, axes, origins;
  bool barrier = false;
  for (Operation *operation : fused) {
    if (auto task =
            operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id"))
      fusedIds.insert(task.getInt());
    auto append = [&](StringRef name, SmallVector<int64_t> &values) {
      auto extra = getI64Array(operation, name);
      values.append(extra.begin(), extra.end());
    };
    append("bosc_ame.compute_predecessors", predecessors);
    append("bosc_ame.varying_program_axes", axes);
    append("bosc_ame.source_load_origin_ids", origins);
    if (auto attr = operation->getAttrOfType<BoolAttr>(
            "bosc_ame.compute_barrier_after"))
      barrier |= attr.getValue();
  }
  llvm::erase_if(predecessors, [&](int64_t predecessor) {
    return fusedIds.contains(predecessor);
  });
  auto normalize = [](SmallVector<int64_t> &values) {
    llvm::sort(values);
    values.erase(std::unique(values.begin(), values.end()), values.end());
  };
  auto setIds = [&](Operation *operation, StringRef name,
                    ArrayRef<int64_t> values) {
    operation->setAttr(name, builder.getDenseI64ArrayAttr(values));
  };
  normalize(predecessors);
  normalize(axes);
  normalize(origins);

  int64_t lastValueId = -1;
  module.walk([&](Operation *operation) {
    for (StringRef name :
         {"bosc_ame.logical_value_ids", "bosc_ame.compute_input_values",
          "bosc_ame.compute_output_values"})
      for (int64_t value : getI64Array(operation, name))
        lastValueId = std::max(lastValueId, value);
    if (auto func = dyn_cast<func::FuncOp>(operation))
      for (unsigned i = 0; i < func.getNumArguments(); ++i)
        if (auto value = func.getArgAttrOfType<IntegerAttr>(
                i, "bosc_ame.logical_value_id"))
          lastValueId = std::max(lastValueId, value.getInt());
  });
  if (lastValueId == std::numeric_limits<int64_t>::max())
    return false;
  int64_t resultId = lastValueId + 1;
  inputs.back() = cInputs.front();
  setIds(matmul, "bosc_ame.compute_input_values", inputs);
  setIds(matmul, "bosc_ame.compute_output_values", {resultId});
  setIds(matmul, "bosc_ame.logical_value_ids", {resultId});
  setIds(matmul, "bosc_ame.compute_predecessors", predecessors);
  setIds(matmul, "bosc_ame.varying_program_axes", axes);
  setIds(matmul, "bosc_ame.source_load_origin_ids", origins);
  matmul->setAttr("bosc_ame.compute_barrier_after",
                  builder.getBoolAttr(barrier));
  matmul->setAttr("bosc_ame.rematerializable", builder.getBoolAttr(false));
  auto invariant = getI64Array(matmul, "bosc_ame.invariant_program_axes");
  llvm::erase_if(invariant,
                 [&](int64_t axis) { return llvm::is_contained(axes, axis); });
  setIds(matmul, "bosc_ame.invariant_program_axes", invariant);

  SmallVector<Operation *> liveTasks;
  DenseMap<int64_t, DenseSet<int64_t>> retargetedConsumers;
  function.walk([&](Operation *operation) {
    if (!operation->hasAttr("bosc_ame.compute_id") ||
        (operation != matmul && llvm::is_contained(fused, operation)))
      return;
    liveTasks.push_back(operation);
    if (operation == matmul)
      return;
    auto dependencies = getI64Array(operation, "bosc_ame.compute_predecessors");
    auto storeId = store->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
    bool readsStoredValue =
        storeId && llvm::is_contained(dependencies, storeId.getInt());
    for (int64_t &predecessor : dependencies)
      if (fusedIds.contains(predecessor))
        predecessor = id.getInt();
    normalize(dependencies);
    setIds(operation, "bosc_ame.compute_predecessors", dependencies);
    // The destination's old ID still denotes C_before for earlier readers.
    // Only readers after the contracted chain see the new C contents.
    if (readsStoredValue && operation->getBlock() == matmul->getBlock() &&
        matmul->isBeforeInBlock(operation)) {
      auto values = getI64Array(operation, "bosc_ame.compute_input_values");
      auto reads = getComputeTaskReadValues(operation);
      for (auto [value, read] : llvm::zip(values, reads))
        if (read == destination) {
          int64_t consumer =
              operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id")
                  .getInt();
          retargetedConsumers[value].insert(consumer);
          value = resultId;
        }
      setIds(operation, "bosc_ame.compute_input_values", values);
    }
  });
  // Refresh reverse task references on surviving values, including function
  // arguments. Bufferization may preserve these on views instead of tasks.
  auto redirectUsers = [&](DenseI64ArrayAttr users, int64_t valueId) {
    SmallVector<int64_t> updated(users.asArrayRef());
    llvm::erase_if(updated, [&](int64_t user) {
      auto moved = retargetedConsumers.find(valueId);
      return moved != retargetedConsumers.end() && moved->second.contains(user);
    });
    for (int64_t &user : updated)
      if (fusedIds.contains(user))
        user = id.getInt();
    normalize(updated);
    return builder.getDenseI64ArrayAttr(updated);
  };
  function.walk([&](Operation *operation) {
    if (auto users = operation->getAttrOfType<ArrayAttr>(
            "bosc_ame.logical_value_users")) {
      SmallVector<Attribute> updated;
      auto values = getI64Array(operation, "bosc_ame.logical_value_ids");
      for (auto [index, entry] : llvm::enumerate(users))
        updated.push_back(
            redirectUsers(cast<DenseI64ArrayAttr>(entry),
                          index < values.size() ? values[index] : -1));
      operation->setAttr("bosc_ame.logical_value_users",
                         builder.getArrayAttr(updated));
    }
  });
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    if (auto users = function.getArgAttrOfType<DenseI64ArrayAttr>(
            i, "bosc_ame.logical_value_users")) {
      auto value = function.getArgAttrOfType<IntegerAttr>(
          i, "bosc_ame.logical_value_id");
      function.setArgAttr(i, "bosc_ame.logical_value_users",
                          redirectUsers(users, value ? value.getInt() : -1));
    }
  SmallVector<int64_t> resultUsers;
  DenseMap<int64_t, Operation *> liveById;
  for (Operation *operation : liveTasks) {
    int64_t taskId =
        operation->getAttrOfType<IntegerAttr>("bosc_ame.compute_id").getInt();
    liveById.try_emplace(taskId, operation);
    if (llvm::is_contained(
            getI64Array(operation, "bosc_ame.compute_input_values"), resultId))
      resultUsers.push_back(taskId);
  }
  normalize(resultUsers);
  matmul->setAttr(
      "bosc_ame.logical_value_users",
      builder.getArrayAttr({builder.getDenseI64ArrayAttr(resultUsers)}));
  llvm::sort(liveTasks, [](Operation *lhs, Operation *rhs) {
    return lhs->getAttrOfType<IntegerAttr>("bosc_ame.compute_id").getInt() <
           rhs->getAttrOfType<IntegerAttr>("bosc_ame.compute_id").getInt();
  });
  for (Operation *operation : liveTasks) {
    int64_t stage = 0;
    for (int64_t predecessor :
         getI64Array(operation, "bosc_ame.compute_predecessors")) {
      auto producer = liveById.find(predecessor);
      if (producer == liveById.end())
        continue;
      auto previousStage = producer->second->getAttrOfType<IntegerAttr>(
          "bosc_ame.compute_stage");
      auto previousBarrier = producer->second->getAttrOfType<BoolAttr>(
          "bosc_ame.compute_barrier_after");
      if (previousStage && previousBarrier)
        stage = std::max(stage, previousStage.getInt() +
                                    int64_t(previousBarrier.getValue()));
    }
    operation->setAttr("bosc_ame.compute_stage",
                       builder.getI64IntegerAttr(stage));
  }
  function->setAttr("bosc_ame.compute_graph_node_count",
                    builder.getI64IntegerAttr(liveTasks.size()));
  return true;
}

/// Fuse the frontend's canonical `previous = tl.load(C); product = dot(...);
/// tl.store(C, product + previous)` expansion. Provenance-bearing identity
/// copies and private buffer handoffs may use either memref.copy or Linalg.
/// A direct accumulator then reads C, performs the signed i8 reduction, and
/// writes C back without a C SPM window or DMA round trip.
static LogicalResult
fuseDirectAmeAccumulator(ModuleOp module, const LinalgToBOSCAMEConfig &config,
                         const DataflowPlan &plan) {
  SmallVector<linalg::MatmulOp> matmuls;
  module.walk([&](linalg::MatmulOp op) {
    if (op->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      matmuls.push_back(op);
  });

  for (linalg::MatmulOp matmul : matmuls) {
    if (fuseLoopedAmeAccumulator(matmul, config, plan))
      continue;
    Value product = matmul.getOutputs().front();
    linalg::FillOp zeroFill;
    for (Operation *user : product.getUsers())
      if (auto fill = dyn_cast<linalg::FillOp>(user))
        if (fill.getOutputs().size() == 1 &&
            fill.getOutputs().front() == product) {
          zeroFill = fill;
          break;
        }
    // Replacing the product's initial value by C is equivalent only for a
    // zero-initialized dot. Integer addition wraps modulo 2^32.
    if (!zeroFill || !matchPattern(zeroFill.getInputs().front(), m_Zero()))
      continue;

    linalg::GenericOp add;
    Value snapshot;
    for (Operation *user : product.getUsers()) {
      auto candidate = dyn_cast<linalg::GenericOp>(user);
      Value other, output;
      if (!candidate || !matchI32Add(candidate, product, other, output) ||
          other == product ||
          getStorageRoot(other) == getStorageRoot(product) || output != other)
        continue;
      add = candidate;
      snapshot = other;
      break;
    }
    if (!add)
      continue;

    Operation *load = nullptr;
    Value loadSource;
    for (Operation *user : snapshot.getUsers()) {
      Value source, target;
      if (!matchIdentityCopy(user, source, target) || target != snapshot ||
          !isUnmaskedMemoryEvent(user, "load", source))
        continue;
      if (load) {
        load = nullptr;
        break;
      }
      load = user;
      loadSource = source;
    }
    if (!load)
      continue;

    Operation *store = nullptr;
    Value destination;
    SmallVector<Operation *> storeChain;
    SmallVector<Value> storeChainTargets;
    Value cursor = snapshot;
    llvm::SmallPtrSet<Operation *, 8> visitedCopies;
    while (!store) {
      Operation *next = nullptr;
      Value nextTarget;
      bool ambiguous = false;
      for (Operation *user : cursor.getUsers()) {
        Value source, target;
        if (!matchIdentityCopy(user, source, target) || source != cursor)
          continue;
        if (next || !visitedCopies.insert(user).second) {
          ambiguous = true;
          break;
        }
        next = user;
        nextTarget = target;
      }
      if (ambiguous || !next)
        break;
      auto memory = next->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
      if (memory) {
        auto kind = memory.getAs<StringAttr>("kind");
        auto external = memory.getAs<StringAttr>("destination");
        if (kind && kind.getValue() == "store" && external &&
            external.getValue() == "external_ddr") {
          store = next;
          destination = nextTarget;
        }
        break;
      }
      // An unannotated copy is only a private compiler handoff. Ownership
      // and all users (including aliases) are checked below before erasing it.
      storeChain.push_back(next);
      storeChainTargets.push_back(nextTarget);
      cursor = nextTarget;
    }
    if (!store || !isExternalDdrBuffer(loadSource) ||
        !isExternalDdrBuffer(destination) || loadSource != destination)
      continue;

    // Retargeting moves C's read and write to the matmul. Keep the arithmetic
    // adjacent, and cross only allocations, pure descriptor arithmetic and
    // the matched copies. All copies must execute in dataflow order.
    if (load->getBlock() != matmul->getBlock() ||
        zeroFill->getNextNode() != matmul.getOperation() ||
        matmul->getNextNode() != add.getOperation())
      continue;
    if (!hasHarmlessFusionGap(load, zeroFill))
      continue;
    Operation *previous = add;
    bool ordered = true;
    for (Operation *copy : storeChain) {
      ordered &= hasHarmlessFusionGap(previous, copy);
      previous = copy;
    }
    if (!ordered || !hasHarmlessFusionGap(previous, store))
      continue;

    // Preserve direct-mode diagnostics for masked or malformed stores rather
    // than erasing their provenance and emitting a full-tile accumulator store.
    if (failed(verifyDirectStoreMetadata(store, destination)))
      return failure();
    auto productType = dyn_cast<MemRefType>(product.getType());
    auto destinationType = dyn_cast<MemRefType>(destination.getType());
    if (!productType || !destinationType ||
        productType.getShape() != destinationType.getShape() ||
        productType.getElementType() != destinationType.getElementType() ||
        !directAmeOutputTilesLegal(
            matmul, findAmeMatmulTilePlan(plan, matmul.getOperation()),
            plan.target) ||
        !directAmeAddressLegal(destination,
                               plan.target.dma.endpointAlignmentBytes))
      continue;
    // An early write to C must not change another tile's A/B input.
    if (getStorageRoot(destination) == getStorageRoot(matmul.getInputs()[0]) ||
        getStorageRoot(destination) == getStorageRoot(matmul.getInputs()[1]))
      continue;

    if (!isPrivateFusionBuffer(product, {matmul, zeroFill, add}) ||
        !isPrivateFusionBuffer(
            snapshot,
            {load, add, storeChain.empty() ? store : storeChain.front()}))
      continue;
    bool privateChain = true;
    for (size_t index = 0; index < storeChain.size(); ++index) {
      Operation *consumer =
          index + 1 < storeChain.size() ? storeChain[index + 1] : store;
      // A buffer-form copy uses its target as an operand too. Both the
      // producing copy and its unique consumer are legitimate users.
      privateChain &= isPrivateFusionBuffer(storeChainTargets[index],
                                            {storeChain[index], consumer});
    }
    if (!privateChain)
      continue;

    if (!updateDirectAccumulatorComputeGraph(matmul, load, zeroFill, add, store,
                                             storeChain, destination))
      continue;
    matmul.getOutputsMutable()[0].set(destination);
    store->erase();
    for (Operation *copy : llvm::reverse(storeChain))
      copy->erase();
    add.erase();
    load->erase();
    zeroFill.erase();
    for (Value target : llvm::reverse(storeChainTargets))
      eraseDeadBuffer(target);
    eraseDeadBuffer(snapshot);
    eraseDeadBuffer(product);
  }
  return success();
}

struct BankAllocation {
  SmallVector<int64_t, 4> banks;
  SmallVector<int64_t, 4> offsets;
  int64_t highWater[2] = {0, 0};
};

/// Exhaustively compare the two-bank assignments for a small operation-local
/// transfer set. The objective is resource-only: minimize peak bank high-water
/// and then total high-water. It does not estimate latency.
static std::optional<BankAllocation>
chooseBankAllocation(ArrayRef<TransferPlan> transfers,
                     ArrayRef<int64_t> reservedBytes, int64_t bankCapacity,
                     int64_t placementAlignmentBytes,
                     ResourceEstimate &resources) {
  constexpr size_t kMaxExactTransferCount = 12;
  ++resources.bankAllocationGroups;

  auto placeMask = [&](uint64_t mask) -> std::optional<BankAllocation> {
    BankAllocation candidate;
    int64_t cursor[2] = {reservedBytes[0], reservedBytes[1]};
    candidate.banks.reserve(transfers.size());
    candidate.offsets.reserve(transfers.size());
    for (size_t i = 0; i < transfers.size(); ++i) {
      int64_t bank = (mask >> i) & 1;
      int64_t capacity = transfers[i].spmCapacity;
      if (capacity <= 0 || capacity > bankCapacity)
        return std::nullopt;
      int64_t offset = llvm::alignTo(cursor[bank], placementAlignmentBytes);
      if (offset > bankCapacity - capacity)
        return std::nullopt;
      candidate.banks.push_back(bank);
      candidate.offsets.push_back(offset);
      cursor[bank] = offset + capacity;
    }
    candidate.highWater[0] = cursor[0];
    candidate.highWater[1] = cursor[1];
    return candidate;
  };

  auto placeGreedy = [&]() -> std::optional<BankAllocation> {
    BankAllocation allocation;
    int64_t used[2] = {reservedBytes[0], reservedBytes[1]};
    for (const TransferPlan &transfer : transfers) {
      int64_t bank = used[0] <= used[1] ? 0 : 1;
      int64_t offset = llvm::alignTo(used[bank], placementAlignmentBytes);
      if (transfer.spmCapacity <= 0 || transfer.spmCapacity > bankCapacity ||
          offset > bankCapacity - transfer.spmCapacity) {
        bank = 1 - bank;
        offset = llvm::alignTo(used[bank], placementAlignmentBytes);
      }
      if (transfer.spmCapacity <= 0 || transfer.spmCapacity > bankCapacity ||
          offset > bankCapacity - transfer.spmCapacity)
        return std::nullopt;
      allocation.banks.push_back(bank);
      allocation.offsets.push_back(offset);
      used[bank] = offset + transfer.spmCapacity;
    }
    allocation.highWater[0] = used[0];
    allocation.highWater[1] = used[1];
    return allocation;
  };
  std::optional<BankAllocation> greedy = placeGreedy();

  if (transfers.size() <= kMaxExactTransferCount) {
    uint64_t candidateCount = uint64_t{1} << transfers.size();
    std::optional<BankAllocation> best;
    for (uint64_t mask = 0; mask < candidateCount; ++mask) {
      ++resources.bankAllocationCandidates;
      auto candidate = placeMask(mask);
      if (!candidate) {
        ++resources.bankAllocationRejectedCapacity;
        continue;
      }
      auto score = [](const BankAllocation &allocation) {
        int64_t peak =
            std::max(allocation.highWater[0], allocation.highWater[1]);
        return std::pair<int64_t, int64_t>{peak, allocation.highWater[0] +
                                                     allocation.highWater[1]};
      };
      bool preferCandidate = !best || score(*candidate) < score(*best);
      if (best && score(*candidate) == score(*best)) {
        auto distanceFromGreedy = [&](const BankAllocation &allocation) {
          if (!greedy)
            return size_t(0);
          size_t distance = 0;
          for (size_t i = 0; i < transfers.size(); ++i)
            distance += allocation.banks[i] != greedy->banks[i];
          return distance;
        };
        size_t candidateDistance = distanceFromGreedy(*candidate);
        size_t bestDistance = distanceFromGreedy(*best);
        if (candidateDistance != bestDistance)
          preferCandidate = candidateDistance < bestDistance;
        for (size_t i = 0; i < transfers.size(); ++i) {
          if (candidateDistance != bestDistance)
            break;
          if (candidate->banks[i] == best->banks[i])
            continue;
          preferCandidate = candidate->banks[i] < best->banks[i];
          break;
        }
      }
      if (preferCandidate) {
        best = std::move(candidate);
      }
    }
    return best;
  }

  ++resources.bankAllocationBoundedGroups;
  if (greedy)
    return greedy;

  // The sorted greedy placement can miss a feasible split across two banks.
  // Search a bounded number of alternatives only after greedy has failed, so
  // ordinary large transfer groups retain their existing fast path.
  constexpr uint64_t kMaxFallbackSearchNodes = 65536;
  if (transfers.size() > 64)
    return std::nullopt;
  ++resources.bankAllocationRepairGroups;
  SmallVector<int64_t, 16> banks(transfers.size(), 0);
  SmallVector<int64_t, 16> offsets(transfers.size(), 0);
  int64_t used[2] = {reservedBytes[0], reservedBytes[1]};
  uint64_t visited = 0;
  std::function<bool(size_t)> search = [&](size_t index) {
    if (index == transfers.size())
      return true;
    if (visited >= kMaxFallbackSearchNodes)
      return false;
    ++visited;
    ++resources.bankAllocationRepairSearchNodes;

    int64_t preferred = used[0] <= used[1] ? 0 : 1;
    for (int64_t bank : {preferred, 1 - preferred}) {
      int64_t capacity = transfers[index].spmCapacity;
      int64_t previous = used[bank];
      int64_t offset = llvm::alignTo(used[bank], placementAlignmentBytes);
      if (capacity <= 0 || capacity > bankCapacity ||
          offset > bankCapacity - capacity) {
        ++resources.bankAllocationRepairRejectedPlacements;
        continue;
      }
      banks[index] = bank;
      offsets[index] = offset;
      used[bank] = offset + capacity;
      if (search(index + 1))
        return true;
      used[bank] = previous;
    }
    return false;
  };
  if (!search(0))
    return std::nullopt;

  BankAllocation repaired;
  repaired.banks = std::move(banks);
  repaired.offsets = std::move(offsets);
  repaired.highWater[0] = used[0];
  repaired.highWater[1] = used[1];
  return repaired;
}

static bool getPacketReadCost(const TransferPlan &transfer, uint64_t &readBytes,
                              uint64_t &packetCount) {
  for (const DmaPacket &packet : transfer.packets) {
    if (packet.bytes <= 0 ||
        static_cast<uint64_t>(packet.bytes) >
            std::numeric_limits<uint64_t>::max() - readBytes ||
        packetCount == std::numeric_limits<uint64_t>::max())
      return false;
    readBytes += static_cast<uint64_t>(packet.bytes);
    ++packetCount;
  }
  return true;
}

static bool hasLowerReadCost(uint64_t lhsBytes, uint64_t lhsPackets,
                             uint64_t rhsBytes, uint64_t rhsPackets) {
  return lhsBytes < rhsBytes ||
         (lhsBytes == rhsBytes && lhsPackets < rhsPackets);
}

static int64_t getStaticTransferCapacity(Value value, int64_t bankCapacity) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() == 0 || !type.hasStaticShape())
    return bankCapacity + 1;
  auto elementBytes = getBOSCAMEElementBytes(type.getElementType());
  if (!elementBytes)
    return bankCapacity + 1;
  int64_t elements = 1;
  for (int64_t dimension : type.getShape())
    if (dimension <= 0 || llvm::MulOverflow(elements, dimension, elements))
      return bankCapacity + 1;
  int64_t capacity = 0;
  if (llvm::MulOverflow(elements, *elementBytes, capacity))
    return bankCapacity + 1;
  return capacity;
}

/// Collect physical roots separately from the sequence of writes into them.
/// The existing access analysis supplies alias roots, layouts and effects.
static void collectBufferPlacementFacts(ModuleOp module,
                                        const KernelFacts &facts,
                                        DataflowPlan &plan) {
  DenseMap<Operation *, unsigned> positions;
  module.walk([&](func::FuncOp function) {
    unsigned position = 0;
    function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
      if (operation != function.getOperation())
        positions[operation] = position++;
    });
  });
  module.walk([&](Operation *allocation) {
    if (!isa<memref::AllocOp, memref::AllocaOp>(allocation))
      return;
    Value root = allocation->getResult(0);
    auto type = cast<MemRefType>(root.getType());
    if (!type.hasStaticShape() || !type.getRank() || spmdma::getSpmBank(type))
      return;
    RootStorageFact storage;
    storage.root = root;
    storage.views.push_back(root);
    storage.shape.assign(type.getShape().begin(), type.getShape().end());
    int64_t ignored;
    (void)type.getStridesAndOffset(storage.strides, ignored);
    storage.capacityBytes = getStaticTransferCapacity(root, plan.bankCapacity);
    if (type.getElementType().isInteger(1))
      storage.capacityBytes = type.getNumElements();
    storage.firstPosition = std::numeric_limits<unsigned>::max();
    storage.aliasProven = true; // A non-escaping allocation is fresh storage.
    auto function = allocation->getParentOfType<func::FuncOp>();
    storage.structuredScope =
        !function || allocation->getParentRegion() != &function.getBody();
    for (size_t i = 0; i < storage.views.size(); ++i)
      for (Operation *user : storage.views[i].getUsers()) {
        if (isStorageViewOperation(user)) {
          for (Value result : user->getResults())
            if (isa<MemRefType>(result.getType()) &&
                !llvm::is_contained(storage.views, result))
              storage.views.push_back(result);
        } else if (!isa<linalg::LinalgOp, memref::CopyOp, memref::DeallocOp,
                        memref::DimOp>(user)) {
          storage.aliasProven = false;
        }
        StringRef name = user->getName().getStringRef();
        storage.feedsAmeDirectRoute |= user->hasAttr(kDirectAOperandAttr) ||
                                       user->hasAttr(kDirectBOperandAttr) ||
                                       user->hasAttr(kDirectCOperandAttr) ||
                                       user->hasAttr("bosc_ame.direct_ddr") ||
                                       name.starts_with("bosc_ame.tile.");
      }
    size_t rootIndex = plan.rootStorageFacts.size();
    DenseMap<Operation *, DenseSet<Value>> stagedReads;
    for (auto [index, access] : llvm::enumerate(facts.bufferAccesses)) {
      if (access.storageRoot != root || !access.operation ||
          (access.effect != BufferAccessEffect::Read &&
           access.effect != BufferAccessEffect::Write))
        continue;
      auto position = positions.find(access.operation);
      if (position == positions.end()) {
        storage.aliasProven = false;
        continue;
      }
      storage.structuredScope |=
          !function ||
          access.operation->getParentRegion() != &function.getBody();
      storage.firstPosition = std::min(storage.firstPosition, position->second);
      storage.lastPosition = std::max(storage.lastPosition, position->second);
      RootStorageAccess use{access.operation, position->second, index};
      if (access.effect == BufferAccessEffect::Read) {
        storage.readers.push_back(use);
        if (access.logicalFootprintBytes && *access.logicalFootprintBytes > 0 &&
            stagedReads[access.operation].insert(access.accessedValue).second) {
          uint64_t bytes = *access.logicalFootprintBytes;
          if (bytes <= std::numeric_limits<uint64_t>::max() -
                           storage.potentialReadSavingsBytes)
            storage.potentialReadSavingsBytes += bytes;
        }
        continue;
      }
      storage.writers.push_back(use);
      ContentVersion version;
      version.rootIndex = rootIndex;
      version.writer = access.operation;
      version.position = position->second;
      version.writtenView = access.accessedValue;
      version.byteOffset = access.effectiveByteOffsetBytes;
      version.bytes = access.physicalSpanBytes;
      version.validShape =
          access.validShape.empty() ? access.shape : access.validShape;
      if (auto attr = access.operation->getAttrOfType<BoolAttr>(
              "bosc_ame.rematerializable"))
        version.rematerializable = attr.getValue();
      version.paddingInitialization =
          access.operation->hasAttr("bosc_ame.padding");
      plan.contentVersions.push_back(std::move(version));
    }
    // A read may depend on both a padding fill and a later prefix write.
    // Stop only when a newer write completely covers the read's byte range.
    for (const RootStorageAccess &reader : storage.readers) {
      // Lexical order cannot identify a reaching content version through a
      // branch or loop. Fixed root reservations need no such version reuse.
      if (storage.structuredScope)
        break;
      const BufferAccessFact &access = facts.bufferAccesses[reader.accessIndex];
      auto readRange =
          getCheckedByteInterval(access.effectiveByteOffsetBytes,
                                 access.physicalSpanBytes.value_or(-1));
      for (ContentVersion &version : llvm::reverse(plan.contentVersions)) {
        if (version.rootIndex != rootIndex ||
            version.position >= reader.position)
          continue;
        auto writeRange = getCheckedByteInterval(version.byteOffset,
                                                 version.bytes.value_or(-1));
        if (readRange && writeRange &&
            areByteIntervalsDisjoint(*readRange, *writeRange))
          continue;
        if (!llvm::is_contained(version.consumers, reader.operation))
          version.consumers.push_back(reader.operation);
        if (readRange && writeRange && writeRange->first <= readRange->first &&
            writeRange->second >= readRange->second)
          break;
      }
    }
    plan.rootStorageFacts.push_back(std::move(storage));
  });
}

/// Validate a complete prospective module budget. No lifetime reuse is used
/// for resident roots. Operation-local staging reuses its ordinary immediate-
/// wait workspace, always above every explicit and resident reservation.
static LogicalResult
planPlacementStaging(ModuleOp module, const LinalgToBOSCAMEConfig &config,
                     const DataflowPlan &trial, ArrayRef<int64_t> reserved,
                     SmallVectorImpl<TransferPlan> &staging,
                     std::string &reason) {
  DenseSet<Value> residentRoots;
  for (const BufferPlacementDecision &decision : trial.bufferPlacements)
    if (decision.kind == BufferPlacementKind::SpmResident)
      residentRoots.insert(trial.rootStorageFacts[decision.rootIndex].root);
  LogicalResult status = success();
  ScopedDiagnosticHandler diagnostics(module.getContext(),
                                      [&](Diagnostic &diagnostic) {
                                        llvm::raw_string_ostream stream(reason);
                                        diagnostic.print(stream);
                                        return success();
                                      });
  module.walk([&](linalg::LinalgOp op) {
    if (failed(status) || !isStageableBufferLinalg(op))
      return;
    auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation());
    if (matmul && !op->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      return;
    SmallVector<TransferPlan, 4> local;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.getRank() || spmdma::getSpmBank(type) ||
          residentRoots.contains(getStorageRoot(operand)) ||
          isPredicateOnlyInput(op, index) || isPredicateOnlyOutput(op, index))
        continue;
      bool write = index >= static_cast<size_t>(op.getNumDpsInputs());
      if (matmul &&
          ((!write && index == 0 && op->hasAttr(kDirectAOperandAttr)) ||
           (!write && index == 1 && op->hasAttr(kDirectBOperandAttr)) ||
           (write && config.storePolicy == "direct-ddr" &&
            op->hasAttr(kDirectCOperandAttr))))
        continue;
      if (write && isDirectExternalStore(op, config))
        continue;
      if (llvm::any_of(local, [&](const TransferPlan &transfer) {
            return transfer.buffer == operand;
          }))
        continue;
      auto transfer = analyzeBOSCAMEFootprint(operand, TransferRole::Temporary,
                                              op, trial.target);
      if (!transfer) {
        status = failure();
        if (reason.empty())
          reason = "missing legal staging for non-scalar operand " +
                   std::to_string(index);
        return;
      }
      transfer->firstConsumer = op;
      transfer->lastConsumer = op;
      transfer->consumerOperations.push_back(op);
      local.push_back(std::move(*transfer));
    }
    ResourceEstimate ignored;
    auto allocation =
        chooseBankAllocation(local, reserved, trial.bankCapacity,
                             trial.target.dma.endpointAlignmentBytes, ignored);
    if (!allocation) {
      reason = "resident + explicit windows leave insufficient per-op staging "
               "capacity for " +
               op->getName().getStringRef().str();
      status = failure();
      return;
    }
    for (auto [index, transfer] : llvm::enumerate(local)) {
      transfer.spmBank = allocation->banks[index];
      transfer.spmOffset = allocation->offsets[index];
      transfer.channel = transfer.spmBank;
      staging.push_back(std::move(transfer));
    }
  });
  return status;
}

static LogicalResult planBufferPlacements(ModuleOp module,
                                          const LinalgToBOSCAMEConfig &config,
                                          DataflowPlan &plan,
                                          ArrayRef<int64_t> explicitReserved) {
  DenseMap<Value, size_t> roots;
  SmallVector<size_t> parents;
  for (auto [index, storage] : llvm::enumerate(plan.rootStorageFacts)) {
    roots[storage.root] = index;
    parents.push_back(index);
  }
  auto representative = [&](size_t index) {
    while (parents[index] != index)
      index = parents[index];
    return index;
  };
  DenseMap<Value, size_t> connected;
  module.walk([&](linalg::LinalgOp op) {
    if (op->hasAttr(kPrefixRangeAttr) || op->hasAttr(kPrefixMaskAuxAttr))
      return;
    SmallVector<Value> buffers;
    std::optional<size_t> first;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.getRank() || isPredicateOnlyInput(op, index) ||
          isPredicateOnlyOutput(op, index))
        continue;
      Value root = getStorageRoot(operand);
      buffers.push_back(root);
      auto found = roots.find(root);
      if (found == roots.end())
        continue;
      if (!first)
        first = found->second;
      else
        parents[representative(found->second)] = representative(*first);
    }
    if (!first)
      return;
    for (Value buffer : buffers) {
      auto [entry, inserted] = connected.try_emplace(buffer, *first);
      if (!inserted)
        parents[representative(entry->second)] = representative(*first);
    }
  });
  DenseMap<size_t, int64_t> groupIds;
  bool hasCalls = false;
  module.walk([&](CallOpInterface) { hasCalls = true; });
  for (auto [index, storage] : llvm::enumerate(plan.rootStorageFacts)) {
    BufferPlacementDecision decision;
    decision.rootIndex = index;
    bool auxiliary = true;
    bool used = false;
    for (Value view : storage.views)
      for (Operation *user : view.getUsers())
        if (auto op = dyn_cast<linalg::LinalgOp>(user)) {
          used = true;
          bool predicateUse = true;
          for (OpOperand &operand : op->getOpOperands())
            if (operand.get() == view &&
                !isPredicateOnlyInput(op, operand.getOperandNumber()) &&
                !isPredicateOnlyOutput(op, operand.getOperandNumber()))
              predicateUse = false;
          if (!op->hasAttr(kPrefixRangeAttr) &&
              !op->hasAttr(kPrefixMaskAuxAttr) && !predicateUse)
            auxiliary = false;
        }
    if (storage.feedsAmeDirectRoute) {
      decision.reason = BufferPlacementReason::AmeDirectRoute;
    } else if (!used) {
      decision.reason = BufferPlacementReason::Unused;
    } else if (auxiliary) {
      decision.reason = BufferPlacementReason::Auxiliary;
    } else {
      auto [entry, inserted] =
          groupIds.try_emplace(representative(index), groupIds.size());
      (void)inserted;
      decision.group = entry->second;
      decision.kind = BufferPlacementKind::SpmResident;
      decision.reason = BufferPlacementReason::Selected;
    }
    plan.bufferPlacements.push_back(std::move(decision));
  }
  int64_t reserved[2] = {explicitReserved[0], explicitReserved[1]};
  for (int64_t group = 0; group < static_cast<int64_t>(groupIds.size());
       ++group) {
    DataflowPlan trial = plan;
    for (BufferPlacementDecision &decision : trial.bufferPlacements)
      if (decision.group > group &&
          decision.kind == BufferPlacementKind::SpmResident)
        decision.kind = BufferPlacementKind::Spare;
    BufferPlacementReason rejection = BufferPlacementReason::Selected;
    std::string detail;
    int64_t proposed[2] = {reserved[0], reserved[1]};
    for (BufferPlacementDecision &decision : trial.bufferPlacements) {
      if (decision.group != group)
        continue;
      const RootStorageFact &storage =
          trial.rootStorageFacts[decision.rootIndex];
      auto type = cast<MemRefType>(storage.root.getType());
      bool serialScope =
          !hasCalls && hasBOSCAMEPlacementScope(storage.root.getDefiningOp());
      for (Value view : storage.views)
        for (Operation *user : view.getUsers())
          serialScope &= hasBOSCAMEPlacementScope(user);
      if (!serialScope) {
        rejection = BufferPlacementReason::ControlFlow;
        detail = "root placement requires serial single-block if/for/"
                 "execute_region scopes without calls";
        break;
      }
      if (!storage.aliasProven) {
        rejection = BufferPlacementReason::UnsupportedUse;
        detail = "root escapes or has a non-Linalg memory access";
        break;
      }
      auto capacity = spmdma::getContiguousZeroOffsetByteSize(type);
      if (!capacity || *capacity <= 0 || *capacity % 64 || type.getRank() > 2) {
        rejection = BufferPlacementReason::UnsupportedLayout;
        detail = "root needs a static contiguous rank-1/rank-2 aligned vector "
                 "layout";
        break;
      }
      // Keep each root's window for its whole scope. Only opposite arms of
      // an if may overlap; iterations and sequential scopes never do. This
      // needs neither a complete task DAG nor dynamic content-version facts.
      auto firstOffset = [&](int64_t bank) -> std::optional<int64_t> {
        int64_t offset = llvm::alignTo(explicitReserved[bank], 64);
        while (*capacity <= trial.bankCapacity &&
               offset <= trial.bankCapacity - *capacity) {
          bool conflict = false;
          for (const BufferPlacementDecision &other : trial.bufferPlacements) {
            if (other.rootIndex == decision.rootIndex ||
                other.kind != BufferPlacementKind::SpmResident ||
                other.bank != bank)
              continue;
            const RootStorageFact &occupied =
                trial.rootStorageFacts[other.rootIndex];
            if (haveExclusiveBOSCAMEPlacementScopes(
                    storage.root.getDefiningOp(),
                    occupied.root.getDefiningOp()))
              continue;
            int64_t end = other.offset + occupied.capacityBytes;
            if (offset < end && other.offset < offset + *capacity) {
              offset = llvm::alignTo(end, 64);
              conflict = true;
              break;
            }
          }
          if (!conflict)
            return offset;
        }
        return std::nullopt;
      };
      auto first = firstOffset(0), second = firstOffset(1);
      if (!first && !second) {
        rejection = BufferPlacementReason::Capacity;
        detail = "resident + explicit windows exceed a bank capacity";
        break;
      }
      int64_t bank = !second || (first && *first <= *second) ? 0 : 1;
      int64_t offset = bank == 0 ? *first : *second;
      decision.bank = bank;
      decision.offset = offset;
      proposed[bank] = std::max(proposed[bank], offset + *capacity);
    }
    SmallVector<TransferPlan, 4> staging;
    if (rejection == BufferPlacementReason::Selected &&
        failed(planPlacementStaging(module, config, trial,
                                    ArrayRef<int64_t>(proposed, 2), staging,
                                    detail)))
      rejection = BufferPlacementReason::MissingStaging;
    if (rejection == BufferPlacementReason::Selected &&
        failed(preflightBufferPlacements(module, trial, staging, detail)))
      rejection = BufferPlacementReason::ComputationLegality;
    for (auto [index, decision] : llvm::enumerate(plan.bufferPlacements)) {
      if (decision.group != group)
        continue;
      if (rejection == BufferPlacementReason::Selected) {
        decision = trial.bufferPlacements[index];
      } else {
        decision.kind = BufferPlacementKind::StagedFallback;
        decision.reason = rejection;
        decision.detail = detail;
        decision.bank = -1;
      }
    }
    if (rejection == BufferPlacementReason::Selected) {
      reserved[0] = proposed[0];
      reserved[1] = proposed[1];
    }
  }
  if (!config.enableSpmDma)
    for (BufferPlacementDecision &decision : plan.bufferPlacements)
      if (decision.kind == BufferPlacementKind::SpmResident) {
        decision.kind = BufferPlacementKind::StagedFallback;
        decision.reason = BufferPlacementReason::MissingStaging;
        decision.detail = "buffer placement requires enable-spm-dma=true";
      }
  return success();
}

FailureOr<DataflowPlan> BOSCAMEPlanner::plan(ModuleOp module) const {
  auto fail = [&](Twine message) -> FailureOr<DataflowPlan> {
    module.emitError() << message;
    return failure();
  };
  if (config.bm <= 0 || config.bn <= 0 || config.bk <= 0)
    return fail("BOSCAME tile sizes must be positive");
  auto target = resolveTargetInfo(config.target);
  if (!target)
    return fail("unsupported BOSCAME target; expected nr-fpga");
  if (config.storePolicy != "direct-ddr" && config.storePolicy != "spm-dma")
    return fail("invalid BOSCAME store policy; expected direct-ddr or "
                "spm-dma");
  if (config.storePolicy == "spm-dma" && !config.enableSpmDma)
    return fail("BOSCAME spm-dma store policy requires enable-spm-dma=true");
  if (config.enableSpmDma && config.dmaAbi != spmdma::kDmaAbiAlignedNonAliasing)
    return fail("SPM/DMA lowering requires dma-abi=args_aligned64_nonalias");
  if (failed(verifyCompleteComputeGraphs(module)))
    return failure();

  DataflowPlan workspace;
  workspace.target = *target;
  workspace.bankCapacity = target->spmBankCapacityBytes;
  auto launches = analyzeBOSCAMELaunchFacts(module);
  if (failed(launches))
    return failure();
  workspace.launches = std::move(*launches);
  int64_t tileM = std::min<int64_t>(target->ame.maxTileM, config.bm);
  int64_t tileN = std::min<int64_t>(target->ame.maxTileN, config.bn);
  int64_t tileK = std::min<int64_t>(target->ame.maxTileK, config.bk);
  module.walk([&](linalg::MatmulOp matmul) {
    if (matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      appendAmeMatmulTilePlan(matmul, tileM, tileN, tileK,
                              workspace.ameMatmulTiles);
  });
  // Retarget any fused accumulator store before making the direct-placement
  // ABI decision.  The marker is the planner's proof that an AME operand is
  // actually emitted as a direct DDR tile endpoint; an external C buffer that
  // fails that proof follows the ordinary SPM/DMA path.
  collectBufferSourceEvents(module, workspace);
  if (config.storePolicy == "direct-ddr" &&
      failed(fuseDirectAmeAccumulator(module, config, workspace)))
    return failure();
  if (failed(verifyCompleteComputeGraphs(module)))
    return failure();
  module.walk([&](linalg::MatmulOp matmul) {
    markLogicalAmeInputs(
        matmul, config, findAmeMatmulTilePlan(workspace, matmul.getOperation()),
        workspace.target);
  });
  LogicalResult directOutputStatus = success();
  module.walk([&](linalg::MatmulOp matmul) {
    if (succeeded(directOutputStatus))
      directOutputStatus = verifyDirectAmeOutput(
          matmul, config,
          findAmeMatmulTilePlan(workspace, matmul.getOperation()),
          workspace.target);
  });
  if (failed(directOutputStatus))
    return failure();

  // Direct stores use the same caller promise as DMA endpoints: a function
  // argument is only known to be 64-byte aligned and disjoint when the module
  // carries the explicit ABI contract.  Check this before any direct rewrite
  // can erase the provenance-bearing copy.
  {
    bool needsDirectAbi = false;
    module.walk([&](Operation *operation) {
      if (operation->hasAttr("bosc_ame.direct_ddr"))
        needsDirectAbi = true;
      if (auto linalg = dyn_cast<linalg::LinalgOp>(operation)) {
        if (config.storePolicy == "direct-ddr" &&
            isDirectExternalStore(linalg, config))
          needsDirectAbi = true;
        if (config.storePolicy == "direct-ddr" &&
            linalg->hasAttr(::buddy::boscame::lowering::kLowerToAme))
          for (OpOperand &operand : linalg->getOpOperands())
            if (operand.getOperandNumber() >= linalg.getNumDpsInputs() &&
                !spmdma::getSpmBank(operand.get().getType()) &&
                isExternalDdrBuffer(operand.get()) &&
                (!isa<linalg::MatmulOp>(linalg) ||
                 linalg->hasAttr(kDirectCOperandAttr)))
              needsDirectAbi = true;
        if (auto matmul = dyn_cast<linalg::MatmulOp>(operation))
          if (matmul->hasAttr(kDirectAOperandAttr) ||
              matmul->hasAttr(kDirectBOperandAttr) ||
              matmul->hasAttr(kDirectCOperandAttr))
            needsDirectAbi = true;
      }
    });
    if (needsDirectAbi && config.dmaAbi != spmdma::kDmaAbiAlignedNonAliasing)
      return fail("direct DDR AME operands/stores require "
                  "dma-abi=args_aligned64_nonalias");
  }

  bool invalidTensor = false;
  bool invalidSpmSignature = false;
  module.walk([&](Operation *op) {
    if (llvm::any_of(op->getOperandTypes(),
                     [](Type type) { return isa<TensorType>(type); }) ||
        llvm::any_of(op->getResultTypes(),
                     [](Type type) { return isa<TensorType>(type); }))
      invalidTensor = true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          invalidTensor |= isa<TensorType>(argument.getType());
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      invalidTensor |=
          llvm::any_of(function.getFunctionType().getInputs(),
                       [](Type t) { return isa<TensorType>(t); }) ||
          llvm::any_of(function.getFunctionType().getResults(),
                       [](Type t) { return isa<TensorType>(t); });
      invalidSpmSignature |=
          llvm::any_of(
              function.getFunctionType().getInputs(),
              [](Type t) { return spmdma::getSpmBank(t).has_value(); }) ||
          llvm::any_of(function.getFunctionType().getResults(), [](Type t) {
            return spmdma::getSpmBank(t).has_value();
          });
    }
    if (op->getName().getStringRef() == "builtin.unrealized_conversion_cast")
      invalidTensor = true;
  });
  if (invalidTensor)
    return fail("linalg-to-boscame input must be pure buffer-form Linalg "
                "without tensor SSA or unrealized_conversion_cast");
  if (invalidSpmSignature)
    return fail("BOSCAME function signatures cannot contain SPM memrefs; "
                "use explicit bosc_ame.spm.view windows");

  // Residency candidates are derived from the complete graph before transfer
  // planning. Incomplete metadata deliberately keeps the operation-local
  // behavior below, where existing legality checks remain sufficient.
  KernelFacts graphFacts;
  graphFacts.launches = workspace.launches;
  collectKernelComputeFacts(module, graphFacts);
  collectBufferAccessFacts(module, graphFacts);
  auto aliasContext = getBufferAliasContext(module, config.dmaAbi);
  collectMemoryDependencies(graphFacts, aliasContext);
  auto residentCandidates =
      findResidentCandidates(module, graphFacts, aliasContext);
  DenseMap<Value, size_t> residentTransfers;

  // An explicit window can remain live across a call into another function.
  // Reserve every such window across the module, not just the consumer's
  // enclosing function, so staging in a callee cannot clobber its caller.
  struct ExplicitSpmWindow {
    int64_t bank;
    uint64_t begin;
    uint64_t end;
  };
  SmallVector<ExplicitSpmWindow> explicitWindows;
  int64_t reservedBytes[spmdma::kSpmBankCount] = {};
  LogicalResult reservationStatus = success();
  module.walk([&](SpmViewOp view) {
    if (failed(reservationStatus))
      return;
    int64_t bank = view.getBank();
    if (bank < 0 || bank >= workspace.target.spmBankCount) {
      reservationStatus = view.emitOpError()
                          << "SPM window bank must be in [0, "
                          << (workspace.target.spmBankCount - 1) << "], got "
                          << bank;
      return;
    }

    auto type = dyn_cast<MemRefType>(view.getView().getType());
    auto size = type ? spmdma::getContiguousZeroOffsetByteSize(type)
                     : std::optional<int64_t>();
    if (!size || *size <= 0) {
      reservationStatus = view.emitOpError(
          "SPM window must have a positive static contiguous byte size");
      return;
    }
    uint64_t offset = view.getOffset();
    if (*size > workspace.bankCapacity ||
        offset > static_cast<uint64_t>(workspace.bankCapacity - *size)) {
      reservationStatus =
          view.emitOpError()
          << "SPM window byte range [" << offset << ", "
          << (offset <= static_cast<uint64_t>(workspace.bankCapacity)
                  ? offset + *size
                  : 0)
          << ") exceeds the " << workspace.bankCapacity << "-byte SPM bank "
          << bank;
      return;
    }
    uint64_t end = offset + *size;
    for (const ExplicitSpmWindow &other : explicitWindows) {
      if (other.bank != bank || (offset >= other.end || other.begin >= end))
        continue;
      reservationStatus = view.emitOpError()
                          << "overlapping explicit SPM windows in bank " << bank
                          << ": [" << offset << ", " << end << ") overlaps ["
                          << other.begin << ", " << other.end << ")";
      return;
    }
    explicitWindows.push_back({bank, offset, end});
    reservedBytes[bank] =
        std::max(reservedBytes[bank], static_cast<int64_t>(end));
  });
  if (failed(reservationStatus))
    return failure();

  if (config.bufferPlacement) {
    collectBufferPlacementFacts(module, graphFacts, workspace);
    if (failed(planBufferPlacements(module, config, workspace,
                                    ArrayRef<int64_t>(reservedBytes, 2))) ||
        failed(applyBufferPlacements(module, workspace)) ||
        failed(verifyCompleteComputeGraphs(module)))
      return failure();
    // Allocation replacement changes alias handles, not logical task IDs.
    // Recollect all placement inputs before the ordinary transfer planner.
    graphFacts = KernelFacts();
    graphFacts.launches = workspace.launches;
    collectKernelComputeFacts(module, graphFacts);
    collectBufferAccessFacts(module, graphFacts);
    aliasContext = getBufferAliasContext(module, config.dmaAbi);
    collectMemoryDependencies(graphFacts, aliasContext);
    residentCandidates =
        findResidentCandidates(module, graphFacts, aliasContext);
    for (const BufferPlacementDecision &decision : workspace.bufferPlacements)
      if (decision.applied) {
        int64_t end =
            decision.offset +
            workspace.rootStorageFacts[decision.rootIndex].capacityBytes;
        reservedBytes[decision.bank] =
            std::max(reservedBytes[decision.bank], end);
        workspace.usedBytes[decision.bank] = reservedBytes[decision.bank];
      }
  }

  SmallVector<FunctionArgumentResidencyDecision, 4> argumentStrategies;
  SmallVector<size_t, 4> trafficBeneficialStrategies;
  if (config.enableSpmDma)
    module.walk([&](func::FuncOp function) {
      if (function.isExternal())
        return;
      for (unsigned index = 0; index < function.getNumArguments(); ++index) {
        Value argument = function.getArgument(index);
        auto candidate = residentCandidates.find(argument);
        if (candidate == residentCandidates.end() || candidate->second.producer)
          continue;

        FunctionArgumentResidencyDecision decision;
        decision.argument = argument;
        decision.consumerCount = candidate->second.consumers.size();
        bool exact = !candidate->second.consumers.empty();
        bool hasResidentCost = false;
        for (Operation *consumer : candidate->second.consumers) {
          auto linalg = dyn_cast<linalg::LinalgOp>(consumer);
          if (!linalg) {
            exact = false;
            break;
          }
          unsigned operandIndex = linalg->getNumOperands();
          for (auto [operandNumber, operand] :
               llvm::enumerate(linalg->getOperands()))
            if (operand == argument) {
              operandIndex = operandNumber;
              break;
            }
          if (operandIndex >= linalg.getNumDpsInputs()) {
            exact = false;
            break;
          }
          TransferRole role =
              operandIndex == 0 ? TransferRole::InputA : TransferRole::InputB;
          auto footprint = analyzeBOSCAMEFootprint(argument, role, consumer,
                                                   workspace.target);
          if (!footprint) {
            exact = false;
            break;
          }
          uint64_t useBytes = 0;
          uint64_t usePackets = 0;
          if (!getPacketReadCost(*footprint, useBytes, usePackets) ||
              useBytes == 0 || usePackets == 0) {
            exact = false;
            break;
          }
          if (!hasResidentCost) {
            decision.capacityBytes = footprint->spmCapacity;
            decision.residentReadBytes = useBytes;
            decision.residentPackets = usePackets;
            hasResidentCost = true;
          } else if (decision.capacityBytes != footprint->spmCapacity) {
            exact = false;
            break;
          }
          if (useBytes > std::numeric_limits<uint64_t>::max() -
                             decision.reloadReadBytes ||
              usePackets > std::numeric_limits<uint64_t>::max() -
                               decision.reloadPackets) {
            exact = false;
            break;
          }
          decision.reloadReadBytes += useBytes;
          decision.reloadPackets += usePackets;
        }
        if (!exact || !hasResidentCost)
          continue;
        size_t decisionIndex = argumentStrategies.size();
        if (hasLowerReadCost(decision.residentReadBytes,
                             decision.residentPackets, decision.reloadReadBytes,
                             decision.reloadPackets))
          trafficBeneficialStrategies.push_back(decisionIndex);
        argumentStrategies.push_back(decision);
      }
    });

  struct CapacityRequirement {
    Value buffer;
    int64_t capacity = 0;
    unsigned operandIndex = 0;
  };
  struct OperationCapacityGroup {
    Operation *operation = nullptr;
    Block *block = nullptr;
    unsigned position = 0;
    unsigned numDpsInputs = 0;
    SmallVector<CapacityRequirement, 4> requirements;
  };
  auto getBlockPosition = [](Block *block, Operation *operation) {
    unsigned position = 0;
    for (Operation &candidate : *block) {
      if (&candidate == operation)
        return position;
      ++position;
    }
    return std::numeric_limits<unsigned>::max();
  };
  SmallVector<OperationCapacityGroup, 8> capacityGroups;
  if (config.enableSpmDma)
    module.walk([&](linalg::LinalgOp op) {
      bool ordinaryMatmul = false;
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation()))
        ordinaryMatmul =
            !matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme);
      if (ordinaryMatmul)
        return;
      bool usesFixedSpm = llvm::any_of(op->getOperandTypes(), [](Type type) {
        return spmdma::getSpmBank(type).has_value();
      });
      bool stagesBuffers = !ordinaryMatmul && isStageableBufferLinalg(op) &&
                           llvm::any_of(op->getOperandTypes(), [](Type type) {
                             auto memref = dyn_cast<MemRefType>(type);
                             return memref && memref.getRank() > 0;
                           });
      if (!stagesBuffers && !usesFixedSpm)
        return;

      bool directExternalStore = isDirectExternalStore(op, config);
      OperationCapacityGroup group;
      group.operation = op.getOperation();
      group.block = op->getBlock();
      group.position = getBlockPosition(group.block, group.operation);
      group.numDpsInputs = op.getNumDpsInputs();
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        auto type = dyn_cast<MemRefType>(operand.getType());
        if (!type || type.getRank() == 0 || spmdma::getSpmBank(type) ||
            isPredicateOnlyInput(op, index) || isPredicateOnlyOutput(op, index))
          continue;
        bool write = index >= static_cast<size_t>(op.getNumDpsInputs());
        if (directExternalStore && write)
          continue;
        if (!write) {
          if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation()))
            if ((index == 0 && matmul->hasAttr(kDirectAOperandAttr)) ||
                (index == 1 && matmul->hasAttr(kDirectBOperandAttr)))
              continue;
        }
        auto memory = op->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
        auto kind = memory ? memory.getAs<StringAttr>("kind") : StringAttr();
        auto destination =
            memory ? memory.getAs<StringAttr>("destination") : StringAttr();
        bool externalStore = kind && destination &&
                             kind.getValue() == "store" &&
                             destination.getValue() == "external_ddr";
        if (write && config.storePolicy == "direct-ddr" &&
            (externalStore || op->hasAttr(kDirectCOperandAttr)) &&
            isExternalDdrBuffer(operand))
          continue;
        if (llvm::any_of(group.requirements, [&](const CapacityRequirement &r) {
              return r.buffer == operand;
            }))
          continue;
        group.requirements.push_back(
            {operand,
             getStaticTransferCapacity(operand, workspace.bankCapacity),
             static_cast<unsigned>(index)});
      }
      if (!group.requirements.empty())
        capacityGroups.push_back(std::move(group));
    });

  Block *residentBlock = nullptr;
  bool singleResidentBlock = true;
  for (const auto &entry : residentCandidates) {
    const ResidentCandidate &candidate = entry.second;
    Operation *first =
        candidate.producer ? candidate.producer : candidate.consumers.front();
    if (!residentBlock)
      residentBlock = first->getBlock();
    else if (residentBlock != first->getBlock())
      singleResidentBlock = false;
  }
  for (const OperationCapacityGroup &group : capacityGroups)
    if (residentBlock && group.block != residentBlock)
      singleResidentBlock = false;
  bool hasDynamicPrefixTransfer = false;
  module.walk([&](memref::CopyOp copy) {
    auto memory = copy->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
    auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
    hasDynamicPrefixTransfer |=
        memory && sourceType && !sourceType.hasStaticShape();
  });
  bool reuseResidentSlots = residentBlock && singleResidentBlock &&
                            !capacityGroups.empty() &&
                            !hasDynamicPrefixTransfer;

  SmallVector<std::pair<memref::CopyOp, TransferPlan>, 2>
      dynamicPrefixTransfers;
  LogicalResult dynamicPrefixStatus = success();
  if (config.enableSpmDma)
    module.walk([&](memref::CopyOp copy) {
      if (failed(dynamicPrefixStatus))
        return;
      auto memory = copy->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
      auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
      if (!memory || !sourceType || sourceType.hasStaticShape())
        return;
      auto transfer = analyzeDynamicPrefixCopy(copy, config, workspace.target,
                                               aliasContext);
      if (failed(transfer) || failed(verifySerialExecution(copy))) {
        dynamicPrefixStatus = failure();
        return;
      }
      dynamicPrefixTransfers.emplace_back(copy, std::move(*transfer));
    });
  if (failed(dynamicPrefixStatus))
    return failure();

  DenseMap<Value, unsigned> residentLastPositions;
  if (reuseResidentSlots)
    for (const auto &entry : residentCandidates)
      residentLastPositions[entry.first] =
          getBlockPosition(residentBlock, entry.second.consumers.back());

  struct ResidentWindowRange {
    unsigned lastPosition = 0;
    int64_t bank = 0;
    int64_t end = 0;
  };
  auto selectionFits = [&](const DenseSet<Value> &selectedArguments,
                           const DenseSet<Value> &selectedProducers) {
    int64_t simulatedReserved[spmdma::kSpmBankCount] = {reservedBytes[0],
                                                        reservedBytes[1]};
    DenseSet<Value> materializedResidents;
    SmallVector<ResidentWindowRange, 4> activeResidents;
    ResourceEstimate ignoredResources;
    for (const OperationCapacityGroup &group : capacityGroups) {
      if (reuseResidentSlots) {
        llvm::erase_if(activeResidents, [&](const ResidentWindowRange &item) {
          return item.lastPosition < group.position;
        });
        simulatedReserved[0] = reservedBytes[0];
        simulatedReserved[1] = reservedBytes[1];
        for (const ResidentWindowRange &item : activeResidents)
          simulatedReserved[item.bank] =
              std::max(simulatedReserved[item.bank], item.end);
      }
      SmallVector<TransferPlan, 4> transfers;
      for (const CapacityRequirement &requirement : group.requirements) {
        auto candidate = residentCandidates.find(requirement.buffer);
        bool candidateIsArgument = candidate != residentCandidates.end() &&
                                   !candidate->second.producer &&
                                   isFunctionEntryArgument(requirement.buffer);
        bool candidateIsProducer =
            candidate != residentCandidates.end() && candidate->second.producer;
        bool resident =
            candidate != residentCandidates.end() &&
            (candidateIsProducer
                 ? selectedProducers.contains(requirement.buffer)
                 : (!candidateIsArgument ||
                    selectedArguments.contains(requirement.buffer)));
        bool isFirstResidentUse = false;
        if (resident) {
          const ResidentCandidate &residency = candidate->second;
          if (residency.producer == group.operation &&
              requirement.operandIndex >= group.numDpsInputs) {
            isFirstResidentUse = true;
          } else if (requirement.operandIndex < group.numDpsInputs &&
                     llvm::is_contained(residency.consumers, group.operation)) {
            if (materializedResidents.contains(requirement.buffer))
              continue;
            if (!residency.producer && !residency.consumers.empty() &&
                residency.consumers.front() == group.operation) {
              isFirstResidentUse = true;
            } else {
              return false;
            }
          }
        }
        TransferPlan transfer;
        transfer.buffer = requirement.buffer;
        transfer.spmCapacity = requirement.capacity;
        transfer.crossTaskResident = isFirstResidentUse;
        transfers.push_back(std::move(transfer));
        if (isFirstResidentUse)
          materializedResidents.insert(requirement.buffer);
      }
      if (transfers.empty())
        continue;
      llvm::stable_sort(transfers,
                        [](const TransferPlan &lhs, const TransferPlan &rhs) {
                          return lhs.spmCapacity > rhs.spmCapacity;
                        });
      auto allocation = chooseBankAllocation(
          transfers,
          ArrayRef<int64_t>(simulatedReserved, spmdma::kSpmBankCount),
          workspace.bankCapacity, workspace.target.dma.endpointAlignmentBytes,
          ignoredResources);
      if (!allocation)
        return false;
      for (auto [index, transfer] : llvm::enumerate(transfers))
        if (transfer.crossTaskResident) {
          simulatedReserved[allocation->banks[index]] =
              std::max(simulatedReserved[allocation->banks[index]],
                       allocation->offsets[index] + transfer.spmCapacity);
          if (reuseResidentSlots) {
            auto lastPosition = residentLastPositions.find(transfer.buffer);
            if (lastPosition == residentLastPositions.end())
              return false;
            activeResidents.push_back(
                {lastPosition->second, allocation->banks[index],
                 allocation->offsets[index] + transfer.spmCapacity});
          }
        }
    }
    for (const auto &entry : dynamicPrefixTransfers) {
      SmallVector<TransferPlan, 1> transfer{entry.second};
      if (!chooseBankAllocation(
              transfer,
              ArrayRef<int64_t>(simulatedReserved, spmdma::kSpmBankCount),
              workspace.bankCapacity,
              workspace.target.dma.endpointAlignmentBytes, ignoredResources))
        return false;
    }
    for (Value argument : selectedArguments)
      if (!materializedResidents.contains(argument))
        return false;
    for (Value producer : selectedProducers)
      if (!materializedResidents.contains(producer))
        return false;
    return true;
  };

  // Rank producer-output residency by the exact DDR reload traffic it removes.
  // The producer writeback is still required in either plan, so only consumer
  // reloads contribute to this comparison.
  SmallVector<ProducerOutputResidencyDecision, 4> producerStrategies;
  SmallVector<Value, 4> producerResidencyCandidates;
  module.walk([&](linalg::LinalgOp op) {
    for (Value output : op.getDpsInits()) {
      auto candidate = residentCandidates.find(output);
      if (candidate != residentCandidates.end() &&
          candidate->second.producer == op.getOperation() &&
          !llvm::is_contained(producerResidencyCandidates, output))
        producerResidencyCandidates.push_back(output);
    }
  });
  for (Value output : producerResidencyCandidates) {
    const ResidentCandidate &candidate =
        residentCandidates.find(output)->second;
    ProducerOutputResidencyDecision decision;
    decision.output = output;
    decision.producer = candidate.producer;
    decision.consumerCount = candidate.consumers.size();
    bool exact = !candidate.consumers.empty();
    bool hasCapacity = false;
    for (Operation *consumer : candidate.consumers) {
      auto linalg = dyn_cast<linalg::LinalgOp>(consumer);
      if (!linalg) {
        exact = false;
        break;
      }
      unsigned operandIndex = linalg->getNumOperands();
      for (auto [index, operand] : llvm::enumerate(linalg->getOperands()))
        if (operand == output) {
          operandIndex = index;
          break;
        }
      if (operandIndex >= linalg.getNumDpsInputs()) {
        exact = false;
        break;
      }
      TransferRole role =
          operandIndex == 0 ? TransferRole::InputA : TransferRole::InputB;
      auto footprint =
          analyzeBOSCAMEFootprint(output, role, consumer, workspace.target);
      if (!footprint) {
        exact = false;
        break;
      }
      uint64_t readBytes = 0;
      uint64_t packets = 0;
      if (!getPacketReadCost(*footprint, readBytes, packets) || !readBytes ||
          !packets) {
        exact = false;
        break;
      }
      if (!hasCapacity) {
        decision.capacityBytes = footprint->spmCapacity;
        hasCapacity = true;
      } else if (decision.capacityBytes != footprint->spmCapacity) {
        exact = false;
        break;
      }
      if (readBytes >
              std::numeric_limits<uint64_t>::max() - decision.savedReadBytes ||
          packets >
              std::numeric_limits<uint64_t>::max() - decision.savedPackets) {
        exact = false;
        break;
      }
      decision.savedReadBytes += readBytes;
      decision.savedPackets += packets;
    }
    if (exact && hasCapacity && decision.savedReadBytes &&
        decision.savedPackets)
      producerStrategies.push_back(decision);
  }

  constexpr size_t kMaxExactResidencyCandidates = 8;
  enum class ResidencyCandidateKind { ProducerOutput, FunctionArgument };
  struct ResidencyCandidateRef {
    ResidencyCandidateKind kind;
    size_t index;
    uint64_t savedReadBytes;
    bool packetCostIncrease;
    uint64_t packetDelta;
  };
  SmallVector<ResidencyCandidateRef, 8> jointCandidates;
  // Preserve module operation order for producer outputs, followed by
  // function/argument order. Exact ties retain this order; the bounded sort
  // below is stable as well.
  for (auto [index, decision] : llvm::enumerate(producerStrategies))
    jointCandidates.push_back({ResidencyCandidateKind::ProducerOutput, index,
                               decision.savedReadBytes, false,
                               decision.savedPackets});
  for (size_t index : trafficBeneficialStrategies) {
    const FunctionArgumentResidencyDecision &decision =
        argumentStrategies[index];
    bool packetCostIncrease = decision.residentPackets > decision.reloadPackets;
    uint64_t packetDelta =
        packetCostIncrease ? decision.residentPackets - decision.reloadPackets
                           : decision.reloadPackets - decision.residentPackets;
    jointCandidates.push_back(
        {ResidencyCandidateKind::FunctionArgument, index,
         decision.reloadReadBytes - decision.residentReadBytes,
         packetCostIncrease, packetDelta});
  }

  bool jointSearchBounded =
      jointCandidates.size() > kMaxExactResidencyCandidates;
  workspace.producerOutputResidencySearchBounded =
      jointSearchBounded && !producerStrategies.empty();
  workspace.functionArgumentResidencySearchBounded =
      jointSearchBounded && !trafficBeneficialStrategies.empty();
  DenseSet<Value> selectedProducerResidencies;
  DenseSet<Value> selectedArgumentResidencies;
  auto selectionCost = [&](const DenseSet<Value> &selectedArguments,
                           const DenseSet<Value> &selectedProducers,
                           std::pair<uint64_t, uint64_t> &cost) {
    auto addCost = [&](uint64_t bytes, uint64_t packets) {
      if (bytes > std::numeric_limits<uint64_t>::max() - cost.first ||
          packets > std::numeric_limits<uint64_t>::max() - cost.second)
        return false;
      cost.first += bytes;
      cost.second += packets;
      return true;
    };
    for (const FunctionArgumentResidencyDecision &decision :
         argumentStrategies) {
      bool resident = selectedArguments.contains(decision.argument);
      if (!addCost(
              resident ? decision.residentReadBytes : decision.reloadReadBytes,
              resident ? decision.residentPackets : decision.reloadPackets))
        return false;
    }
    for (const ProducerOutputResidencyDecision &decision : producerStrategies)
      if (!selectedProducers.contains(decision.output) &&
          !addCost(decision.savedReadBytes, decision.savedPackets))
        return false;
    return true;
  };
  auto addCandidate = [&](const ResidencyCandidateRef &candidate,
                          DenseSet<Value> &arguments,
                          DenseSet<Value> &producers) {
    if (candidate.kind == ResidencyCandidateKind::ProducerOutput)
      producers.insert(producerStrategies[candidate.index].output);
    else
      arguments.insert(argumentStrategies[candidate.index].argument);
  };

  if (!jointSearchBounded) {
    uint64_t candidateCount = uint64_t{1} << jointCandidates.size();
    bool haveFeasibleSelection = false;
    std::pair<uint64_t, uint64_t> bestCost{
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max()};
    for (uint64_t mask = 0; mask < candidateCount; ++mask) {
      DenseSet<Value> selectedArguments;
      DenseSet<Value> selectedProducers;
      for (size_t index = 0; index < jointCandidates.size(); ++index)
        if ((mask >> index) & 1)
          addCandidate(jointCandidates[index], selectedArguments,
                       selectedProducers);
      std::pair<uint64_t, uint64_t> cost{0, 0};
      if (!selectionCost(selectedArguments, selectedProducers, cost) ||
          !selectionFits(selectedArguments, selectedProducers))
        continue;
      if (!haveFeasibleSelection || cost < bestCost) {
        haveFeasibleSelection = true;
        bestCost = cost;
        selectedArgumentResidencies = std::move(selectedArguments);
        selectedProducerResidencies = std::move(selectedProducers);
      }
    }
  } else {
    llvm::stable_sort(jointCandidates, [](const ResidencyCandidateRef &lhs,
                                          const ResidencyCandidateRef &rhs) {
      if (lhs.savedReadBytes != rhs.savedReadBytes)
        return lhs.savedReadBytes > rhs.savedReadBytes;
      if (lhs.packetCostIncrease != rhs.packetCostIncrease)
        return !lhs.packetCostIncrease;
      if (lhs.packetDelta != rhs.packetDelta)
        return lhs.packetCostIncrease ? lhs.packetDelta < rhs.packetDelta
                                      : lhs.packetDelta > rhs.packetDelta;
      return false;
    });
    for (const ResidencyCandidateRef &candidate : jointCandidates) {
      DenseSet<Value> trialArguments = selectedArgumentResidencies;
      DenseSet<Value> trialProducers = selectedProducerResidencies;
      addCandidate(candidate, trialArguments, trialProducers);
      if (selectionFits(trialArguments, trialProducers)) {
        selectedArgumentResidencies = std::move(trialArguments);
        selectedProducerResidencies = std::move(trialProducers);
      }
    }
  }

  for (ProducerOutputResidencyDecision &decision : producerStrategies)
    decision.selected = selectedProducerResidencies.contains(decision.output);
  workspace.producerOutputResidency = std::move(producerStrategies);
  for (FunctionArgumentResidencyDecision &decision : argumentStrategies) {
    if (selectedArgumentResidencies.contains(decision.argument)) {
      decision.reason = ResidencyDecisionReason::Selected;
    } else if (hasLowerReadCost(
                   decision.residentReadBytes, decision.residentPackets,
                   decision.reloadReadBytes, decision.reloadPackets)) {
      decision.reason = ResidencyDecisionReason::Capacity;
    } else {
      decision.reason = ResidencyDecisionReason::NoTrafficBenefit;
    }
  }
  workspace.functionArgumentResidency = std::move(argumentStrategies);

  llvm::SmallPtrSet<Operation *, 16> workspaceFunctions;
  SmallVector<ResidentWindowRange, 4> activeResidentWindows;
  LogicalResult status = success();
  module.walk([&](memref::CopyOp copy) {
    if (failed(status) || (!spmdma::getSpmBank(copy.getSource().getType()) &&
                           !spmdma::getSpmBank(copy.getTarget().getType())))
      return;
    status = verifySerialExecution(copy);
    if (succeeded(status))
      workspaceFunctions.insert(copy->getParentOfType<func::FuncOp>());
  });
  module.walk([&](linalg::LinalgOp op) {
    if (failed(status) || !op.hasPureBufferSemantics()) {
      if (succeeded(status))
        status = op.emitError("linalg-to-boscame requires pure buffer "
                              "semantics");
      return;
    }
    if (reuseResidentSlots && op->getBlock() == residentBlock) {
      unsigned position = getBlockPosition(residentBlock, op.getOperation());
      llvm::erase_if(activeResidentWindows,
                     [&](const ResidentWindowRange &item) {
                       return item.lastPosition < position;
                     });
    }
    bool ordinaryMatmul = false;
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation())) {
      if (matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme) &&
          !config.enableAME) {
        status = op.emitError("marked matmul requires enable-ame=true");
        return;
      }
      if (!isSupportedAmeMatmul(matmul, target->ame)) {
        status = op.emitError("marked AME matmul must be signed i8 x i8 -> "
                              "i32 with rank-2 buffers");
        return;
      }
      // Unmarked matmuls stay on the ordinary scalar/RVV path.  They do not
      // acquire an SPM workspace merely because the planner option is enabled;
      // requiring 64-byte DMA packets for a tiny CPU matmul would turn a
      // perfectly valid RVV lowering into a spurious transfer diagnostic.
      ordinaryMatmul =
          !matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme);
    }
    bool usesFixedSpm = llvm::any_of(op->getOperandTypes(), [](Type type) {
      return spmdma::getSpmBank(type).has_value();
    });
    if (config.bufferPlacement && usesFixedSpm)
      if (auto boundary = planResidentBoundaryTransfer(op, workspace.target)) {
        if (failed(verifySerialExecution(op))) {
          status = failure();
          return;
        }
        workspace.transfers.push_back(std::move(*boundary));
        workspaceFunctions.insert(op->getParentOfType<func::FuncOp>());
        return;
      }
    bool directExternalStore = isDirectExternalStore(op, config);
    SmallVector<Value> directDestinations;
    if (directExternalStore) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.getOperandNumber() < op.getNumDpsInputs())
          continue;
        if (failed(verifyDirectStoreMetadata(op, operand.get()))) {
          status = failure();
          return;
        }
        if (!isExternalDdrBuffer(operand.get())) {
          status = op.emitError(
              "external_ddr store provenance does not name a function DDR "
              "buffer");
          return;
        }
        directDestinations.push_back(operand.get());
      }
    }
    // AME accumulator stores carry their direct-DDR marker on the matmul
    // rather than on a frontend memory event. Apply the same definite-alias
    // check to an external C output versus A/B inputs.
    if (config.storePolicy == "direct-ddr" &&
        op->hasAttr(::buddy::boscame::lowering::kLowerToAme) &&
        op->hasAttr(kDirectCOperandAttr))
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.getOperandNumber() < op.getNumDpsInputs())
          continue;
        auto type = dyn_cast<MemRefType>(operand.get().getType());
        if (type && !spmdma::getSpmBank(type) &&
            isExternalDdrBuffer(operand.get()))
          directDestinations.push_back(operand.get());
      }
    // A direct store has no temporary write barrier. If an input is a view of
    // the same storage root, a later direct write could overwrite bytes that
    // a future iteration still needs. Keep same-root accesses only when their
    // byte ranges are proven disjoint; overlap and unknown ranges remain
    // rejected.
    for (Value destination : directDestinations)
      for (Value input : op.getDpsInputs()) {
        if (getStorageRoot(destination) != getStorageRoot(input))
          continue;
        BufferAccessFact destinationAccess;
        destinationAccess.operation = op.getOperation();
        destinationAccess.effect = BufferAccessEffect::Write;
        destinationAccess.accessedValue = destination;
        populateBufferAccessLayout(destinationAccess);
        BufferAccessFact inputAccess;
        inputAccess.operation = op.getOperation();
        inputAccess.effect = BufferAccessEffect::Read;
        inputAccess.accessedValue = input;
        populateBufferAccessLayout(inputAccess);
        if (!getDisjointAccessProof(destinationAccess, inputAccess,
                                    aliasContext)) {
          status = op.emitError(
              "direct-ddr store destination aliases a Linalg input; "
              "select store-policy=spm-dma or use distinct buffers");
          return;
        }
      }
    bool stagesBuffers = config.enableSpmDma && !ordinaryMatmul &&
                         isStageableBufferLinalg(op) &&
                         llvm::any_of(op->getOperandTypes(), [](Type type) {
                           auto memref = dyn_cast<MemRefType>(type);
                           return memref && memref.getRank() > 0;
                         });
    if ((usesFixedSpm || stagesBuffers ||
         op->hasAttr(::buddy::boscame::lowering::kLowerToAme)) &&
        failed(verifySerialExecution(op))) {
      status = failure();
      return;
    }
    if (usesFixedSpm || stagesBuffers ||
        op->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      workspaceFunctions.insert(op->getParentOfType<func::FuncOp>());
    if (ordinaryMatmul)
      return;
    if (!config.enableSpmDma)
      return;
    // Only the buffer forms that the SPM lowering understands may acquire a
    // staged workspace.  Unsupported element types, reductions and generic
    // affine maps stay on the ordinary lowering path; staging them here would
    // emit DMA requests for an operation that is never rewritten to consume
    // the corresponding SPM tiles.  Explicit SPM operands remain covered by
    // the checks above and are handled by their dedicated lowering path.
    if (!stagesBuffers && !usesFixedSpm)
      return;
    SmallVector<TransferPlan, 4> operationTransfers;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || type.getRank() == 0)
        continue; // Rank-zero reduction results stay in ordinary memory.
      if (spmdma::getSpmBank(type))
        continue;
      if (isPredicateOnlyInput(op, index))
        continue; // RVV consumes the predicate; it is never a DMA payload.
      if (isPredicateOnlyOutput(op, index))
        continue; // Predicate results stay in ordinary temporary storage.
      bool residentFirst = false;
      auto resident = residentCandidates.find(operand);
      bool residentEnabled =
          resident != residentCandidates.end() &&
          (resident->second.producer
               ? selectedProducerResidencies.contains(operand)
               : (!isFunctionEntryArgument(operand) ||
                  selectedArgumentResidencies.contains(operand)));
      if (residentEnabled) {
        if (resident->second.producer == op.getOperation()) {
          // The producer's output is the first materialization. Keep that
          // SPM window live through all read consumers and emit the DDR write
          // only after the final consumer.
          residentFirst = true;
        } else if (index < static_cast<size_t>(op.getNumDpsInputs())) {
          if (resident->second.producer ||
              op.getOperation() != resident->second.consumers.front()) {
            auto existing = residentTransfers.find(operand);
            if (existing != residentTransfers.end()) {
              TransferPlan &shared = workspace.transfers[existing->second];
              shared.lastConsumer = op.getOperation();
              continue;
            }
            // The module walk is expected to encounter the first consumer
            // first; retain the ordinary transfer if that invariant fails.
          } else {
            residentFirst = true;
          }
        } else {
          residentFirst = true;
        }
      }
      bool write = index >= static_cast<size_t>(op.getNumDpsInputs());
      bool read =
          !write || op.payloadUsesValueFromOperand(&op->getOpOperand(index));
      auto provenance = op->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
      bool externalStore =
          provenance && provenance.getAs<StringAttr>("kind") &&
          provenance.getAs<StringAttr>("kind").getValue() == "store" &&
          provenance.getAs<StringAttr>("destination") &&
          provenance.getAs<StringAttr>("destination").getValue() ==
              "external_ddr";
      if (directExternalStore && write) {
        if (!isExternalDdrBuffer(operand)) {
          status = op.emitError(
              "external_ddr store provenance does not name a function DDR "
              "buffer");
          return;
        }
        continue;
      }
      // Dense logical A/B loads are consumed directly by AME.  They have no
      // SPM lifetime and therefore must not acquire a DMA transfer merely
      // because the generic planner sees an external memref operand.
      if (!write) {
        if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation())) {
          if ((index == 0 && matmul->hasAttr(kDirectAOperandAttr)) ||
              (index == 1 && matmul->hasAttr(kDirectBOperandAttr)))
            continue;
        }
      }
      // In direct-DDR mode an AME accumulator is loaded from and written back
      // to the external C buffer itself.  Do not reserve an SPM window or emit
      // a pair of C DMA packets for that output.  Explicit SPM destinations
      // retain their existing semantics and continue through the compatibility
      // path.
      if (write && config.storePolicy == "direct-ddr" &&
          (externalStore || op->hasAttr(kDirectCOperandAttr)) &&
          !spmdma::getSpmBank(type) && isExternalDdrBuffer(operand))
        continue;
      auto found =
          llvm::find_if(operationTransfers, [&](const TransferPlan &t) {
            return t.buffer == operand;
          });
      if (found != operationTransfers.end()) {
        found->read |= read;
        found->write |= write;
        if (residentFirst) {
          found->crossTaskResident = true;
          found->consumerOperations = resident->second.consumers;
        }
        continue;
      }
      TransferRole role =
          write ? TransferRole::OutputC
                : (index == 0 ? TransferRole::InputA : TransferRole::InputB);
      auto transfer =
          analyzeBOSCAMEFootprint(operand, role, op, workspace.target);
      if (!transfer) {
        status = failure();
        return;
      }
      transfer->read = read;
      transfer->write = write;
      transfer->firstConsumer = op;
      transfer->lastConsumer = op;
      transfer->consumerOperations.push_back(op.getOperation());
      if (residentFirst) {
        transfer->crossTaskResident = true;
        transfer->consumerOperations = resident->second.consumers;
        if (resident->second.producer)
          transfer->consumerOperations.insert(
              transfer->consumerOperations.begin(), resident->second.producer);
        transfer->lastConsumer = transfer->consumerOperations.back();
      }
      operationTransfers.push_back(std::move(*transfer));
    }
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation())) {
      AmeMatmulTilePlan *tilePlan =
          findAmeMatmulTilePlan(workspace, matmul.getOperation());
      if (tilePlan &&
          matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme)) {
        auto chooseInputRoute = [&](unsigned index, bool eligible,
                                    StringLiteral directAttr,
                                    uint64_t directBytes, uint64_t &stagedBytes,
                                    bool &capacityFallback) {
          if (!eligible || matmul->hasAttr(directAttr))
            return;
          Value input = matmul.getInputs()[index];
          auto transfer = llvm::find_if(operationTransfers,
                                        [&](const TransferPlan &candidate) {
                                          return candidate.buffer == input;
                                        });
          if (transfer == operationTransfers.end())
            return;
          uint64_t packets = 0;
          stagedBytes = 0;
          if (!getPacketReadCost(*transfer, stagedBytes, packets))
            return;
          (void)packets;
          bool stagingFits =
              stagedBytes <= static_cast<uint64_t>(workspace.bankCapacity);
          if (stagingFits && directBytes > stagedBytes)
            return;
          if (!stagingFits)
            capacityFallback = true;
          matmul->setAttr(directAttr, UnitAttr::get(matmul.getContext()));
          operationTransfers.erase(transfer);
        };
        chooseInputRoute(0, tilePlan->directAEligible, kDirectAOperandAttr,
                         tilePlan->directAReadBytes, tilePlan->stagedAReadBytes,
                         tilePlan->directAByCapacity);
        chooseInputRoute(1, tilePlan->directBEligible, kDirectBOperandAttr,
                         tilePlan->directBReadBytes, tilePlan->stagedBReadBytes,
                         tilePlan->directBByCapacity);
      }
    }
    for (size_t i = 0; i < operationTransfers.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        if (operationTransfers[i].ddrRoot == operationTransfers[j].ddrRoot &&
            (operationTransfers[i].write || operationTransfers[j].write) &&
            !areTransferByteIntervalsDisjoint(operationTransfers[i],
                                              operationTransfers[j])) {
          status = op.emitError("BOSCAME cannot prove distinct writable views "
                                "of one DDR root do not alias");
          return;
        }
    // Operation-local transfers are waited before their consumer and writes
    // are drained immediately afterward. Resident windows remain reserved
    // through their final consumer, including a delayed producer writeback.
    llvm::stable_sort(operationTransfers, [](const auto &a, const auto &b) {
      return a.spmCapacity > b.spmCapacity;
    });
    int64_t allocationReserved[spmdma::kSpmBankCount] = {reservedBytes[0],
                                                         reservedBytes[1]};
    if (reuseResidentSlots)
      for (const ResidentWindowRange &item : activeResidentWindows)
        allocationReserved[item.bank] =
            std::max(allocationReserved[item.bank], item.end);
    auto allocation = chooseBankAllocation(
        operationTransfers, ArrayRef<int64_t>(allocationReserved, 2),
        workspace.bankCapacity, workspace.target.dma.endpointAlignmentBytes,
        workspace.resources);
    if (!allocation) {
      // A/B traffic selection is made before the complete operation-local
      // sequence is known. If staging the selected input set cannot fit with
      // the other live transfers, retry the small set of eligible logical
      // inputs on direct DDR. This is a capacity fallback only; the normal
      // byte comparison remains the first choice.
      auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation());
      AmeMatmulTilePlan *tilePlan =
          matmul ? findAmeMatmulTilePlan(workspace, matmul.getOperation())
                 : nullptr;
      struct DirectInputFallback {
        unsigned index;
        Value buffer;
        uint64_t directBytes;
        uint64_t stagedBytes;
      };
      SmallVector<DirectInputFallback, 2> fallbackInputs;
      if (tilePlan &&
          matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme)) {
        auto addFallbackInput = [&](unsigned index, bool eligible,
                                    StringLiteral directAttr,
                                    uint64_t directBytes) {
          if (!eligible || matmul->hasAttr(directAttr))
            return;
          Value buffer = matmul.getInputs()[index];
          auto staged = llvm::find_if(operationTransfers,
                                      [&](const TransferPlan &transfer) {
                                        return transfer.buffer == buffer;
                                      });
          if (staged == operationTransfers.end())
            return;
          uint64_t stagedBytes = 0, stagedPackets = 0;
          if (!getPacketReadCost(*staged, stagedBytes, stagedPackets))
            return;
          (void)stagedPackets;
          fallbackInputs.push_back({index, buffer, directBytes, stagedBytes});
        };
        addFallbackInput(0, tilePlan->directAEligible, kDirectAOperandAttr,
                         tilePlan->directAReadBytes);
        addFallbackInput(1, tilePlan->directBEligible, kDirectBOperandAttr,
                         tilePlan->directBReadBytes);
      }
      if (!fallbackInputs.empty()) {
        SmallVector<TransferPlan, 4> originalTransfers = operationTransfers;
        std::optional<BankAllocation> bestAllocation;
        SmallVector<TransferPlan, 4> bestTransfers;
        uint64_t bestTrafficBytes = std::numeric_limits<uint64_t>::max();
        unsigned bestDirectCount = 0;
        uint64_t bestMask = std::numeric_limits<uint64_t>::max();
        const ResourceEstimate resourcesBeforeRouteSearch = workspace.resources;
        uint64_t combinations = uint64_t{1} << fallbackInputs.size();
        for (uint64_t mask = 1; mask < combinations; ++mask) {
          SmallVector<TransferPlan, 4> trialTransfers;
          for (const TransferPlan &transfer : originalTransfers) {
            bool remove = false;
            for (auto [candidateIndex, candidate] :
                 llvm::enumerate(fallbackInputs))
              if ((mask & (uint64_t{1} << candidateIndex)) &&
                  transfer.buffer == candidate.buffer)
                remove = true;
            if (!remove)
              trialTransfers.push_back(transfer);
          }
          uint64_t trafficBytes = 0;
          bool trafficOverflow = false;
          unsigned directCount = 0;
          for (auto [candidateIndex, candidate] :
               llvm::enumerate(fallbackInputs)) {
            bool direct = mask & (uint64_t{1} << candidateIndex);
            uint64_t bytes =
                direct ? candidate.directBytes : candidate.stagedBytes;
            if (bytes > std::numeric_limits<uint64_t>::max() - trafficBytes) {
              trafficOverflow = true;
              break;
            }
            trafficBytes += bytes;
            directCount += direct;
          }
          if (trafficOverflow)
            continue;
          auto trialResources = resourcesBeforeRouteSearch;
          auto trialAllocation = chooseBankAllocation(
              trialTransfers, ArrayRef<int64_t>(allocationReserved, 2),
              workspace.bankCapacity,
              workspace.target.dma.endpointAlignmentBytes, trialResources);
          if (!trialAllocation)
            continue;
          if (!bestAllocation || trafficBytes < bestTrafficBytes ||
              (trafficBytes == bestTrafficBytes &&
               directCount > bestDirectCount) ||
              (trafficBytes == bestTrafficBytes &&
               directCount == bestDirectCount && mask < bestMask)) {
            bestAllocation = std::move(trialAllocation);
            bestTransfers = std::move(trialTransfers);
            workspace.resources = trialResources;
            bestTrafficBytes = trafficBytes;
            bestDirectCount = directCount;
            bestMask = mask;
          }
        }
        if (bestAllocation) {
          for (auto [candidateIndex, candidate] :
               llvm::enumerate(fallbackInputs))
            if (bestMask & (uint64_t{1} << candidateIndex)) {
              if (candidate.index == 0) {
                matmul->setAttr(kDirectAOperandAttr,
                                UnitAttr::get(matmul.getContext()));
                tilePlan->directAByCapacity = true;
              } else {
                matmul->setAttr(kDirectBOperandAttr,
                                UnitAttr::get(matmul.getContext()));
                tilePlan->directBByCapacity = true;
              }
            }
          operationTransfers = std::move(bestTransfers);
          allocation = std::move(bestAllocation);
        }
      }
    }
    if (!allocation) {
      status = op.emitError("BOSCAME SPM workspace capacity exceeded");
      return;
    }
    for (auto [index, transfer] : llvm::enumerate(operationTransfers)) {
      transfer.spmBank = allocation->banks[index];
      transfer.channel = transfer.spmBank; // Bank/channel are separate fields.
      transfer.spmOffset = allocation->offsets[index];
      if (transfer.crossTaskResident && reuseResidentSlots) {
        auto lastPosition = residentLastPositions.find(transfer.buffer);
        if (lastPosition == residentLastPositions.end()) {
          status = op.emitError("missing cross-task residency endpoint");
          return;
        }
        activeResidentWindows.push_back(
            {lastPosition->second, transfer.spmBank,
             transfer.spmOffset + transfer.spmCapacity});
      } else if (transfer.crossTaskResident) {
        reservedBytes[transfer.spmBank] =
            std::max(reservedBytes[transfer.spmBank],
                     transfer.spmOffset + transfer.spmCapacity);
      }
      size_t transferIndex = workspace.transfers.size();
      if (transfer.crossTaskResident)
        residentTransfers[transfer.buffer] = transferIndex;
      workspace.transfers.push_back(std::move(transfer));
    }
    workspace.usedBytes[0] =
        std::max(workspace.usedBytes[0], allocation->highWater[0]);
    workspace.usedBytes[1] =
        std::max(workspace.usedBytes[1], allocation->highWater[1]);
  });

  if (config.enableSpmDma)
    module.walk([&](memref::CopyOp copy) {
      if (failed(status))
        return;
      auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
      if (!sourceType || sourceType.hasStaticShape())
        return;
      auto cached =
          llvm::find_if(dynamicPrefixTransfers,
                        [&](const auto &entry) { return entry.first == copy; });
      if (cached == dynamicPrefixTransfers.end())
        return;
      SmallVector<TransferPlan, 1> group;
      group.push_back(cached->second);
      auto allocation = chooseBankAllocation(
          group, ArrayRef<int64_t>(reservedBytes, 2), workspace.bankCapacity,
          workspace.target.dma.endpointAlignmentBytes, workspace.resources);
      if (!allocation) {
        status = copy.emitError("BOSCAME SPM workspace capacity exceeded");
        return;
      }
      TransferPlan &planned = group.front();
      planned.spmBank = allocation->banks.front();
      planned.channel = planned.spmBank;
      planned.spmOffset = allocation->offsets.front();
      workspace.transfers.push_back(std::move(planned));
      workspace.usedBytes[0] =
          std::max(workspace.usedBytes[0], allocation->highWater[0]);
      workspace.usedBytes[1] =
          std::max(workspace.usedBytes[1], allocation->highWater[1]);
      workspaceFunctions.insert(copy->getParentOfType<func::FuncOp>());
    });
  if (failed(status))
    return failure();

  // A serial callee is still concurrent when reached from a parallel caller.
  // Propagate workspace use through calls before checking their region scopes.
  SmallVector<CallOpInterface> calls;
  module.walk([&](CallOpInterface call) { calls.push_back(call); });
  auto mayUseWorkspace = [&](CallOpInterface call) {
    Operation *callee = call.resolveCallable();
    return workspaceFunctions.contains(callee) ||
           (!callee && isa<Value>(call.getCallableForCallee()) &&
            !workspaceFunctions.empty());
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (CallOpInterface call : calls)
      if (mayUseWorkspace(call))
        if (auto caller = call->getParentOfType<func::FuncOp>())
          changed |= workspaceFunctions.insert(caller).second;
  }
  for (CallOpInterface call : calls)
    if (mayUseWorkspace(call) && failed(verifySerialExecution(call)))
      return failure();

  module->setAttr("bosc_ame.linalg_to_boscame",
                  UnitAttr::get(module.getContext()));
  module->setAttr("bosc_ame.enable_spm_dma",
                  BoolAttr::get(module.getContext(), config.enableSpmDma));
  module->setAttr("bosc_ame.enable_ame",
                  BoolAttr::get(module.getContext(), config.enableAME));
  module->setAttr("bosc_ame.store_policy",
                  StringAttr::get(module.getContext(), config.storePolicy));
  module->setAttr(
      "bosc_ame.bm",
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), config.bm));
  module->setAttr(
      "bosc_ame.bn",
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), config.bn));
  module->setAttr(
      "bosc_ame.bk",
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), config.bk));
  if (!config.dmaAbi.empty())
    module->setAttr(spmdma::kDmaAbiAttrName,
                    StringAttr::get(module.getContext(), config.dmaAbi));
  if (!workspace.transfers.empty())
    module->setAttr(::buddy::boscame::lowering::kCpuDdrPolicy,
                    UnitAttr::get(module.getContext()));
  if (!config.target.empty())
    module->setAttr(kAmeTargetAttrName,
                    StringAttr::get(module.getContext(), config.target));
  collectDataflowFacts(module, workspace);
  collectBufferAccessFacts(module, workspace, workspace.transfers);
  collectBufferInventoryFacts(workspace);
  collectBufferAliasPairs(module, workspace, aliasContext);
  collectMemoryDependencies(workspace, aliasContext);
  refinePhysicalLifetimes(workspace, aliasContext);
  return workspace;
}
