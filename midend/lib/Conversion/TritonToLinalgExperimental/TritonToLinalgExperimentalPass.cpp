//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Analysis/LinalgComputationFacts.h"
#include "Analysis/LinalgComputationFactsPrinter.h"
#include "Conversion/StructuredToMemref/StructuredToMemref.h"
#include "Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "Conversion/TritonToStructured/TritonToStructured.h"
#include "Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/TPtr/IR/TPtrDialect.h"
#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>

using namespace mlir;
using namespace triton;

#define GEN_PASS_DECL_TRITONTOLINALGEXPERIMENTAL
#include "Conversion/TritonToLinalgExperimental/Passes.h.inc"
#define GEN_PASS_DEF_TRITONTOLINALGEXPERIMENTAL
#include "Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

static constexpr StringLiteral kZeroTailSelectPrefixAttr =
    "bosc_ame.zero_tail_select_prefix";
static constexpr StringLiteral kPrefixMaskAuxAttr = "bosc_ame.prefix_mask_aux";
static constexpr StringLiteral kPrefixRangeAttr = "bosc_ame.prefix_range";

static std::optional<int64_t> getSplatInteger(Value value) {
  Attribute valueAttr;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = constant.getValue();
  } else if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    if (fill.getInputs().size() != 1)
      return std::nullopt;
    auto constant = fill.getInputs().front().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return std::nullopt;
    valueAttr = constant.getValue();
  } else {
    return std::nullopt;
  }
  if (auto integer = dyn_cast<IntegerAttr>(valueAttr))
    return integer.getInt();
  auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!dense || !dense.isSplat() || !isa<IntegerType>(dense.getElementType()))
    return std::nullopt;
  return dense.getSplatValue<APInt>().getSExtValue();
}

static bool isSplatZero(Value value) {
  Attribute valueAttr;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = constant.getValue();
  } else if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    if (fill.getInputs().size() != 1)
      return false;
    auto constant = fill.getInputs().front().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return false;
    valueAttr = constant.getValue();
  } else {
    return false;
  }
  if (auto integer = dyn_cast<IntegerAttr>(valueAttr))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(valueAttr))
    return floating.getValue().isZero();
  auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
  if (auto integers = dyn_cast<DenseIntElementsAttr>(dense))
    return integers.isSplat() && integers.getSplatValue<APInt>().isZero();
  if (auto floats = dyn_cast<DenseFPElementsAttr>(dense))
    return floats.isSplat() && floats.getSplatValue<APFloat>().isZero();
  return false;
}

static bool hasIdentityParallelMaps(linalg::GenericOp op) {
  auto maps = op.getIndexingMapsArray();
  return op.getIteratorTypesArray().size() == 1 &&
         op.getIteratorTypesArray()[0] == utils::IteratorType::parallel &&
         maps.size() == op->getNumOperands() &&
         llvm::all_of(maps, [](AffineMap map) { return map.isIdentity(); });
}

static linalg::GenericOp getZeroBasedRange(Value value) {
  auto range = value.getDefiningOp<linalg::GenericOp>();
  if (!range || !range.getInputs().empty() || range.getOutputs().size() != 1 ||
      range.getResults().size() != 1 || range.getResult(0) != value ||
      !range.hasPureTensorSemantics() || !hasIdentityParallelMaps(range))
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 1 || type.isDynamicDim(0) ||
      type.getDimSize(0) <= 0)
    return {};
  Block &body = range.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 3)
    return {};
  auto indexOp = dyn_cast<linalg::IndexOp>(body.front());
  auto cast = dyn_cast<arith::IndexCastOp>(body.front().getNextNode());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!indexOp || indexOp.getDim() != 0 || !cast ||
      cast.getIn() != indexOp.getResult() || !cast.getType().isInteger(32) ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != cast.getResult())
    return {};
  return range;
}

static std::optional<int64_t> matchStaticPrefixPredicate(Value predicate,
                                                         Operation **rangeOp,
                                                         Operation **boundOp) {
  auto compare = predicate.getDefiningOp<linalg::GenericOp>();
  if (!compare || compare.getInputs().size() != 2 ||
      compare.getOutputs().size() != 1 || compare.getResults().size() != 1 ||
      compare.getResult(0) != predicate || !compare.hasPureTensorSemantics() ||
      !hasIdentityParallelMaps(compare))
    return std::nullopt;
  Block &body = compare.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return std::nullopt;
  auto cmp = dyn_cast<arith::CmpIOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
      cmp.getLhs() != body.getArgument(0) ||
      cmp.getRhs() != body.getArgument(1) || !yield ||
      yield.getValues().size() != 1 || yield.getValues()[0] != cmp.getResult())
    return std::nullopt;
  linalg::GenericOp range = getZeroBasedRange(compare.getInputs()[0]);
  auto predicateType = dyn_cast<RankedTensorType>(predicate.getType());
  auto rangeType =
      range ? dyn_cast<RankedTensorType>(range.getResult(0).getType())
            : RankedTensorType();
  if (!range || !predicateType || !rangeType ||
      predicateType.getShape() != rangeType.getShape())
    return std::nullopt;
  std::optional<int64_t> prefix = getSplatInteger(compare.getInputs()[1]);
  if (!prefix || *prefix <= 0 || *prefix >= rangeType.getDimSize(0))
    return std::nullopt;
  *rangeOp = range;
  if (auto fill = compare.getInputs()[1].getDefiningOp<linalg::FillOp>())
    *boundOp = fill;
  else
    *boundOp = nullptr;
  return prefix;
}

static void annotateStaticZeroTailSelects(ModuleOp module) {
  Builder builder(module.getContext());
  module.walk([&](linalg::GenericOp selectGeneric) {
    if (selectGeneric.getInputs().empty() ||
        selectGeneric.getOutputs().size() != 1 ||
        selectGeneric.getResults().size() != 1 ||
        !selectGeneric.hasPureTensorSemantics() ||
        !hasIdentityParallelMaps(selectGeneric))
      return;
    auto outputType =
        dyn_cast<RankedTensorType>(selectGeneric.getResult(0).getType());
    if (!outputType || outputType.getRank() != 1 || outputType.isDynamicDim(0))
      return;
    Block &body = selectGeneric.getRegion().front();
    if (std::distance(body.begin(), body.end()) != 2)
      return;
    auto select = dyn_cast<arith::SelectOp>(body.front());
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!select || !yield || yield.getValues().size() != 1 ||
        yield.getValues()[0] != select.getResult())
      return;
    auto condition = dyn_cast<BlockArgument>(select.getCondition());
    auto trueValue = dyn_cast<BlockArgument>(select.getTrueValue());
    auto falseValue = dyn_cast<BlockArgument>(select.getFalseValue());
    if (!condition || !trueValue || !falseValue ||
        condition.getArgNumber() >= selectGeneric.getInputs().size() ||
        trueValue.getArgNumber() >= selectGeneric.getInputs().size() ||
        falseValue.getArgNumber() >= selectGeneric.getInputs().size())
      return;
    Value falseInput = selectGeneric.getInputs()[falseValue.getArgNumber()];
    if (!isSplatZero(falseInput))
      return;
    Operation *range = nullptr;
    Operation *bound = nullptr;
    Value predicate = selectGeneric.getInputs()[condition.getArgNumber()];
    std::optional<int64_t> prefix =
        matchStaticPrefixPredicate(predicate, &range, &bound);
    if (!prefix || outputType.getDimSize(0) <= *prefix ||
        !predicate.hasOneUse() || !range->getResult(0).hasOneUse() ||
        selectGeneric.getInputs()[trueValue.getArgNumber()].getType() !=
            outputType ||
        selectGeneric.getInputs()[falseValue.getArgNumber()].getType() !=
            outputType)
      return;

    selectGeneric->setAttr(kZeroTailSelectPrefixAttr,
                           builder.getI64IntegerAttr(*prefix));
    range->setAttr(kPrefixRangeAttr, builder.getUnitAttr());
    if (Operation *predicate =
            selectGeneric.getInputs()[condition.getArgNumber()].getDefiningOp())
      predicate->setAttr(kPrefixMaskAuxAttr, builder.getUnitAttr());
    if (bound)
      bound->setAttr(kPrefixMaskAuxAttr, builder.getUnitAttr());
  });
}

struct ComputeNodeFacts {
  Operation *operation = nullptr;
  int64_t id = -1;
  int64_t stage = 0;
  bool barrierAfter = false;
  bool rematerializable = false;
  SmallVector<int64_t> predecessors;
  SmallVector<int64_t> sourcePredecessors;
  // Source lineage may contain loop-carried recurrence edges. Keep these
  // separate from predecessors, which describes the acyclic compute order.
  SmallVector<int64_t> lineagePredecessors;
  SmallVector<int64_t> inputValues;
  SmallVector<int64_t> outputValues;
  SmallVector<int64_t> sourceLoadOrigins;
  SmallVector<int64_t> varyingProgramAxes;
};

struct LaunchGridContract {
  Operation *function;
  DenseI32ArrayAttr grid;
};

static FailureOr<LaunchGridContract>
parseLaunchGridContract(ModuleOp module, StringRef specification) {
  size_t colon = specification.find(':');
  if (colon == StringRef::npos ||
      specification.find(':', colon + 1) != StringRef::npos)
    return failure();

  StringRef symbolName = specification.take_front(colon).trim();
  if (symbolName.starts_with("@"))
    symbolName = symbolName.drop_front();
  if (symbolName.empty() || symbolName.contains(' ') ||
      symbolName.contains('\t'))
    return failure();

  SmallVector<StringRef, 4> dimensions;
  specification.drop_front(colon + 1).split(dimensions, ',', -1, true);
  if (dimensions.size() != 3)
    return failure();

  SmallVector<int32_t, 3> extents;
  for (StringRef dimension : dimensions) {
    int64_t extent = 0;
    dimension = dimension.trim();
    if (dimension.empty() || dimension.getAsInteger(10, extent) ||
        extent <= 0 || extent > std::numeric_limits<int32_t>::max())
      return failure();
    extents.push_back(static_cast<int32_t>(extent));
  }

  Operation *function = SymbolTable::lookupSymbolIn(module, symbolName);
  if (!function || function->getName().getStringRef() != "tt.func")
    return failure();

  return LaunchGridContract{
      function, DenseI32ArrayAttr::get(module.getContext(), extents)};
}

static bool isLogicalValue(Type type) {
  return isa<TensorType, MemRefType>(type);
}

struct CallResultDependencies {
  SmallVector<SmallVector<unsigned, 2>, 2> formalArgumentsByResult;
};

using CallResultDependencyMap = DenseMap<Operation *, CallResultDependencies>;
using ValueFormalDependencies = DenseMap<Value, SmallVector<unsigned, 2>>;

static void appendUniqueSorted(SmallVectorImpl<unsigned> &destination,
                               ArrayRef<unsigned> values) {
  destination.append(values.begin(), values.end());
  llvm::sort(destination);
  destination.erase(std::unique(destination.begin(), destination.end()),
                    destination.end());
}

static void collectPayloadFormalDependencies(
    Value value, Block &payload, linalg::GenericOp generic,
    const ValueFormalDependencies &dependencies, DenseSet<Value> &visited,
    SmallVectorImpl<unsigned> &formalArguments) {
  if (!visited.insert(value).second)
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() == &payload) {
      unsigned operandNumber = argument.getArgNumber();
      if (operandNumber < generic->getNumOperands())
        if (auto found = dependencies.find(generic->getOperand(operandNumber));
            found != dependencies.end())
          appendUniqueSorted(formalArguments, found->second);
      return;
    }
  }
  if (Operation *def = value.getDefiningOp()) {
    if (def->getBlock() == &payload) {
      for (Value operand : def->getOperands())
        collectPayloadFormalDependencies(
            operand, payload, generic, dependencies, visited, formalArguments);
      return;
    }
  }
  if (auto found = dependencies.find(value); found != dependencies.end())
    appendUniqueSorted(formalArguments, found->second);
}

static std::optional<CallResultDependencies>
summarizePureTensorHelper(func::FuncOp function) {
  if (function.isExternal() || function.getBody().getBlocks().size() != 1)
    return std::nullopt;
  if (function.getNumResults() != 1 ||
      !isa<TensorType>(function.getResultTypes().front()))
    return std::nullopt;
  if (llvm::any_of(function.getArgumentTypes(),
                   [](Type type) { return isa<BaseMemRefType>(type); }) ||
      llvm::any_of(function.getResultTypes(),
                   [](Type type) { return isa<BaseMemRefType>(type); }))
    return std::nullopt;

  Block &body = function.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != function.getNumResults())
    return std::nullopt;

  unsigned genericCount = 0;
  for (Operation &operation : body.without_terminator()) {
    if (auto generic = dyn_cast<linalg::GenericOp>(&operation)) {
      ++genericCount;
      if (!generic.hasPureTensorSemantics() || generic.getNumResults() != 1 ||
          generic.getRegion().getBlocks().size() != 1)
        return std::nullopt;
      continue;
    }
    if (!isa<arith::ConstantOp, tensor::EmptyOp>(&operation))
      return std::nullopt;
  }
  if (genericCount == 0)
    return std::nullopt;

  bool supported = true;
  function.walk([&](Operation *op) {
    if (op == function.getOperation() ||
        isa<func::ReturnOp, linalg::YieldOp>(op))
      return;
    if (isa<func::CallOp, func::CallIndirectOp>(op) ||
        (op->getNumRegions() && !isa<linalg::GenericOp>(op)) ||
        !isMemoryEffectFree(op)) {
      supported = false;
      return;
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      if (generic->getBlock() != &body || !generic.hasPureTensorSemantics() ||
          generic.getRegion().getBlocks().size() != 1)
        supported = false;
    }
  });
  if (!supported)
    return std::nullopt;

  ValueFormalDependencies dependencies;
  for (auto [index, argument] : llvm::enumerate(body.getArguments()))
    if (isa<TensorType>(argument.getType()))
      dependencies[argument].push_back(index);

  for (Operation &operation : body.without_terminator()) {
    if (auto generic = dyn_cast<linalg::GenericOp>(&operation)) {
      Block &payload = generic.getRegion().front();
      auto yield = dyn_cast<linalg::YieldOp>(payload.getTerminator());
      if (!yield || yield.getValues().size() != generic.getNumResults())
        return std::nullopt;
      for (auto [resultNumber, result] :
           llvm::enumerate(generic.getResults())) {
        SmallVector<unsigned, 2> resultDependencies;
        DenseSet<Value> visited;
        collectPayloadFormalDependencies(yield.getValues()[resultNumber],
                                         payload, generic, dependencies,
                                         visited, resultDependencies);
        dependencies[result].assign(resultDependencies.begin(),
                                    resultDependencies.end());
      }
      continue;
    }

    SmallVector<unsigned, 2> operandDependencies;
    for (Value operand : operation.getOperands())
      if (auto found = dependencies.find(operand); found != dependencies.end())
        appendUniqueSorted(operandDependencies, found->second);
    for (Value result : operation.getResults())
      dependencies[result] = operandDependencies;
  }

  CallResultDependencies summary;
  summary.formalArgumentsByResult.resize(function.getNumResults());
  for (auto [resultNumber, returned] :
       llvm::enumerate(returnOp.getOperands())) {
    if (auto found = dependencies.find(returned); found != dependencies.end())
      summary.formalArgumentsByResult[resultNumber] = found->second;
  }
  return summary;
}

static CallResultDependencyMap summarizePureTensorHelperCalls(ModuleOp module) {
  DenseMap<Operation *, CallResultDependencies> helperSummaries;
  module.walk([&](func::FuncOp function) {
    if (auto summary = summarizePureTensorHelper(function))
      helperSummaries.try_emplace(function.getOperation(), std::move(*summary));
  });

  CallResultDependencyMap callSummaries;
  module.walk([&](func::CallOp call) {
    Operation *callee = SymbolTable::lookupSymbolIn(module, call.getCallee());
    if (auto found = helperSummaries.find(callee);
        found != helperSummaries.end())
      callSummaries.try_emplace(call.getOperation(), found->second);
  });
  return callSummaries;
}

static bool isComputeNode(Operation *op) {
  return isa<linalg::LinalgOp>(op) || op->hasAttr("bosc_ame.memory");
}

static StringRef getComputeNodeKind(Operation *op) {
  if (auto memory = op->getAttrOfType<DictionaryAttr>("bosc_ame.memory")) {
    auto kind = memory.getAs<StringAttr>("kind");
    if (kind)
      return kind.getValue() == "load" ? "memory-load" : "memory-store";
  }
  if (isa<linalg::ReduceOp>(op))
    return "reduction";
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
    return matmul->hasAttr("bosc_ame.lower_to_ame") ? "ame-matmul" : "matmul";
  if (isa<linalg::FillOp>(op))
    return "fill";
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return generic.getNumReductionLoops() ? "reduction" : "elementwise";
  return "linalg";
}

static bool isDataflowBarrier(Operation *op) {
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    if (generic.getNumReductionLoops())
      return true;
  return isa<linalg::ReduceOp, linalg::MatmulOp>(op) ||
         op->hasAttr("bosc_ame.memory");
}

static bool hasUnmodeledDataflowEffects(Operation *op) {
  if (isa<func::FuncOp, bufferization::ToTensorOp, memref::AllocOp,
          memref::AllocaOp, memref::DeallocOp>(op) ||
      isComputeNode(op))
    return false;
  if (hasUnknownEffects(op))
    return true;
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return true;
  return llvm::any_of(*effects,
                      [](const MemoryEffects::EffectInstance &effect) {
                        return isa<MemoryEffects::Read, MemoryEffects::Write>(
                            effect.getEffect());
                      });
}

static Value getDataflowStorageRoot(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  while (Operation *def = value.getDefiningOp()) {
    if (!visited.insert(def).second)
      break;
    StringRef name = def->getName().getStringRef();
    if (name == "memref.subview" || name == "memref.reinterpret_cast" ||
        name == "memref.cast" || name == "memref.transpose" ||
        name == "memref.memory_space_cast" || name == "memref.view" ||
        name == "memref.collapse_shape" || name == "memref.expand_shape" ||
        name == "bufferization.to_tensor") {
      value = def->getOperand(0);
      continue;
    }
    if (auto executeRegion = dyn_cast<scf::ExecuteRegionOp>(def)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result || !executeRegion.getRegion().hasOneBlock())
        break;
      auto yield = dyn_cast<scf::YieldOp>(
          executeRegion.getRegion().front().getTerminator());
      unsigned resultNumber = result.getResultNumber();
      if (!yield || resultNumber >= yield.getNumOperands())
        break;
      value = yield.getOperand(resultNumber);
      continue;
    }
    // SCF canonicalization can represent a result-only join as a select.
    if (auto select = dyn_cast<arith::SelectOp>(def)) {
      Value trueRoot = getDataflowStorageRoot(select.getTrueValue());
      Value falseRoot = getDataflowStorageRoot(select.getFalseValue());
      if (trueRoot != falseRoot)
        break;
      value = trueRoot;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        break;
      unsigned resultNumber = result.getResultNumber();
      Value commonRoot;
      bool hasCommonRoot = false;
      bool allPathsAgree = true;
      for (Region &region : ifOp->getRegions()) {
        if (region.empty()) {
          allPathsAgree = false;
          break;
        }
        auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
        if (!yield || resultNumber >= yield.getNumOperands()) {
          allPathsAgree = false;
          break;
        }
        Value branchRoot =
            getDataflowStorageRoot(yield.getOperand(resultNumber));
        if (!hasCommonRoot) {
          commonRoot = branchRoot;
          hasCommonRoot = true;
        } else if (branchRoot != commonRoot) {
          allPathsAgree = false;
          break;
        }
      }
      if (!allPathsAgree || !hasCommonRoot)
        break;
      value = commonRoot;
      continue;
    }
    if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(def)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        break;
      unsigned resultNumber = result.getResultNumber();
      Value commonRoot;
      bool hasCommonRoot = false;
      bool allPathsAgree = true;
      auto visitRegion = [&](Region &region) {
        if (!allPathsAgree || region.empty()) {
          allPathsAgree = false;
          return;
        }
        auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
        if (!yield || resultNumber >= yield.getNumOperands()) {
          allPathsAgree = false;
          return;
        }
        Value branchRoot =
            getDataflowStorageRoot(yield.getOperand(resultNumber));
        if (!hasCommonRoot) {
          commonRoot = branchRoot;
          hasCommonRoot = true;
        } else if (branchRoot != commonRoot) {
          allPathsAgree = false;
        }
      };
      visitRegion(switchOp.getDefaultRegion());
      for (Region &region : switchOp.getCaseRegions())
        visitRegion(region);
      if (!allPathsAgree || !hasCommonRoot)
        break;
      value = commonRoot;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      if (!isa<MemRefType>(value.getType()))
        break;
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        break;
      unsigned index = result.getResultNumber();
      auto yieldedValues = forOp.getYieldedValues();
      auto iterArgs = forOp.getRegionIterArgs();
      if (index >= forOp.getInitArgs().size() ||
          index >= yieldedValues.size() || index >= iterArgs.size())
        break;
      Value initRoot = getDataflowStorageRoot(forOp.getInitArgs()[index]);
      Value yieldedRoot = getDataflowStorageRoot(yieldedValues[index]);
      if (!initRoot ||
          (yieldedRoot != iterArgs[index] && yieldedRoot != initRoot))
        break;
      value = initRoot;
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
      if (!isa<MemRefType>(value.getType()))
        break;
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        break;
      unsigned index = result.getResultNumber();
      auto conditionArgs = whileOp.getConditionOp().getArgs();
      auto yieldedValues = whileOp.getYieldOp().getOperands();
      auto beforeArgs = whileOp.getBeforeArguments();
      auto afterArgs = whileOp.getAfterArguments();
      if (index >= whileOp.getInits().size() || index >= conditionArgs.size() ||
          index >= yieldedValues.size() || index >= beforeArgs.size() ||
          index >= afterArgs.size())
        break;
      Value initRoot = getDataflowStorageRoot(whileOp.getInits()[index]);
      Value conditionRoot = getDataflowStorageRoot(conditionArgs[index]);
      Value yieldedRoot = getDataflowStorageRoot(yieldedValues[index]);
      if (!initRoot ||
          (conditionRoot != beforeArgs[index] && conditionRoot != initRoot) ||
          (yieldedRoot != afterArgs[index] && yieldedRoot != initRoot))
        break;
      value = initRoot;
      continue;
    }
    break;
  }
  return value;
}

using ReachingWriterMap = DenseMap<Value, SmallVector<int64_t, 2>>;
using TensorProducerMap = DenseMap<Value, SmallVector<int64_t, 2>>;

static void mergeReachingWriters(ReachingWriterMap &destination,
                                 const ReachingWriterMap &source) {
  for (const auto &entry : source) {
    auto &writers = destination[entry.first];
    writers.append(entry.second.begin(), entry.second.end());
    llvm::sort(writers);
    writers.erase(std::unique(writers.begin(), writers.end()), writers.end());
  }
}

static bool reachingWritersEqual(const ReachingWriterMap &lhs,
                                 const ReachingWriterMap &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const auto &entry : lhs) {
    auto found = rhs.find(entry.first);
    if (found == rhs.end() || entry.second != found->second)
      return false;
  }
  return true;
}

static void mergeTensorProducers(TensorProducerMap &destination,
                                 const TensorProducerMap &source) {
  for (const auto &entry : source) {
    auto &producers = destination[entry.first];
    producers.append(entry.second.begin(), entry.second.end());
    llvm::sort(producers);
    producers.erase(std::unique(producers.begin(), producers.end()),
                    producers.end());
  }
}

static void normalizeIds(SmallVectorImpl<int64_t> &ids) {
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

static void appendUniqueSorted(SmallVectorImpl<int64_t> &destination,
                               ArrayRef<int64_t> values) {
  destination.append(values.begin(), values.end());
  normalizeIds(destination);
}

static void
collectComputeProducers(Value value, Operation *consumer,
                        const DenseMap<Operation *, int64_t> &nodeIds,
                        const ReachingWriterMap &lastWriter,
                        const TensorProducerMap &tensorProducers,
                        const CallResultDependencyMap &callDependencies,
                        DenseSet<Value> &visited,
                        SmallVectorImpl<int64_t> &predecessors) {
  if (!visited.insert(value).second)
    return;
  // Mutable storage is described by the writers that reach this read. In
  // particular, following a to_buffer's original tensor operand after a full
  // overwrite would resurrect producers whose contents have been replaced.
  if (isa<MemRefType>(value.getType())) {
    Value root = getDataflowStorageRoot(value);
    if (auto writer = lastWriter.find(root); writer != lastWriter.end()) {
      predecessors.append(writer->second.begin(), writer->second.end());
      return;
    }
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto forOp =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (forOp) {
      for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (iterArg != value || index >= forOp.getInitArgs().size())
          continue;
        collectComputeProducers(forOp.getInitArgs()[index], consumer, nodeIds,
                                lastWriter, tensorProducers, callDependencies,
                                visited, predecessors);
        return;
      }
      return;
    }
    auto whileOp =
        dyn_cast_or_null<scf::WhileOp>(argument.getOwner()->getParentOp());
    if (!whileOp)
      return;
    unsigned index = argument.getArgNumber();
    if (argument.getOwner() == whileOp.getBeforeBody()) {
      if (index < whileOp.getInits().size())
        collectComputeProducers(whileOp.getInits()[index], consumer, nodeIds,
                                lastWriter, tensorProducers, callDependencies,
                                visited, predecessors);
    } else if (argument.getOwner() == whileOp.getAfterBody()) {
      auto conditionArgs = whileOp.getConditionOp().getArgs();
      if (index < conditionArgs.size())
        collectComputeProducers(conditionArgs[index], consumer, nodeIds,
                                lastWriter, tensorProducers, callDependencies,
                                visited, predecessors);
    }
    return;
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return;
  if (auto executeRegion = dyn_cast<scf::ExecuteRegionOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result || !executeRegion.getRegion().hasOneBlock())
      return;
    auto yield = dyn_cast<scf::YieldOp>(
        executeRegion.getRegion().front().getTerminator());
    unsigned resultNumber = result.getResultNumber();
    if (yield && resultNumber < yield.getNumOperands())
      collectComputeProducers(yield.getOperand(resultNumber), consumer, nodeIds,
                              lastWriter, tensorProducers, callDependencies,
                              visited, predecessors);
    return;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    auto result = cast<OpResult>(value);
    unsigned resultNumber = result.getResultNumber();
    collectComputeProducers(ifOp.getCondition(), consumer, nodeIds, lastWriter,
                            tensorProducers, callDependencies, visited,
                            predecessors);
    for (Region &region : ifOp->getRegions()) {
      if (region.empty())
        continue;
      Operation *terminator = region.front().getTerminator();
      if (!isa<scf::YieldOp>(terminator) ||
          resultNumber >= terminator->getNumOperands())
        continue;
      collectComputeProducers(terminator->getOperand(resultNumber), consumer,
                              nodeIds, lastWriter, tensorProducers,
                              callDependencies, visited, predecessors);
    }
    return;
  }
  if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(def)) {
    auto result = cast<OpResult>(value);
    unsigned resultNumber = result.getResultNumber();
    collectComputeProducers(switchOp.getArg(), consumer, nodeIds, lastWriter,
                            tensorProducers, callDependencies, visited,
                            predecessors);
    auto visitRegion = [&](Region &region) {
      if (region.empty())
        return;
      Operation *terminator = region.front().getTerminator();
      if (!isa<scf::YieldOp>(terminator) ||
          resultNumber >= terminator->getNumOperands())
        return;
      collectComputeProducers(terminator->getOperand(resultNumber), consumer,
                              nodeIds, lastWriter, tensorProducers,
                              callDependencies, visited, predecessors);
    };
    visitRegion(switchOp.getDefaultRegion());
    for (Region &region : switchOp.getCaseRegions())
      visitRegion(region);
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return;
    unsigned resultNumber = result.getResultNumber();
    collectComputeProducers(forOp.getLowerBound(), consumer, nodeIds,
                            lastWriter, tensorProducers, callDependencies,
                            visited, predecessors);
    collectComputeProducers(forOp.getUpperBound(), consumer, nodeIds,
                            lastWriter, tensorProducers, callDependencies,
                            visited, predecessors);
    collectComputeProducers(forOp.getStep(), consumer, nodeIds, lastWriter,
                            tensorProducers, callDependencies, visited,
                            predecessors);
    if (resultNumber < forOp.getInitArgs().size())
      collectComputeProducers(forOp.getInitArgs()[resultNumber], consumer,
                              nodeIds, lastWriter, tensorProducers,
                              callDependencies, visited, predecessors);
    if (resultNumber < forOp.getYieldedValues().size())
      collectComputeProducers(forOp.getYieldedValues()[resultNumber], consumer,
                              nodeIds, lastWriter, tensorProducers,
                              callDependencies, visited, predecessors);
    return;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return;
    unsigned index = result.getResultNumber();
    collectComputeProducers(whileOp.getConditionOp().getCondition(), consumer,
                            nodeIds, lastWriter, tensorProducers,
                            callDependencies, visited, predecessors);
    auto conditionArgs = whileOp.getConditionOp().getArgs();
    if (index < conditionArgs.size())
      collectComputeProducers(conditionArgs[index], consumer, nodeIds,
                              lastWriter, tensorProducers, callDependencies,
                              visited, predecessors);
    return;
  }
  if (auto call = dyn_cast<func::CallOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    auto summary = callDependencies.find(def);
    if (result && summary != callDependencies.end() &&
        result.getResultNumber() <
            summary->second.formalArgumentsByResult.size()) {
      for (unsigned formalArgument :
           summary->second.formalArgumentsByResult[result.getResultNumber()])
        if (formalArgument < call.getNumOperands())
          collectComputeProducers(call.getOperand(formalArgument), consumer,
                                  nodeIds, lastWriter, tensorProducers,
                                  callDependencies, visited, predecessors);
      return;
    }
    for (Value operand : call.getOperands())
      collectComputeProducers(operand, consumer, nodeIds, lastWriter,
                              tensorProducers, callDependencies, visited,
                              predecessors);
    return;
  }
  if (def->getName().getStringRef() == "bufferization.to_tensor" &&
      def->getNumOperands() == 1 &&
      isa<MemRefType>(def->getOperand(0).getType())) {
    if (auto producers = tensorProducers.find(value);
        producers != tensorProducers.end()) {
      predecessors.append(producers->second.begin(), producers->second.end());
      return;
    }
    Value root = getDataflowStorageRoot(def->getOperand(0));
    if (auto writer = lastWriter.find(root); writer != lastWriter.end()) {
      predecessors.append(writer->second.begin(), writer->second.end());
      return;
    }
  }
  auto node = nodeIds.find(def);
  if (node != nodeIds.end()) {
    if (def != consumer)
      predecessors.push_back(node->second);
    return;
  }
  for (Value operand : def->getOperands())
    collectComputeProducers(operand, consumer, nodeIds, lastWriter,
                            tensorProducers, callDependencies, visited,
                            predecessors);
}

// Collect data-source lineage independently of compute order, including
// loop-carried tensor producers and writers captured during loop replay.
static void
collectLoopLineageProducers(Value value, Operation *consumer,
                            const DenseMap<Operation *, int64_t> &nodeIds,
                            const ReachingWriterMap &lastWriter,
                            const TensorProducerMap &tensorProducers,
                            const CallResultDependencyMap &callDependencies,
                            DenseSet<Value> &visited,
                            SmallVectorImpl<int64_t> &predecessors) {
  if (!visited.insert(value).second)
    return;

  if (isa<MemRefType>(value.getType())) {
    Value root = getDataflowStorageRoot(value);
    if (auto writer = lastWriter.find(root); writer != lastWriter.end()) {
      appendUniqueSorted(predecessors, writer->second);
      return;
    }
  }

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto forOp =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (forOp) {
      for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (iterArg != value || index >= forOp.getInitArgs().size() ||
            index >= forOp.getYieldedValues().size())
          continue;
        collectLoopLineageProducers(forOp.getInitArgs()[index], consumer,
                                    nodeIds, lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
        collectLoopLineageProducers(forOp.getYieldedValues()[index], consumer,
                                    nodeIds, lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
        return;
      }
      return;
    }
    auto whileOp =
        dyn_cast_or_null<scf::WhileOp>(argument.getOwner()->getParentOp());
    if (!whileOp)
      return;
    unsigned index = argument.getArgNumber();
    if (argument.getOwner() == whileOp.getBeforeBody()) {
      if (index < whileOp.getInits().size())
        collectLoopLineageProducers(whileOp.getInits()[index], consumer,
                                    nodeIds, lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
      auto yieldedValues = whileOp.getYieldOp().getOperands();
      if (index < yieldedValues.size())
        collectLoopLineageProducers(yieldedValues[index], consumer, nodeIds,
                                    lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
    } else if (argument.getOwner() == whileOp.getAfterBody()) {
      auto conditionArgs = whileOp.getConditionOp().getArgs();
      if (index < conditionArgs.size())
        collectLoopLineageProducers(conditionArgs[index], consumer, nodeIds,
                                    lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
    }
    return;
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return;
  if (auto executeRegion = dyn_cast<scf::ExecuteRegionOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result || !executeRegion.getRegion().hasOneBlock())
      return;
    auto yield = dyn_cast<scf::YieldOp>(
        executeRegion.getRegion().front().getTerminator());
    unsigned resultNumber = result.getResultNumber();
    if (yield && resultNumber < yield.getNumOperands())
      collectLoopLineageProducers(yield.getOperand(resultNumber), consumer,
                                  nodeIds, lastWriter, tensorProducers,
                                  callDependencies, visited, predecessors);
    return;
  }
  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      unsigned index = result.getResultNumber();
      collectLoopLineageProducers(forOp.getLowerBound(), consumer, nodeIds,
                                  lastWriter, tensorProducers, callDependencies,
                                  visited, predecessors);
      collectLoopLineageProducers(forOp.getUpperBound(), consumer, nodeIds,
                                  lastWriter, tensorProducers, callDependencies,
                                  visited, predecessors);
      collectLoopLineageProducers(forOp.getStep(), consumer, nodeIds,
                                  lastWriter, tensorProducers, callDependencies,
                                  visited, predecessors);
      if (index < forOp.getInitArgs().size())
        collectLoopLineageProducers(forOp.getInitArgs()[index], consumer,
                                    nodeIds, lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
      if (index < forOp.getYieldedValues().size())
        collectLoopLineageProducers(forOp.getYieldedValues()[index], consumer,
                                    nodeIds, lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
      return;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        return;
      unsigned index = result.getResultNumber();
      collectLoopLineageProducers(
          whileOp.getConditionOp().getCondition(), consumer, nodeIds,
          lastWriter, tensorProducers, callDependencies, visited, predecessors);
      auto conditionArgs = whileOp.getConditionOp().getArgs();
      if (index < conditionArgs.size())
        collectLoopLineageProducers(conditionArgs[index], consumer, nodeIds,
                                    lastWriter, tensorProducers,
                                    callDependencies, visited, predecessors);
      return;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      unsigned index = result.getResultNumber();
      collectLoopLineageProducers(ifOp.getCondition(), consumer, nodeIds,
                                  lastWriter, tensorProducers, callDependencies,
                                  visited, predecessors);
      for (Region &region : ifOp->getRegions()) {
        if (region.empty())
          continue;
        Operation *terminator = region.front().getTerminator();
        if (isa<scf::YieldOp>(terminator) &&
            index < terminator->getNumOperands())
          collectLoopLineageProducers(terminator->getOperand(index), consumer,
                                      nodeIds, lastWriter, tensorProducers,
                                      callDependencies, visited, predecessors);
      }
      return;
    }
    if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(def)) {
      unsigned index = result.getResultNumber();
      collectLoopLineageProducers(switchOp.getArg(), consumer, nodeIds,
                                  lastWriter, tensorProducers, callDependencies,
                                  visited, predecessors);
      auto visitRegion = [&](Region &region) {
        if (region.empty())
          return;
        Operation *terminator = region.front().getTerminator();
        if (isa<scf::YieldOp>(terminator) &&
            index < terminator->getNumOperands())
          collectLoopLineageProducers(terminator->getOperand(index), consumer,
                                      nodeIds, lastWriter, tensorProducers,
                                      callDependencies, visited, predecessors);
      };
      visitRegion(switchOp.getDefaultRegion());
      for (Region &region : switchOp.getCaseRegions())
        visitRegion(region);
      return;
    }
  }
  if (def->getName().getStringRef() == "bufferization.to_tensor") {
    if (auto producers = tensorProducers.find(value);
        producers != tensorProducers.end()) {
      appendUniqueSorted(predecessors, producers->second);
    }
    // An as-yet unevaluated bridge is bottom, not a read of the mutable
    // buffer at the consumer's program point. Loop replay will capture it.
    return;
  }
  if (auto call = dyn_cast<func::CallOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    auto summary = callDependencies.find(def);
    if (result && summary != callDependencies.end() &&
        result.getResultNumber() <
            summary->second.formalArgumentsByResult.size()) {
      for (unsigned formalArgument :
           summary->second.formalArgumentsByResult[result.getResultNumber()])
        if (formalArgument < call.getNumOperands())
          collectLoopLineageProducers(call.getOperand(formalArgument), consumer,
                                      nodeIds, lastWriter, tensorProducers,
                                      callDependencies, visited, predecessors);
      return;
    }
    for (Value operand : call.getOperands())
      collectLoopLineageProducers(operand, consumer, nodeIds, lastWriter,
                                  tensorProducers, callDependencies, visited,
                                  predecessors);
    return;
  }
  if (auto node = nodeIds.find(def); node != nodeIds.end()) {
    if (def != consumer)
      predecessors.push_back(node->second);
    return;
  }
  for (Value operand : def->getOperands())
    collectLoopLineageProducers(operand, consumer, nodeIds, lastWriter,
                                tensorProducers, callDependencies, visited,
                                predecessors);
}

static void collectComputeConsumers(
    Value value, const DenseMap<Operation *, int64_t> &nodeIds,
    const CallResultDependencyMap &callDependencies, DenseSet<Value> &visited,
    SmallVectorImpl<int64_t> &consumers) {
  if (!visited.insert(value).second)
    return;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto node = nodeIds.find(user); node != nodeIds.end()) {
      consumers.push_back(node->second);
      continue;
    }

    if (auto call = dyn_cast<func::CallOp>(user)) {
      auto summary = callDependencies.find(user);
      if (summary == callDependencies.end())
        continue;
      unsigned formalArgument = use.getOperandNumber();
      for (auto [resultNumber, dependencies] :
           llvm::enumerate(summary->second.formalArgumentsByResult)) {
        if (!llvm::is_contained(dependencies, formalArgument) ||
            resultNumber >= call.getNumResults())
          continue;
        collectComputeConsumers(call.getResult(resultNumber), nodeIds,
                                callDependencies, visited, consumers);
      }
      continue;
    }

    // A value yielded from a structured region reaches the corresponding
    // parent result. Follow every possible path so reverse users remain a
    // conservative union even when the condition is not known here.
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      Region *region = yield->getParentRegion();
      Operation *parent = region ? region->getParentOp() : nullptr;
      unsigned slot = use.getOperandNumber();
      if (auto ifOp = dyn_cast_or_null<scf::IfOp>(parent)) {
        if (slot < ifOp.getNumResults())
          collectComputeConsumers(ifOp.getResult(slot), nodeIds,
                                  callDependencies, visited, consumers);
        continue;
      }
      if (auto switchOp = dyn_cast_or_null<scf::IndexSwitchOp>(parent)) {
        if (slot < switchOp.getNumResults())
          collectComputeConsumers(switchOp.getResult(slot), nodeIds,
                                  callDependencies, visited, consumers);
        continue;
      }
      if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
        if (slot < forOp.getNumResults()) {
          collectComputeConsumers(forOp.getResult(slot), nodeIds,
                                  callDependencies, visited, consumers);
          if (slot < forOp.getRegionIterArgs().size())
            collectComputeConsumers(forOp.getRegionIterArgs()[slot], nodeIds,
                                    callDependencies, visited, consumers);
        }
        continue;
      }
      if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(parent)) {
        // scf.while's after yield feeds the next before-region iteration.
        // Follow that block argument; the condition terminator maps it to the
        // loop result when the loop exits.
        if (yield->getBlock() == whileOp.getAfterBody() &&
            slot < whileOp.getBeforeArguments().size())
          collectComputeConsumers(whileOp.getBeforeArguments()[slot], nodeIds,
                                  callDependencies, visited, consumers);
        continue;
      }
      if (auto executeRegion = dyn_cast_or_null<scf::ExecuteRegionOp>(parent)) {
        if (slot < executeRegion.getNumResults())
          collectComputeConsumers(executeRegion.getResult(slot), nodeIds,
                                  callDependencies, visited, consumers);
        continue;
      }
      continue;
    }

    if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
      // Operand zero is the condition itself. Only forwarded values (the
      // remaining operands) have logical consumers through the loop result or
      // after-region argument.
      unsigned operandNumber = use.getOperandNumber();
      if (operandNumber == 0)
        continue;
      unsigned slot = operandNumber - 1;
      auto whileOp = dyn_cast<scf::WhileOp>(condition->getParentOp());
      if (!whileOp)
        continue;
      if (slot < whileOp.getNumResults())
        collectComputeConsumers(whileOp.getResult(slot), nodeIds,
                                callDependencies, visited, consumers);
      if (slot < whileOp.getAfterArguments().size())
        collectComputeConsumers(whileOp.getAfterArguments()[slot], nodeIds,
                                callDependencies, visited, consumers);
      continue;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [slot, init] : llvm::enumerate(forOp.getInitArgs())) {
        if (init != value)
          continue;
        if (slot < forOp.getNumResults())
          collectComputeConsumers(forOp.getResult(slot), nodeIds,
                                  callDependencies, visited, consumers);
        if (slot < forOp.getRegionIterArgs().size())
          collectComputeConsumers(forOp.getRegionIterArgs()[slot], nodeIds,
                                  callDependencies, visited, consumers);
      }
      continue;
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      for (auto [slot, init] : llvm::enumerate(whileOp.getInits())) {
        if (init != value)
          continue;
        if (slot < whileOp.getNumResults())
          collectComputeConsumers(whileOp.getResult(slot), nodeIds,
                                  callDependencies, visited, consumers);
        if (slot < whileOp.getBeforeArguments().size())
          collectComputeConsumers(whileOp.getBeforeArguments()[slot], nodeIds,
                                  callDependencies, visited, consumers);
      }
      continue;
    }

    if (auto select = dyn_cast<arith::SelectOp>(user)) {
      if (select.getCondition() == value || select.getTrueValue() == value ||
          select.getFalseValue() == value)
        collectComputeConsumers(select.getResult(), nodeIds, callDependencies,
                                visited, consumers);
      continue;
    }

    if (user->getName().getDialectNamespace() == "arith" &&
        user->getNumRegions() == 0) {
      // Regionless arith ops are pure SSA forwarding steps for dependency
      // discovery; each result may depend on the used operand.
      for (Value result : user->getResults())
        collectComputeConsumers(result, nodeIds, callDependencies, visited,
                                consumers);
      continue;
    }

    StringRef name = user->getName().getStringRef();
    if (name == "tensor.extract" && use.getOperandNumber() == 0 &&
        user->getNumResults() == 1) {
      // The extracted scalar can feed a Linalg scalar input such as fill.
      collectComputeConsumers(user->getResult(0), nodeIds, callDependencies,
                              visited, consumers);
      continue;
    }
    if ((name == "tensor.concat" || name == "tensor.insert_slice") &&
        user->getNumResults() == 1 &&
        isLogicalValue(user->getResult(0).getType())) {
      // Every tensor input contributes to the result. Preserve reverse
      // consumer edges even when the value is not the first operand. The
      // canonicalizer commonly lowers tensor.concat to insert_slice.
      collectComputeConsumers(user->getResult(0), nodeIds, callDependencies,
                              visited, consumers);
      continue;
    }
    if ((name == "tensor.insert" && use.getOperandNumber() == 1) ||
        name == "tensor.scatter") {
      if (user->getNumResults() != 1 || !isLogicalValue(value.getType()) ||
          !isLogicalValue(user->getResult(0).getType()))
        continue;
      // tensor.insert preserves destination elements outside its index;
      // every tensor operand of scatter contributes values or coordinates.
      collectComputeConsumers(user->getResult(0), nodeIds, callDependencies,
                              visited, consumers);
      continue;
    }
    bool isValueForwarder =
        name == "bufferization.to_buffer" ||
        name == "bufferization.to_tensor" || name == "memref.cast" ||
        name == "memref.subview" || name == "memref.reinterpret_cast" ||
        name == "memref.transpose" || name == "memref.memory_space_cast" ||
        name == "memref.view" || name == "memref.collapse_shape" ||
        name == "memref.expand_shape" || name == "memref.reshape" ||
        name == "tensor.cast" || name == "tensor.bitcast" ||
        name == "tensor.extract_slice" || name == "tensor.collapse_shape" ||
        name == "tensor.expand_shape" || name == "tensor.reshape" ||
        name == "tensor.pad";
    if (!isValueForwarder || user->getNumOperands() == 0 ||
        user->getOperand(0) != value || user->getNumResults() != 1 ||
        !isLogicalValue(user->getResult(0).getType()))
      continue;
    collectComputeConsumers(user->getResult(0), nodeIds, callDependencies,
                            visited, consumers);
  }
}

static void collectProgramAxes(Value value, func::FuncOp function,
                               const DenseMap<Operation *, int64_t> &nodeIds,
                               DenseSet<Value> &visited,
                               DenseSet<int64_t> &axes) {
  if (!visited.insert(value).second)
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() != &function.getBody().front()) {
      Operation *parent = argument.getOwner()->getParentOp();
      if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
        unsigned index = argument.getArgNumber();
        if (index == 0) {
          collectProgramAxes(forOp.getLowerBound(), function, nodeIds, visited,
                             axes);
          collectProgramAxes(forOp.getUpperBound(), function, nodeIds, visited,
                             axes);
          collectProgramAxes(forOp.getStep(), function, nodeIds, visited, axes);
        } else {
          index -= 1;
          if (index < forOp.getInitArgs().size()) {
            collectProgramAxes(forOp.getInitArgs()[index], function, nodeIds,
                               visited, axes);
            auto yieldedValues = forOp.getYieldedValues();
            if (index < yieldedValues.size())
              collectProgramAxes(yieldedValues[index], function, nodeIds,
                                 visited, axes);
          }
        }
      } else if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(parent)) {
        unsigned index = argument.getArgNumber();
        if (argument.getOwner() == whileOp.getBeforeBody()) {
          if (index < whileOp.getInits().size()) {
            collectProgramAxes(whileOp.getInits()[index], function, nodeIds,
                               visited, axes);
            auto yieldedValues = whileOp.getYieldOp().getOperands();
            if (index < yieldedValues.size())
              collectProgramAxes(yieldedValues[index], function, nodeIds,
                                 visited, axes);
          }
        } else if (argument.getOwner() == whileOp.getAfterBody()) {
          auto conditionArgs = whileOp.getConditionOp().getArgs();
          if (index < conditionArgs.size())
            collectProgramAxes(conditionArgs[index], function, nodeIds, visited,
                               axes);
          collectProgramAxes(whileOp.getConditionOp().getCondition(), function,
                             nodeIds, visited, axes);
        }
      }
      return;
    }
    unsigned index = argument.getArgNumber();
    auto role =
        function.getArgAttrOfType<StringAttr>(index, "bosc_ame.program_role");
    auto axis =
        function.getArgAttrOfType<IntegerAttr>(index, "bosc_ame.program_axis");
    if (role && role.getValue() == "program_id" && axis)
      axes.insert(axis.getInt());
    return;
  }
  Operation *def = value.getDefiningOp();
  if (!def)
    return;

  // Usually node axes are inherited through the acyclic predecessor map. A
  // loop-carried value can, however, first become varying on the backedge,
  // which is intentionally absent from that map. Walk a compute node's
  // operands here as well; the value guard above prevents recurrence cycles.
  if (nodeIds.count(def)) {
    for (Value operand : def->getOperands())
      collectProgramAxes(operand, function, nodeIds, visited, axes);
    return;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    collectProgramAxes(ifOp.getCondition(), function, nodeIds, visited, axes);
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return;
    unsigned resultNumber = result.getResultNumber();
    for (Region &region : ifOp->getRegions()) {
      if (region.empty())
        continue;
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (yield && resultNumber < yield.getNumOperands())
        collectProgramAxes(yield.getOperand(resultNumber), function, nodeIds,
                           visited, axes);
    }
    return;
  }

  if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(def)) {
    collectProgramAxes(switchOp.getArg(), function, nodeIds, visited, axes);
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return;
    unsigned resultNumber = result.getResultNumber();
    auto visitRegion = [&](Region &region) {
      if (region.empty())
        return;
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (yield && resultNumber < yield.getNumOperands())
        collectProgramAxes(yield.getOperand(resultNumber), function, nodeIds,
                           visited, axes);
    };
    visitRegion(switchOp.getDefaultRegion());
    for (Region &region : switchOp.getCaseRegions())
      visitRegion(region);
    return;
  }

  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    collectProgramAxes(forOp.getLowerBound(), function, nodeIds, visited, axes);
    collectProgramAxes(forOp.getUpperBound(), function, nodeIds, visited, axes);
    collectProgramAxes(forOp.getStep(), function, nodeIds, visited, axes);
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return;
    unsigned resultNumber = result.getResultNumber();
    if (resultNumber < forOp.getInitArgs().size())
      collectProgramAxes(forOp.getInitArgs()[resultNumber], function, nodeIds,
                         visited, axes);
    if (resultNumber < forOp.getYieldedValues().size())
      collectProgramAxes(forOp.getYieldedValues()[resultNumber], function,
                         nodeIds, visited, axes);
    return;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
    for (Value init : whileOp.getInits())
      collectProgramAxes(init, function, nodeIds, visited, axes);
    collectProgramAxes(whileOp.getConditionOp().getCondition(), function,
                       nodeIds, visited, axes);
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return;
    unsigned resultNumber = result.getResultNumber();
    auto conditionArgs = whileOp.getConditionOp().getArgs();
    if (resultNumber < conditionArgs.size())
      collectProgramAxes(conditionArgs[resultNumber], function, nodeIds,
                         visited, axes);
    auto yieldedValues = whileOp.getYieldOp().getOperands();
    if (resultNumber < yieldedValues.size())
      collectProgramAxes(yieldedValues[resultNumber], function, nodeIds,
                         visited, axes);
    return;
  }

  if (auto executeRegion = dyn_cast<scf::ExecuteRegionOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result || !executeRegion.getRegion().hasOneBlock())
      return;
    auto yield = dyn_cast<scf::YieldOp>(
        executeRegion.getRegion().front().getTerminator());
    unsigned resultNumber = result.getResultNumber();
    if (yield && resultNumber < yield.getNumOperands())
      collectProgramAxes(yield.getOperand(resultNumber), function, nodeIds,
                         visited, axes);
    return;
  }

  // Result lineage excludes shape-only dependencies, so call axes stay
  // conservative across every actual argument.
  for (Value operand : def->getOperands())
    collectProgramAxes(operand, function, nodeIds, visited, axes);
}

static void getReadWriteValues(Operation *op, SmallVectorImpl<Value> &reads,
                               SmallVectorImpl<Value> &writes) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    for (Value input : linalgOp.getDpsInputs())
      reads.push_back(input);
    for (OpOperand &output : linalgOp.getDpsInitsMutable()) {
      if (linalgOp.payloadUsesValueFromOperand(&output))
        reads.push_back(output.get());
      writes.push_back(output.get());
    }
    return;
  }

  StringRef name = op->getName().getStringRef();
  if (name == "memref.copy" && op->getNumOperands() >= 2) {
    reads.push_back(op->getOperand(0));
    writes.push_back(op->getOperand(1));
  } else if (name == "memref.load" && op->getNumOperands() >= 1) {
    reads.push_back(op->getOperand(0));
  } else if (name == "memref.store" && op->getNumOperands() >= 2) {
    reads.push_back(op->getOperand(0));
    writes.push_back(op->getOperand(1));
  }
}

static bool hasUnmodeledArgumentAliases(ModuleOp module, func::FuncOp function,
                                        ArrayRef<ComputeNodeFacts> nodes) {
  struct ArgumentAccess {
    bool read = false;
    bool write = false;
  };
  DenseMap<Value, ArgumentAccess> argumentAccesses;
  auto recordAccess = [&](Value value, bool write) {
    Value root = getDataflowStorageRoot(value);
    auto argument = dyn_cast<BlockArgument>(root);
    if (!argument || argument.getOwner() != &function.getBody().front() ||
        !isa<BaseMemRefType>(root.getType()))
      return;
    auto &access = argumentAccesses[root];
    access.read |= !write;
    access.write |= write;
  };
  for (const ComputeNodeFacts &node : nodes) {
    SmallVector<Value> reads, writes;
    getReadWriteValues(node.operation, reads, writes);
    for (Value read : reads)
      recordAccess(read, false);
    for (Value write : writes)
      recordAccess(write, true);
  }
  if (argumentAccesses.size() < 2 ||
      llvm::none_of(argumentAccesses,
                    [](const auto &entry) { return entry.second.write; }))
    return false;

  // The module ABI describes external entry callers. A helper's distinct
  // formals can receive the same actual buffer, even if the helper is public.
  using namespace buddy::boscame::spmdma;
  auto abi = module->getAttrOfType<StringAttr>(kDmaAbiAttrName);
  bool abiEntry = abi && abi.getValue() == kDmaAbiAlignedNonAliasing &&
                  function.isPublic() &&
                  SymbolTable::symbolKnownUseEmpty(function.getOperation(),
                                                   module.getOperation());
  return !abiEntry || llvm::any_of(argumentAccesses, [](const auto &entry) {
    return bool(cast<BaseMemRefType>(entry.first.getType()).getMemorySpace());
  });
}

/// Preserve tensor-form use-def facts before bufferization erases tensor SSA.
/// IDs are deterministic for a fixed module traversal, but are intentionally
/// not stable across source or operation-order changes.
static void annotateComputeGraph(ModuleOp module) {
  MLIRContext *context = module.getContext();
  int64_t nextNodeId = 0;
  int64_t nextValueId = 0;
  CallResultDependencyMap callDependencies =
      summarizePureTensorHelperCalls(module);

  module.walk([&](func::FuncOp function) {
    if (function.isExternal())
      return;

    struct LoopCarriedValueFacts {
      Operation *loop = nullptr;
      unsigned slot = 0;
      Value initValue;
      Value iterArgValue;
      Value conditionValue;
      Value afterArgValue;
      Value yieldedValue;
      Value resultValue;
      int64_t initValueId = -1;
      int64_t iterArgValueId = -1;
      int64_t conditionValueId = -1;
      int64_t afterArgValueId = -1;
      int64_t yieldedValueId = -1;
      int64_t resultValueId = -1;
    };

    DenseMap<Value, int64_t> valueIds;
    SmallVector<LoopCarriedValueFacts> loopCarriedValues;
    auto assignValueId = [&](Value value) {
      if (!isLogicalValue(value.getType()))
        return int64_t(-1);
      auto inserted = valueIds.try_emplace(value, nextValueId);
      if (inserted.second)
        ++nextValueId;
      return inserted.first->second;
    };
    auto getValueId = [&](Value value) {
      auto found = valueIds.find(value);
      return found == valueIds.end() ? int64_t(-1) : found->second;
    };

    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      Value argument = function.getArgument(index);
      int64_t valueId = assignValueId(argument);
      if (valueId >= 0)
        function.setArgAttr(
            index, "bosc_ame.logical_value_id",
            IntegerAttr::get(IntegerType::get(context, 64), valueId));
    }
    function.walk([&](Operation *op) {
      SmallVector<int64_t> resultIds(op->getNumResults(), -1);
      bool hasLogicalResults = false;
      for (auto [index, result] : llvm::enumerate(op->getResults())) {
        resultIds[index] = assignValueId(result);
        hasLogicalResults |= resultIds[index] >= 0;
      }
      if (hasLogicalResults)
        op->setAttr("bosc_ame.logical_value_ids",
                    DenseI64ArrayAttr::get(context, resultIds));
    });
    function.walk([&](scf::ForOp forOp) {
      unsigned slotCount =
          std::min({static_cast<unsigned>(forOp.getInitArgs().size()),
                    static_cast<unsigned>(forOp.getRegionIterArgs().size()),
                    static_cast<unsigned>(forOp.getYieldedValues().size()),
                    static_cast<unsigned>(forOp.getNumResults())});
      for (unsigned slot = 0; slot < slotCount; ++slot) {
        Value initValue = forOp.getInitArgs()[slot];
        Value iterArgValue = forOp.getRegionIterArgs()[slot];
        Value yieldedValue = forOp.getYieldedValues()[slot];
        Value resultValue = forOp.getResult(slot);
        if (!isLogicalValue(iterArgValue.getType()))
          continue;

        int64_t iterArgValueId = assignValueId(iterArgValue);
        loopCarriedValues.push_back(
            {forOp, slot, initValue, iterArgValue, Value(), Value(),
             yieldedValue, resultValue, getValueId(initValue), iterArgValueId,
             -1, -1, getValueId(yieldedValue), getValueId(resultValue)});
      }
    });
    function.walk([&](scf::WhileOp whileOp) {
      auto conditionArgs = whileOp.getConditionOp().getArgs();
      auto yieldedValues = whileOp.getYieldOp().getOperands();
      auto beforeArgs = whileOp.getBeforeArguments();
      auto afterArgs = whileOp.getAfterArguments();
      unsigned slotCount =
          std::min({static_cast<unsigned>(whileOp.getInits().size()),
                    static_cast<unsigned>(beforeArgs.size()),
                    static_cast<unsigned>(conditionArgs.size()),
                    static_cast<unsigned>(afterArgs.size()),
                    static_cast<unsigned>(yieldedValues.size()),
                    static_cast<unsigned>(whileOp.getNumResults())});
      for (unsigned slot = 0; slot < slotCount; ++slot) {
        Value beforeArg = beforeArgs[slot];
        if (!isLogicalValue(beforeArg.getType()))
          continue;
        Value initValue = whileOp.getInits()[slot];
        Value conditionValue = conditionArgs[slot];
        Value afterArg = afterArgs[slot];
        Value yieldedValue = yieldedValues[slot];
        Value resultValue = whileOp.getResult(slot);
        int64_t beforeArgValueId = assignValueId(beforeArg);
        int64_t afterArgValueId = assignValueId(afterArg);
        loopCarriedValues.push_back(
            {whileOp, slot, initValue, beforeArg, conditionValue, afterArg,
             yieldedValue, resultValue, getValueId(initValue), beforeArgValueId,
             assignValueId(conditionValue), afterArgValueId,
             getValueId(yieldedValue), getValueId(resultValue)});
      }
    });

    SmallVector<ComputeNodeFacts, 4> nodes;
    DenseMap<Operation *, int64_t> nodeIds;
    function.walk([&](Operation *op) {
      if (!isComputeNode(op))
        return;
      ComputeNodeFacts node;
      node.operation = op;
      node.id = nextNodeId++;
      node.barrierAfter = isDataflowBarrier(op);
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        node.rematerializable =
            linalgOp.hasPureTensorSemantics() &&
            (isa<linalg::FillOp>(op) || (isa<linalg::GenericOp>(op) &&
                                         linalgOp.getNumReductionLoops() == 0));
        if (node.rematerializable)
          op->walk([&](Operation *nested) {
            if (nested != op && !isMemoryEffectFree(nested))
              node.rematerializable = false;
          });
      }
      nodeIds.try_emplace(op, node.id);
      nodes.push_back(std::move(node));
    });

    DenseSet<int64_t> gridAxes;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      auto role =
          function.getArgAttrOfType<StringAttr>(index, "bosc_ame.program_role");
      auto axis = function.getArgAttrOfType<IntegerAttr>(
          index, "bosc_ame.program_axis");
      if (role && role.getValue() == "num_programs" && axis)
        gridAxes.insert(axis.getInt());
    }

    ReachingWriterMap lastWriter;
    TensorProducerMap tensorProducers;
    struct LoopReachingWriters {
      ReachingWriterMap entry;
      ReachingWriterMap condition;
      ReachingWriterMap bodyExit;
    };
    DenseMap<Operation *, LoopReachingWriters> loopReachingWriters;
    DenseMap<Operation *, size_t> nodeIndices;
    for (auto [index, node] : llvm::enumerate(nodes))
      nodeIndices.try_emplace(node.operation, index);
    DenseMap<int64_t, SmallVector<int64_t>> valueConsumers;
    enum class AnalysisMode { Order, Transfer, Lineage };
    auto collectProducers = [&](Value value, Operation *consumer,
                                const ReachingWriterMap &writers,
                                const TensorProducerMap &snapshots,
                                SmallVectorImpl<int64_t> &producers,
                                AnalysisMode mode) {
      if (mode == AnalysisMode::Order) {
        DenseSet<Value> visited;
        collectComputeProducers(value, consumer, nodeIds, writers, snapshots,
                                callDependencies, visited, producers);
      } else {
        DenseSet<Value> visited;
        collectLoopLineageProducers(value, consumer, nodeIds, writers,
                                    snapshots, callDependencies, visited,
                                    producers);
      }
      normalizeIds(producers);
    };
    auto analyzeNode = [&](ComputeNodeFacts &node,
                           ReachingWriterMap &reachingWriters,
                           const TensorProducerMap &snapshots,
                           AnalysisMode mode) {
      Operation *op = node.operation;
      SmallVector<Value> reads, writes;
      getReadWriteValues(op, reads, writes);

      if (mode != AnalysisMode::Transfer) {
        for (Value read : reads) {
          auto &producers = mode == AnalysisMode::Order
                                ? node.sourcePredecessors
                                : node.lineagePredecessors;
          collectProducers(read, op, reachingWriters, snapshots, producers,
                           mode);
          if (mode == AnalysisMode::Order && isLogicalValue(read.getType())) {
            auto valueId = valueIds.find(read);
            node.inputValues.push_back(
                valueId == valueIds.end() ? -1 : valueId->second);
            if (valueId != valueIds.end())
              valueConsumers[valueId->second].push_back(node.id);
          }
        }
      }

      if (mode == AnalysisMode::Order) {
        node.predecessors.append(node.sourcePredecessors.begin(),
                                 node.sourcePredecessors.end());
        ValueRange outputs =
            op->getNumResults() ? op->getResults() : ValueRange(writes);
        for (Value output : outputs) {
          if (!isLogicalValue(output.getType()))
            continue;
          auto valueId = valueIds.find(output);
          node.outputValues.push_back(
              valueId == valueIds.end() ? -1 : valueId->second);
        }
      }

      for (Value write : writes) {
        Value root = getDataflowStorageRoot(write);
        if (!isa<MemRefType>(root.getType()))
          continue;
        auto &writers = reachingWriters[root];
        if (mode == AnalysisMode::Order)
          node.predecessors.append(writers.begin(), writers.end());
        // Only full-root copies/fills prove a complete overwrite. Partial
        // writes retain sources for the elements that may remain untouched.
        bool completeOverwrite = write == root && (isa<memref::CopyOp>(op) ||
                                                   isa<linalg::FillOp>(op));
        if (completeOverwrite)
          writers.clear();
        writers.push_back(node.id);
        normalizeIds(writers);
      }
    };

    // Order runs once to construct the lexical compute DAG. The independent
    // lineage pass solves loop recurrences over finite producer-ID sets;
    // Transfer only simulates state and never mutates node/consumer facts.
    std::function<void(Block &, ReachingWriterMap &, TensorProducerMap &,
                       AnalysisMode)>
        analyzeBlock;
    analyzeBlock = [&](Block &block, ReachingWriterMap &reachingWriters,
                       TensorProducerMap &snapshots, AnalysisMode mode) {
      for (Operation &operation : block) {
        if (isa<memref::AllocOp, memref::AllocaOp>(operation))
          // Each iteration creates a new storage incarnation at this point.
          reachingWriters[operation.getResult(0)].clear();
        if (auto toBuffer = dyn_cast<bufferization::ToBufferOp>(operation)) {
          SmallVector<int64_t> producers;
          collectProducers(toBuffer.getTensor(), toBuffer, reachingWriters,
                           snapshots, producers, mode);
          auto &writers = reachingWriters[toBuffer.getBuffer()];
          writers.assign(producers.begin(), producers.end());
        }
        if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(operation)) {
          // Capture at the bridge, before any later mutable-buffer writes.
          SmallVector<int64_t> producers;
          collectProducers(toTensor.getBuffer(), toTensor, reachingWriters,
                           snapshots, producers, mode);
          auto &snapshot = snapshots[toTensor.getResult()];
          snapshot.assign(producers.begin(), producers.end());
        }
        if (auto executeRegion = dyn_cast<scf::ExecuteRegionOp>(&operation)) {
          // A single-block execute_region is an exactly-once forwarding
          // container. Analyze its body in place so writers and bridge
          // snapshots reach users after the result. Multi-block CFGs remain
          // conservative and are handled by the generic incomplete path.
          if (executeRegion.getRegion().hasOneBlock() &&
              isa<scf::YieldOp>(
                  executeRegion.getRegion().front().getTerminator()))
            analyzeBlock(executeRegion.getRegion().front(), reachingWriters,
                         snapshots, mode);
          continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
          ReachingWriterMap thenWriters = reachingWriters;
          TensorProducerMap thenSnapshots = snapshots;
          analyzeBlock(ifOp.getThenRegion().front(), thenWriters, thenSnapshots,
                       mode);
          ReachingWriterMap elseWriters = reachingWriters;
          TensorProducerMap elseSnapshots = snapshots;
          if (!ifOp.getElseRegion().empty())
            analyzeBlock(ifOp.getElseRegion().front(), elseWriters,
                         elseSnapshots, mode);
          reachingWriters = std::move(thenWriters);
          mergeReachingWriters(reachingWriters, elseWriters);
          snapshots = std::move(thenSnapshots);
          mergeTensorProducers(snapshots, elseSnapshots);
          continue;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
          ReachingWriterMap entryWriters = reachingWriters;
          if (mode == AnalysisMode::Order) {
            ReachingWriterMap bodyWriters = entryWriters;
            loopReachingWriters[forOp].entry = entryWriters;
            analyzeBlock(*forOp.getBody(), bodyWriters, snapshots, mode);
            loopReachingWriters[forOp].bodyExit = bodyWriters;
            mergeReachingWriters(reachingWriters, bodyWriters);
            continue;
          }

          ReachingWriterMap headerWriters = entryWriters;
          TensorProducerMap headerSnapshots = snapshots;
          ReachingWriterMap bodyExitWriters;
          while (true) {
            // Replay from the header, so full overwrites still kill sources
            // before later reads; do not accumulate body-exit lexical states.
            bodyExitWriters = headerWriters;
            TensorProducerMap iterationSnapshots = headerSnapshots;
            analyzeBlock(*forOp.getBody(), bodyExitWriters, iterationSnapshots,
                         AnalysisMode::Transfer);
            ReachingWriterMap nextHeaderWriters = entryWriters;
            mergeReachingWriters(nextHeaderWriters, bodyExitWriters);
            TensorProducerMap nextHeaderSnapshots = headerSnapshots;
            mergeTensorProducers(nextHeaderSnapshots, iterationSnapshots);
            if (reachingWritersEqual(nextHeaderWriters, headerWriters) &&
                reachingWritersEqual(nextHeaderSnapshots, headerSnapshots))
              break;
            headerWriters = std::move(nextHeaderWriters);
            headerSnapshots = std::move(nextHeaderSnapshots);
          }
          if (mode == AnalysisMode::Lineage) {
            ReachingWriterMap lineageWriters = headerWriters;
            analyzeBlock(*forOp.getBody(), lineageWriters, headerSnapshots,
                         mode);
          }
          snapshots = std::move(headerSnapshots);
          reachingWriters = entryWriters;
          // Retain the entry path even when the bounds happen to be static.
          mergeReachingWriters(reachingWriters, bodyExitWriters);
          continue;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
          ReachingWriterMap entryWriters = reachingWriters;
          TensorProducerMap entrySnapshots = snapshots;
          if (mode == AnalysisMode::Order) {
            ReachingWriterMap beforeWriters = entryWriters;
            TensorProducerMap beforeSnapshots = entrySnapshots;
            analyzeBlock(*whileOp.getBeforeBody(), beforeWriters,
                         beforeSnapshots, mode);
            ReachingWriterMap afterWriters = beforeWriters;
            TensorProducerMap afterSnapshots = beforeSnapshots;
            analyzeBlock(*whileOp.getAfterBody(), afterWriters, afterSnapshots,
                         mode);

            ReachingWriterMap headerWriters = entryWriters;
            TensorProducerMap headerSnapshots = entrySnapshots;
            ReachingWriterMap beforeExitWriters;
            ReachingWriterMap afterExitWriters;
            while (true) {
              beforeExitWriters = headerWriters;
              TensorProducerMap iterationSnapshots = headerSnapshots;
              analyzeBlock(*whileOp.getBeforeBody(), beforeExitWriters,
                           iterationSnapshots, AnalysisMode::Transfer);
              afterExitWriters = beforeExitWriters;
              analyzeBlock(*whileOp.getAfterBody(), afterExitWriters,
                           iterationSnapshots, AnalysisMode::Transfer);
              ReachingWriterMap nextHeaderWriters = entryWriters;
              mergeReachingWriters(nextHeaderWriters, afterExitWriters);
              TensorProducerMap nextHeaderSnapshots = entrySnapshots;
              mergeTensorProducers(nextHeaderSnapshots, iterationSnapshots);
              if (reachingWritersEqual(nextHeaderWriters, headerWriters) &&
                  reachingWritersEqual(nextHeaderSnapshots, headerSnapshots))
                break;
              headerWriters = std::move(nextHeaderWriters);
              headerSnapshots = std::move(nextHeaderSnapshots);
            }
            beforeExitWriters = headerWriters;
            TensorProducerMap exitSnapshots = headerSnapshots;
            analyzeBlock(*whileOp.getBeforeBody(), beforeExitWriters,
                         exitSnapshots, AnalysisMode::Transfer);
            ReachingWriterMap conditionWriters = beforeExitWriters;
            afterExitWriters = beforeExitWriters;
            analyzeBlock(*whileOp.getAfterBody(), afterExitWriters,
                         exitSnapshots, AnalysisMode::Transfer);
            loopReachingWriters[whileOp].entry = entryWriters;
            loopReachingWriters[whileOp].condition = conditionWriters;
            loopReachingWriters[whileOp].bodyExit = afterExitWriters;
            mergeTensorProducers(headerSnapshots, beforeSnapshots);
            mergeTensorProducers(headerSnapshots, afterSnapshots);
            mergeTensorProducers(headerSnapshots, exitSnapshots);
            reachingWriters = std::move(beforeExitWriters);
            snapshots = std::move(headerSnapshots);
            continue;
          }

          ReachingWriterMap headerWriters = entryWriters;
          TensorProducerMap headerSnapshots = entrySnapshots;
          ReachingWriterMap beforeExitWriters;
          ReachingWriterMap afterExitWriters;
          while (true) {
            beforeExitWriters = headerWriters;
            TensorProducerMap beforeSnapshots = headerSnapshots;
            analyzeBlock(*whileOp.getBeforeBody(), beforeExitWriters,
                         beforeSnapshots, AnalysisMode::Transfer);
            afterExitWriters = beforeExitWriters;
            TensorProducerMap afterSnapshots = beforeSnapshots;
            analyzeBlock(*whileOp.getAfterBody(), afterExitWriters,
                         afterSnapshots, AnalysisMode::Transfer);

            ReachingWriterMap nextHeaderWriters = entryWriters;
            mergeReachingWriters(nextHeaderWriters, afterExitWriters);
            TensorProducerMap nextHeaderSnapshots = entrySnapshots;
            mergeTensorProducers(nextHeaderSnapshots, beforeSnapshots);
            mergeTensorProducers(nextHeaderSnapshots, afterSnapshots);
            if (reachingWritersEqual(nextHeaderWriters, headerWriters) &&
                reachingWritersEqual(nextHeaderSnapshots, headerSnapshots))
              break;
            headerWriters = std::move(nextHeaderWriters);
            headerSnapshots = std::move(nextHeaderSnapshots);
          }

          if (mode == AnalysisMode::Lineage) {
            beforeExitWriters = headerWriters;
            TensorProducerMap beforeSnapshots = headerSnapshots;
            analyzeBlock(*whileOp.getBeforeBody(), beforeExitWriters,
                         beforeSnapshots, mode);
            loopReachingWriters[whileOp].condition = beforeExitWriters;
            afterExitWriters = beforeExitWriters;
            TensorProducerMap afterSnapshots = beforeSnapshots;
            analyzeBlock(*whileOp.getAfterBody(), afterExitWriters,
                         afterSnapshots, mode);
            mergeTensorProducers(headerSnapshots, beforeSnapshots);
            mergeTensorProducers(headerSnapshots, afterSnapshots);
          }

          // A while exits from its before region. The after region only feeds
          // the next header; it is not itself a post-loop writer state.
          loopReachingWriters[whileOp].entry = entryWriters;
          if (mode != AnalysisMode::Lineage)
            loopReachingWriters[whileOp].condition = beforeExitWriters;
          loopReachingWriters[whileOp].bodyExit = afterExitWriters;
          reachingWriters = std::move(beforeExitWriters);
          snapshots = std::move(headerSnapshots);
          continue;
        }
        if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(operation)) {
          ReachingWriterMap entryWriters = reachingWriters;
          TensorProducerMap entrySnapshots = snapshots;
          ReachingWriterMap joinedWriters;
          TensorProducerMap joinedSnapshots;
          bool firstRegion = true;
          auto analyzeRegion = [&](Region &region) {
            ReachingWriterMap branchWriters = entryWriters;
            TensorProducerMap branchSnapshots = entrySnapshots;
            if (!region.empty())
              analyzeBlock(region.front(), branchWriters, branchSnapshots,
                           mode);
            if (firstRegion) {
              joinedWriters = std::move(branchWriters);
              joinedSnapshots = std::move(branchSnapshots);
              firstRegion = false;
            } else {
              mergeReachingWriters(joinedWriters, branchWriters);
              mergeTensorProducers(joinedSnapshots, branchSnapshots);
            }
          };
          analyzeRegion(switchOp.getDefaultRegion());
          for (Region &region : switchOp.getCaseRegions())
            analyzeRegion(region);
          if (!firstRegion) {
            reachingWriters = std::move(joinedWriters);
            snapshots = std::move(joinedSnapshots);
          }
          continue;
        }
        if (auto found = nodeIndices.find(&operation);
            found != nodeIndices.end())
          analyzeNode(nodes[found->second], reachingWriters, snapshots, mode);

        // Unknown region semantics remain incomplete; conservatively retain
        // both entry state and each region's possible writers/snapshots.
        if (operation.getNumRegions()) {
          ReachingWriterMap joinedWriters = reachingWriters;
          TensorProducerMap joinedSnapshots = snapshots;
          for (Region &region : operation.getRegions())
            for (Block &nestedBlock : region) {
              ReachingWriterMap nestedWriters = reachingWriters;
              TensorProducerMap nestedSnapshots = snapshots;
              analyzeBlock(nestedBlock, nestedWriters, nestedSnapshots, mode);
              mergeReachingWriters(joinedWriters, nestedWriters);
              mergeTensorProducers(joinedSnapshots, nestedSnapshots);
            }
          reachingWriters = std::move(joinedWriters);
          snapshots = std::move(joinedSnapshots);
        }
      }
    };
    for (Block &block : function.getBody())
      analyzeBlock(block, lastWriter, tensorProducers, AnalysisMode::Order);
    ReachingWriterMap lineageWriters;
    TensorProducerMap lineageSnapshots;
    for (Block &block : function.getBody())
      analyzeBlock(block, lineageWriters, lineageSnapshots,
                   AnalysisMode::Lineage);

    // Build address/program dependencies separately from tensor data edges.
    // A producer's axes are inherited by its consumers; view/index SSA
    // expressions are inspected at the operation that materializes them.
    DenseMap<int64_t, DenseSet<int64_t>> varyingAxesByNode;
    for (ComputeNodeFacts &node : nodes) {
      DenseSet<int64_t> varying;
      Operation *op = node.operation;
      for (Value operand : op->getOperands()) {
        DenseSet<Value> visited;
        collectProgramAxes(operand, function, nodeIds, visited, varying);
      }
      for (int64_t predecessor : node.predecessors)
        if (auto axes = varyingAxesByNode.find(predecessor);
            axes != varyingAxesByNode.end())
          varying.insert(axes->second.begin(), axes->second.end());
      varyingAxesByNode.try_emplace(node.id, varying);
      node.varyingProgramAxes.assign(varying.begin(), varying.end());
      llvm::sort(node.varyingProgramAxes);
      llvm::sort(node.predecessors);
      node.predecessors.erase(
          std::unique(node.predecessors.begin(), node.predecessors.end()),
          node.predecessors.end());
      for (int64_t predecessor : node.predecessors) {
        auto parent =
            llvm::find_if(nodes, [&](const ComputeNodeFacts &candidate) {
              return candidate.id == predecessor;
            });
        if (parent != nodes.end())
          node.stage = std::max(node.stage,
                                parent->stage + int64_t(parent->barrierAfter));
      }
    }

    bool graphComplete = !hasUnmodeledArgumentAliases(module, function, nodes);
    function.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if ((name.starts_with("scf.") && name != "scf.yield") ||
          name == "func.call" || name == "func.call_indirect" ||
          name == "cf.br" || name == "cf.cond_br")
        graphComplete = false;
      if (hasUnmodeledDataflowEffects(op))
        graphComplete = false;
    });
    function->setAttr("bosc_ame.compute_graph_complete",
                      BoolAttr::get(context, graphComplete));
    function->setAttr(
        "bosc_ame.compute_graph_node_count",
        IntegerAttr::get(IntegerType::get(context, 64), nodes.size()));
    if (!graphComplete)
      function->setAttr(
          "bosc_ame.compute_graph_limitation",
          StringAttr::get(context,
                          "control flow, calls, memory effects, or potential "
                          "argument aliases are not modeled"));
    function->setAttr("bosc_ame.source_lineage_complete",
                      BoolAttr::get(context, graphComplete));

    // Carry source-load identities along data dependencies. Write-order
    // dependencies do not make the previous buffer contents an input to a
    // complete overwrite, and must not contaminate logical source lineage.
    DenseMap<int64_t, unsigned> nodeIndexById;
    for (auto [index, node] : llvm::enumerate(nodes))
      nodeIndexById.try_emplace(node.id, index);
    bool lineageChanged;
    do {
      lineageChanged = false;
      for (ComputeNodeFacts &node : nodes) {
        SmallVector<int64_t> origins(node.sourceLoadOrigins);
        SmallVector<int64_t> producers(node.sourcePredecessors);
        producers.append(node.lineagePredecessors.begin(),
                         node.lineagePredecessors.end());
        normalizeIds(producers);
        for (int64_t predecessor : producers) {
          auto parentIndex = nodeIndexById.find(predecessor);
          if (parentIndex == nodeIndexById.end())
            continue;
          const auto &parentOrigins =
              nodes[parentIndex->second].sourceLoadOrigins;
          origins.append(parentOrigins.begin(), parentOrigins.end());
        }
        if (node.operation->hasAttr("bosc_ame.memory")) {
          auto memory =
              node.operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
          auto kind = memory.getAs<StringAttr>("kind");
          auto origin = memory.getAs<IntegerAttr>("origin_id");
          if (kind && kind.getValue() == "load" && origin)
            origins.push_back(origin.getInt());
        }
        llvm::sort(origins);
        origins.erase(std::unique(origins.begin(), origins.end()),
                      origins.end());
        if (origins != node.sourceLoadOrigins) {
          node.sourceLoadOrigins = std::move(origins);
          lineageChanged = true;
        }
      }
    } while (lineageChanged);

    for (ComputeNodeFacts &node : nodes) {
      Operation *op = node.operation;
      op->setAttr("bosc_ame.compute_id",
                  IntegerAttr::get(IntegerType::get(context, 64), node.id));
      op->setAttr("bosc_ame.compute_kind",
                  StringAttr::get(context, getComputeNodeKind(op)));
      op->setAttr("bosc_ame.compute_stage",
                  IntegerAttr::get(IntegerType::get(context, 64), node.stage));
      op->setAttr("bosc_ame.compute_predecessors",
                  DenseI64ArrayAttr::get(context, node.predecessors));
      op->setAttr("bosc_ame.compute_input_values",
                  DenseI64ArrayAttr::get(context, node.inputValues));
      op->setAttr("bosc_ame.compute_output_values",
                  DenseI64ArrayAttr::get(context, node.outputValues));
      op->setAttr("bosc_ame.source_load_origin_ids",
                  DenseI64ArrayAttr::get(context, node.sourceLoadOrigins));
      op->setAttr("bosc_ame.varying_program_axes",
                  DenseI64ArrayAttr::get(context, node.varyingProgramAxes));
      SmallVector<int64_t> invariantAxes;
      for (int64_t axis : gridAxes)
        if (!llvm::is_contained(node.varyingProgramAxes, axis))
          invariantAxes.push_back(axis);
      llvm::sort(invariantAxes);
      op->setAttr("bosc_ame.invariant_program_axes",
                  DenseI64ArrayAttr::get(context, invariantAxes));
      op->setAttr("bosc_ame.compute_barrier_after",
                  BoolAttr::get(context, node.barrierAfter));
      op->setAttr("bosc_ame.rematerializable",
                  BoolAttr::get(context, node.rematerializable));
      auto origin = op->getAttrOfType<IntegerAttr>("bosc_ame.origin_id");
      if (!origin)
        if (auto memory = op->getAttrOfType<DictionaryAttr>("bosc_ame.memory"))
          origin = memory.getAs<IntegerAttr>("origin_id");
      if (origin)
        op->setAttr("bosc_ame.compute_origin_id", origin);
    }

    DenseMap<Operation *, SmallVector<Attribute>> loopSlotsByOperation;
    Builder builder(context);
    auto getTaskUsers = [&](int64_t valueId) {
      SmallVector<int64_t> users;
      if (auto found = valueConsumers.find(valueId);
          found != valueConsumers.end())
        users = found->second;
      llvm::sort(users);
      users.erase(std::unique(users.begin(), users.end()), users.end());
      return users;
    };
    auto getForwardedTaskUsers = [&](Value value) {
      SmallVector<int64_t> users;
      DenseSet<Value> visited;
      collectComputeConsumers(value, nodeIds, callDependencies, visited, users);
      llvm::sort(users);
      users.erase(std::unique(users.begin(), users.end()), users.end());
      return users;
    };
    auto getProducerTasks = [&](Value value, Operation *loop, bool backedge,
                                bool condition = false) {
      SmallVector<int64_t> producers;
      DenseSet<Value> visited;
      auto states = loopReachingWriters.find(loop);
      const ReachingWriterMap &writers =
          states == loopReachingWriters.end()
              ? lastWriter
              : (backedge ? states->second.bodyExit
                          : (condition ? states->second.condition
                                       : states->second.entry));
      collectComputeProducers(value, loop, nodeIds, writers, tensorProducers,
                              callDependencies, visited, producers);
      llvm::sort(producers);
      producers.erase(std::unique(producers.begin(), producers.end()),
                      producers.end());
      return producers;
    };
    for (const LoopCarriedValueFacts &slot : loopCarriedValues) {
      SmallVector<NamedAttribute> attributes;
      attributes.push_back(
          builder.getNamedAttr("slot", builder.getI64IntegerAttr(slot.slot)));
      attributes.push_back(builder.getNamedAttr(
          "init_value_id", builder.getI64IntegerAttr(slot.initValueId)));
      attributes.push_back(builder.getNamedAttr(
          "iter_arg_value_id", builder.getI64IntegerAttr(slot.iterArgValueId)));
      attributes.push_back(builder.getNamedAttr(
          "yielded_value_id", builder.getI64IntegerAttr(slot.yieldedValueId)));
      attributes.push_back(builder.getNamedAttr(
          "result_value_id", builder.getI64IntegerAttr(slot.resultValueId)));
      if (slot.conditionValue) {
        attributes.push_back(builder.getNamedAttr(
            "condition_value_id",
            builder.getI64IntegerAttr(slot.conditionValueId)));
        attributes.push_back(builder.getNamedAttr(
            "after_arg_value_id",
            builder.getI64IntegerAttr(slot.afterArgValueId)));
        attributes.push_back(builder.getNamedAttr(
            "condition_producer_compute_ids",
            DenseI64ArrayAttr::get(context,
                                   getProducerTasks(slot.conditionValue,
                                                    slot.loop, false, true))));
        SmallVector<int64_t> afterArgConsumers =
            getTaskUsers(slot.afterArgValueId);
        attributes.push_back(builder.getNamedAttr(
            "after_arg_consumer_compute_ids",
            DenseI64ArrayAttr::get(context, afterArgConsumers)));
      }
      SmallVector<int64_t> initProducers =
          getProducerTasks(slot.initValue, slot.loop, false);
      SmallVector<int64_t> iterArgConsumers = getTaskUsers(slot.iterArgValueId);
      SmallVector<int64_t> backedgeProducers =
          getProducerTasks(slot.yieldedValue, slot.loop, true);
      SmallVector<int64_t> exitConsumers =
          getForwardedTaskUsers(slot.resultValue);
      attributes.push_back(
          builder.getNamedAttr("init_producer_compute_ids",
                               DenseI64ArrayAttr::get(context, initProducers)));
      attributes.push_back(builder.getNamedAttr(
          "iter_arg_consumer_compute_ids",
          DenseI64ArrayAttr::get(context, iterArgConsumers)));
      attributes.push_back(builder.getNamedAttr(
          "backedge_producer_compute_ids",
          DenseI64ArrayAttr::get(context, backedgeProducers)));
      attributes.push_back(
          builder.getNamedAttr("exit_consumer_compute_ids",
                               DenseI64ArrayAttr::get(context, exitConsumers)));
      loopSlotsByOperation[slot.loop].push_back(
          DictionaryAttr::get(context, attributes));
    }
    for (auto &entry : loopSlotsByOperation)
      entry.first->setAttr("bosc_ame.loop_carried_slots",
                           ArrayAttr::get(context, entry.second));

    for (auto [value, valueId] : valueIds) {
      auto users = valueConsumers.find(valueId);
      if (users == valueConsumers.end())
        continue;
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() == &function.getBody().front()) {
          SmallVector<int64_t> uniqueUsers(users->second);
          llvm::sort(uniqueUsers);
          uniqueUsers.erase(std::unique(uniqueUsers.begin(), uniqueUsers.end()),
                            uniqueUsers.end());
          function.setArgAttr(argument.getArgNumber(),
                              "bosc_ame.logical_value_users",
                              DenseI64ArrayAttr::get(context, uniqueUsers));
        }
      } else if (Operation *def = value.getDefiningOp()) {
        SmallVector<Attribute> valueUsers;
        for (Value result : def->getResults()) {
          auto resultId = valueIds.find(result);
          SmallVector<int64_t> resultUsers;
          if (resultId != valueIds.end()) {
            resultUsers = valueConsumers[resultId->second];
            llvm::sort(resultUsers);
            resultUsers.erase(
                std::unique(resultUsers.begin(), resultUsers.end()),
                resultUsers.end());
          }
          valueUsers.push_back(DenseI64ArrayAttr::get(context, resultUsers));
        }
        def->setAttr("bosc_ame.logical_value_users",
                     ArrayAttr::get(context, valueUsers));
      }
    }
  });

  module->setAttr("bosc_ame.compute_graph_version",
                  IntegerAttr::get(IntegerType::get(context, 32), 1));
}

static StringRef definingOperationName(Value value) {
  if (Operation *def = value.getDefiningOp())
    return def->getName().getStringRef();
  return isa<BlockArgument>(value) ? "block-argument" : "unknown";
}

static StringRef getCmpIPredicateName(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return "eq";
  case arith::CmpIPredicate::ne:
    return "ne";
  case arith::CmpIPredicate::slt:
    return "slt";
  case arith::CmpIPredicate::sle:
    return "sle";
  case arith::CmpIPredicate::sgt:
    return "sgt";
  case arith::CmpIPredicate::sge:
    return "sge";
  case arith::CmpIPredicate::ult:
    return "ult";
  case arith::CmpIPredicate::ule:
    return "ule";
  case arith::CmpIPredicate::ugt:
    return "ugt";
  case arith::CmpIPredicate::uge:
    return "uge";
  }
  llvm_unreachable("unknown arith.cmpi predicate");
}

static void printMaskBoundSource(Value value, llvm::raw_ostream &out,
                                 unsigned depth = 0) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    out << "arg#" << argument.getArgNumber();
    return;
  }
  Operation *def = value.getDefiningOp();
  if (!def) {
    out << "unknown";
    return;
  }
  StringRef name = def->getName().getStringRef();
  if (auto constant = dyn_cast<arith::ConstantOp>(def)) {
    out << "constant(";
    constant.getValue().print(out);
    out << ")";
    return;
  }
  if (depth < 2 && def->getNumOperands() == 1 &&
      (name == "tt.splat" || name == "arith.index_cast" ||
       name == "arith.extsi" || name == "arith.trunci")) {
    out << name << "(";
    printMaskBoundSource(def->getOperand(0), out, depth + 1);
    out << ")";
    return;
  }
  out << name;
}

static void printMaskComparison(Value mask, llvm::raw_ostream &out) {
  auto compare = mask.getDefiningOp<arith::CmpIOp>();
  if (!compare)
    return;
  out << " mask-cmp=" << getCmpIPredicateName(compare.getPredicate())
      << " mask-lhs=" << definingOperationName(compare.getLhs())
      << " mask-rhs=";
  printMaskBoundSource(compare.getRhs(), out);
}

static void printTypeList(TypeRange types, llvm::raw_ostream &out) {
  out << "[";
  llvm::interleaveComma(types, out, [&](Type type) { type.print(out); });
  out << "]";
}

static void printIntegerArray(Attribute attr, llvm::raw_ostream &out) {
  if (attr)
    attr.print(out);
  else
    out << "array<i64>";
}

struct SourceAddressExpressionFacts {
  enum class Kind { Unknown, Base, AddPtr };
  Kind kind = Kind::Unknown;
  Value base;
  Value splatSource;
  Value offset;
};

static SourceAddressExpressionFacts
getSourceAddressExpressionFacts(Value pointerOperand) {
  if (!pointerOperand)
    return {};
  if (isa<BlockArgument>(pointerOperand))
    return {SourceAddressExpressionFacts::Kind::Base, pointerOperand, {}, {}};
  auto addPtr = pointerOperand.getDefiningOp<triton::AddPtrOp>();
  if (!addPtr)
    return {};
  Value base = addPtr.getPtr();
  Value splatSource;
  if (auto splat = base.getDefiningOp<triton::SplatOp>())
    splatSource = splat.getSrc();
  return {SourceAddressExpressionFacts::Kind::AddPtr, base, splatSource,
          addPtr.getOffset()};
}

struct SourceEventFacts {
  Operation *operation;
  int64_t id;
  StringRef kind;
  Value pointerOperand;
  SourceAddressExpressionFacts addressExpression;
  Value other;
};

static SmallVector<SourceEventFacts> collectSourceEventFacts(ModuleOp module) {
  SmallVector<SourceEventFacts> events;
  module.walk([&](Operation *op) {
    auto id = op->getAttrOfType<IntegerAttr>("bosc_ame.origin_id");
    auto kind = op->getAttrOfType<StringAttr>("bosc_ame.event_kind");
    if (!id || !kind)
      return;
    Value pointerOperand;
    if ((kind.getValue() == "load" || kind.getValue() == "store") &&
        op->getNumOperands() > 0)
      pointerOperand = op->getOperand(0);
    Value other;
    if (kind.getValue() == "load" && op->getNumOperands() > 2)
      other = op->getOperand(2);
    events.push_back({op, id.getInt(), kind.getValue(), pointerOperand,
                      getSourceAddressExpressionFacts(pointerOperand), other});
  });
  return events;
}

static void printSourceFacts(ModuleOp moduleOp, llvm::raw_ostream &out) {
  out << "========== BOSCAME FACTS REPORT ==========\nSourceFacts:\n";
  DenseMap<Operation *, SourceEventFacts> eventsByOperation;
  for (const SourceEventFacts &event : collectSourceEventFacts(moduleOp))
    eventsByOperation.try_emplace(event.operation, event);
  moduleOp.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "tt.get_program_id" || name == "tt.get_num_programs") {
      out << "  program-query kind="
          << (name == "tt.get_program_id" ? "program_id" : "num_programs")
          << " axis=";
      if (auto axis = op->getAttrOfType<IntegerAttr>("axis"))
        out << axis.getInt();
      else
        out << "dynamic";
      out << "\n";
    }

    auto eventIt = eventsByOperation.find(op);
    if (eventIt == eventsByOperation.end())
      return;
    const SourceEventFacts &event = eventIt->second;
    out << "  event id=" << event.id << " kind=" << event.kind;
    if (op->getNumResults()) {
      out << " result=";
      op->getResult(0).getType().print(out);
    }
    if (event.pointerOperand) {
      out << " address=";
      event.pointerOperand.printAsOperand(out, OpPrintingFlags());
      out << ":";
      event.pointerOperand.getType().print(out);
      out << " address-expression=";
      switch (event.addressExpression.kind) {
      case SourceAddressExpressionFacts::Kind::Base:
        out << "base(";
        event.addressExpression.base.printAsOperand(out, OpPrintingFlags());
        out << ":";
        event.addressExpression.base.getType().print(out);
        out << ")";
        break;
      case SourceAddressExpressionFacts::Kind::AddPtr:
        out << "addptr(base=";
        event.addressExpression.base.printAsOperand(out, OpPrintingFlags());
        out << ":";
        event.addressExpression.base.getType().print(out);
        if (event.addressExpression.splatSource) {
          out << " splat-source=";
          event.addressExpression.splatSource.printAsOperand(out,
                                                             OpPrintingFlags());
          out << ":";
          event.addressExpression.splatSource.getType().print(out);
        }
        out << " offset=";
        event.addressExpression.offset.printAsOperand(out, OpPrintingFlags());
        out << ":";
        event.addressExpression.offset.getType().print(out);
        out << ")";
        break;
      case SourceAddressExpressionFacts::Kind::Unknown:
        out << "unknown";
        break;
      }
    }
    if (event.kind == "load" && op->getNumOperands() > 1) {
      out << " access-mask=" << definingOperationName(op->getOperand(1));
      if (event.other) {
        out << " other=";
        if (auto constant = event.other.getDefiningOp<arith::ConstantOp>()) {
          constant.getValue().print(out);
        } else {
          out << "ssa(";
          event.other.printAsOperand(out, OpPrintingFlags());
          out << ":";
          event.other.getType().print(out);
          out << ")";
        }
      }
      printMaskComparison(op->getOperand(1), out);
    } else if (event.kind == "store" && op->getNumOperands() > 2) {
      out << " access-mask=" << definingOperationName(op->getOperand(2));
      printMaskComparison(op->getOperand(2), out);
    } else if (event.kind == "dot" && op->getNumOperands() >= 3) {
      out << " a=";
      op->getOperand(0).getType().print(out);
      out << " b=";
      op->getOperand(1).getType().print(out);
      out << " accumulator=";
      op->getOperand(2).getType().print(out);
      out << " accumulator-source=" << definingOperationName(op->getOperand(2));
    }
    out << "\n";
  });
}

static void printGenericBody(Operation *op, llvm::raw_ostream &out) {
  bool first = true;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block) {
        if (nested.hasTrait<OpTrait::IsTerminator>())
          continue;
        if (!first)
          out << ",";
        out << nested.getName().getStringRef();
        first = false;
      }
}

static void printReductionInputChain(
    Value value, const llvm::DenseMap<Operation *, unsigned> &operationNumbers,
    llvm::raw_ostream &out) {
  SmallPtrSet<Operation *, 8> visited;
  bool first = true;
  while (Operation *def = value.getDefiningOp()) {
    if (!visited.insert(def).second ||
        def->getName().getStringRef() != "linalg.generic")
      break;
    if (!first)
      out << " <- ";
    out << "generic#" << operationNumbers.lookup(def) << "[";
    printGenericBody(def, out);
    out << "]";
    first = false;
    if (def->getNumOperands() == 0)
      break;
    value = def->getOperand(0);
  }
}

static void printLinalgValueRef(
    Value value, const llvm::DenseMap<Operation *, unsigned> &operationNumbers,
    const llvm::DenseMap<Operation *, unsigned> &irOperationNumbers,
    llvm::raw_ostream &out) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Block *block = argument.getOwner();
    Region *region = block->getParent();
    Operation *owner = block->getParentOp();
    auto ownerNumber = irOperationNumbers.find(owner);
    if (!region || ownerNumber == irOperationNumbers.end()) {
      out << "unknown-block-arg#" << argument.getArgNumber();
      return;
    }
    out << "ir-op#" << ownerNumber->second << ":"
        << owner->getName().getStringRef() << ".region#"
        << region->getRegionNumber() << ".block#" << block->computeBlockNumber()
        << ".arg#" << argument.getArgNumber();
    return;
  }

  Operation *definingOp = value.getDefiningOp();
  auto result = dyn_cast<OpResult>(value);
  if (!definingOp || !result) {
    out << "unknown";
    return;
  }

  auto number = operationNumbers.find(definingOp);
  if (number != operationNumbers.end()) {
    out << "op#" << number->second << ".result#" << result.getResultNumber();
  } else {
    auto irNumber = irOperationNumbers.find(definingOp);
    out << "ir-op#";
    if (irNumber != irOperationNumbers.end())
      out << irNumber->second;
    else
      out << "?";
    out << ":" << definingOp->getName().getStringRef() << ".result#"
        << result.getResultNumber();
  }
}

static void printLinalgValueList(
    ValueRange values,
    const llvm::DenseMap<Operation *, unsigned> &operationNumbers,
    const llvm::DenseMap<Operation *, unsigned> &irOperationNumbers,
    llvm::raw_ostream &out) {
  out << "[";
  llvm::interleaveComma(values, out, [&](Value value) {
    printLinalgValueRef(value, operationNumbers, irOperationNumbers, out);
    out << ":";
    value.getType().print(out);
  });
  out << "]";
}

static void printLinalgResultUsers(
    Operation *op,
    const llvm::DenseMap<Operation *, unsigned> &operationNumbers,
    const llvm::DenseMap<Operation *, unsigned> &irOperationNumbers,
    llvm::raw_ostream &out) {
  out << " result-users=[";
  bool first = true;
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    if (!first)
      out << ";";
    first = false;
    out << "result#" << index << " users=[";
    SmallVector<Operation *> users;
    SmallPtrSet<Operation *, 8> seenUsers;
    for (Operation *user : result.getUsers()) {
      if (seenUsers.insert(user).second)
        users.push_back(user);
    }
    llvm::sort(users, [&](Operation *lhs, Operation *rhs) {
      auto getKey = [&](Operation *user) {
        auto number = operationNumbers.find(user);
        if (number != operationNumbers.end())
          return std::pair<unsigned, unsigned>(0, number->second);
        return std::pair<unsigned, unsigned>(1,
                                             irOperationNumbers.lookup(user));
      };
      return getKey(lhs) < getKey(rhs);
    });
    for (auto [userIndex, user] : llvm::enumerate(users)) {
      if (userIndex)
        out << ", ";
      auto number = operationNumbers.find(user);
      if (number != operationNumbers.end())
        out << "op#" << number->second;
      else {
        out << "ir-op#" << irOperationNumbers.lookup(user) << ":"
            << user->getName().getStringRef();
      }
    }
    out << "]";
  }
  out << "]";
}

static void printLinalgCommonFacts(
    linalg::LinalgOp op,
    const llvm::DenseMap<Operation *, unsigned> &operationNumbers,
    const llvm::DenseMap<Operation *, unsigned> &irOperationNumbers,
    llvm::raw_ostream &out) {
  out << " inputs=";
  printLinalgValueList(op.getDpsInputs(), operationNumbers, irOperationNumbers,
                       out);
  out << " outputs=";
  printLinalgValueList(op.getDpsInits(), operationNumbers, irOperationNumbers,
                       out);
  if (auto indexingMaps = op->getAttr("indexing_maps")) {
    out << " maps=";
    indexingMaps.print(out);
  }
  if (auto iteratorTypes = op->getAttr("iterator_types")) {
    out << " iterators=";
    iteratorTypes.print(out);
  }
  printLinalgResultUsers(op, operationNumbers, irOperationNumbers, out);
}

static void printLinalgFacts(ModuleOp moduleOp, llvm::raw_ostream &out) {
  buddy::boscame::LinalgComputationFacts linalgSSA =
      buddy::boscame::analyzeLinalgComputationFacts(moduleOp);
  llvm::DenseMap<Operation *, unsigned> operationNumbers;
  llvm::DenseMap<Operation *, unsigned> irOperationNumbers;
  unsigned nextIROperationNumber = 0;
  moduleOp.walk([&](Operation *op) {
    irOperationNumbers.try_emplace(op, nextIROperationNumber++);
  });
  unsigned nextNumber = 0;
  moduleOp.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "linalg.generic" || name == "linalg.reduce" ||
        name == "linalg.matmul")
      operationNumbers.try_emplace(op, nextNumber++);
  });

  out << "LinalgFacts:\n";
  moduleOp.walk([&](func::FuncOp function) {
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      Attribute role = function.getArgAttr(index, "bosc_ame.program_role");
      Attribute axis = function.getArgAttr(index, "bosc_ame.program_axis");
      if (!role || !axis)
        continue;
      out << "  program-argument role=";
      role.print(out);
      out << " axis=";
      axis.print(out);
      out << " type=";
      function.getArgument(index).getType().print(out);
      out << "\n";
    }
  });

  moduleOp.walk([&](Operation *op) {
    if (auto memory = op->getAttrOfType<DictionaryAttr>("bosc_ame.memory")) {
      out << "  memory-event ir-op#" << irOperationNumbers.lookup(op)
          << " op=" << op->getName().getStringRef() << " ";
      memory.print(out);
      out << "\n";
      if (op->getName().getStringRef() == "memref.copy") {
        auto kind = memory.getAs<StringAttr>("kind");
        auto origin = memory.getAs<IntegerAttr>("origin_id");
        out << "  memory-op ir-op#" << irOperationNumbers.lookup(op)
            << " kind=copy role=" << (kind ? kind.getValue() : "unknown")
            << " origin-id=";
        if (origin)
          out << origin.getInt();
        else
          out << "unknown";
        out << " operands=[";
        for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
          if (index)
            out << ",";
          printLinalgValueRef(operand, operationNumbers, irOperationNumbers,
                              out);
        }
        out << "]\n";
      }
    }

    if (isa<linalg::FillOp>(op)) {
      out << "  memory-op ir-op#" << irOperationNumbers.lookup(op)
          << " kind=fill role="
          << (op->hasAttr("bosc_ame.padding") ? "padding" : "logical")
          << " inputs=";
      printLinalgValueList(cast<linalg::FillOp>(op).getInputs(),
                           operationNumbers, irOperationNumbers, out);
      out << " outputs=";
      printLinalgValueList(cast<linalg::FillOp>(op).getOutputs(),
                           operationNumbers, irOperationNumbers, out);
      out << "\n";
    }

    StringRef name = op->getName().getStringRef();
    auto number = operationNumbers.find(op);
    if (number == operationNumbers.end())
      return;
    out << "  compute op#" << number->second << " " << name;
    if (name == "linalg.generic") {
      auto generic = cast<linalg::GenericOp>(op);
      out << " results=";
      printTypeList(op->getResultTypes(), out);
      out << " body=[";
      printGenericBody(op, out);
      out << "]";
      printLinalgCommonFacts(generic, operationNumbers, irOperationNumbers,
                             out);
    } else if (name == "linalg.reduce") {
      if (Attribute dimensions = op->getAttr("dimensions")) {
        out << " dimensions=";
        dimensions.print(out);
      }
      auto reduce = cast<linalg::ReduceOp>(op);
      printLinalgCommonFacts(reduce, operationNumbers, irOperationNumbers, out);
      out << " combiner=[";
      printGenericBody(op, out);
      out << "]";
      if (op->getNumOperands()) {
        out << " input-chain=";
        printReductionInputChain(op->getOperand(0), operationNumbers, out);
      }
    } else if (name == "linalg.matmul") {
      auto matmul = cast<linalg::MatmulOp>(op);
      printLinalgCommonFacts(matmul, operationNumbers, irOperationNumbers, out);
      for (StringRef attrName :
           {"bosc_ame.origin_id", "bosc_ame.a_origin_id",
            "bosc_ame.b_origin_id", "bosc_ame.accumulator_origin_id",
            "bosc_ame.a_source_load_origin_ids",
            "bosc_ame.a_source_lineage_complete",
            "bosc_ame.b_source_load_origin_ids",
            "bosc_ame.b_source_lineage_complete",
            "bosc_ame.accumulator_source_load_origin_ids",
            "bosc_ame.accumulator_source_lineage_complete"}) {
        if (Attribute attr = op->getAttr(attrName)) {
          out << " " << attrName.drop_front(StringRef("bosc_ame.").size())
              << "=";
          attr.print(out);
        }
      }
    }
    out << "\n";
  });
  buddy::boscame::printLinalgComputationFacts(linalgSSA, out, "  ");
  out << "ComputeStageGraph:\n";
  moduleOp.walk([&](func::FuncOp function) {
    auto complete =
        function->getAttrOfType<BoolAttr>("bosc_ame.compute_graph_complete");
    out << "  function " << function.getSymName()
        << " complete=" << (complete && complete.getValue() ? "true" : "false");
    if (Attribute limitation =
            function->getAttr("bosc_ame.compute_graph_limitation")) {
      out << " limitation=";
      limitation.print(out);
    }
    out << "\n";
  });
  moduleOp.walk([&](Operation *op) {
    auto id = op->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
    auto kind = op->getAttrOfType<StringAttr>("bosc_ame.compute_kind");
    auto stage = op->getAttrOfType<IntegerAttr>("bosc_ame.compute_stage");
    if (!id || !kind || !stage)
      return;
    out << "  node id=" << id.getInt() << " kind=" << kind.getValue()
        << " stage=" << stage.getInt() << " predecessors=";
    printIntegerArray(op->getAttr("bosc_ame.compute_predecessors"), out);
    out << " input-values=";
    printIntegerArray(op->getAttr("bosc_ame.compute_input_values"), out);
    out << " output-values=";
    printIntegerArray(op->getAttr("bosc_ame.compute_output_values"), out);
    out << " varying-axes=";
    printIntegerArray(op->getAttr("bosc_ame.varying_program_axes"), out);
    out << " invariant-axes=";
    printIntegerArray(op->getAttr("bosc_ame.invariant_program_axes"), out);
    if (auto barrier =
            op->getAttrOfType<BoolAttr>("bosc_ame.compute_barrier_after"))
      out << " barrier-after=" << (barrier.getValue() ? "true" : "false");
    if (auto rematerializable =
            op->getAttrOfType<BoolAttr>("bosc_ame.rematerializable"))
      out << " rematerializable="
          << (rematerializable.getValue() ? "true" : "false");
    if (auto origin =
            op->getAttrOfType<IntegerAttr>("bosc_ame.compute_origin_id"))
      out << " origin-id=" << origin.getInt();
    if (auto prefix =
            op->getAttrOfType<IntegerAttr>("bosc_ame.zero_tail_select_prefix"))
      out << " zero-tail-prefix=" << prefix.getInt();
    out << "\n";
  });
  out << "==========================================\n";
}

class TritonToLinalgExperimentalPass
    : public ::impl::TritonToLinalgExperimentalBase<
          TritonToLinalgExperimentalPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    vector::VectorDialect, tts::TritonStructuredDialect,
                    tptr::TPtrDialect, ptr::PtrDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    TritonLoweringConfig config{boscame, dmaAbi};
    SmallVector<LaunchGridContract, 4> launchGridContracts;
    bool hasLaunchGridOption = launchGrid.getNumOccurrences() != 0;
    if (hasLaunchGridOption && !config.enableBOSCAME) {
      moduleOp.emitError("launch-grid requires boscame=true");
      signalPassFailure();
      return;
    }
    if (hasLaunchGridOption && launchGrid.empty()) {
      moduleOp.emitError("invalid empty launch-grid option");
      signalPassFailure();
      return;
    }
    DenseSet<Operation *> launchGridFunctions;
    if (!launchGrid.empty()) {
      SmallVector<StringRef, 4> launchGridEntries;
      StringRef(launchGrid).split(launchGridEntries, ';', -1, true);
      for (StringRef specification : launchGridEntries) {
        FailureOr<LaunchGridContract> contract =
            parseLaunchGridContract(moduleOp, specification);
        if (failed(contract)) {
          moduleOp.emitError()
              << "invalid launch-grid entry '" << specification
              << "'; expected symbol:x,y,z with positive int32 extents and a "
                 "tt.func target";
          signalPassFailure();
          return;
        }
        if (!launchGridFunctions.insert(contract->function).second) {
          moduleOp.emitError()
              << "duplicate launch-grid contract for @"
              << SymbolTable::getSymbolName(contract->function).getValue();
          signalPassFailure();
          return;
        }
        Attribute existing = contract->function->getAttr("gpu.known_grid_size");
        if (existing && existing != contract->grid) {
          moduleOp.emitError()
              << "launch-grid conflicts with existing gpu.known_grid_size on @"
              << SymbolTable::getSymbolName(contract->function).getValue();
          signalPassFailure();
          return;
        }
        launchGridContracts.push_back(*contract);
      }
    }
    if (dumpFacts && !config.enableBOSCAME) {
      moduleOp.emitError("dump-facts requires boscame=true");
      signalPassFailure();
      return;
    }
    if (config.enableBOSCAME) {
      int64_t originId = 0;
      moduleOp.walk([&](Operation *op) {
        StringRef name = op->getName().getStringRef();
        StringRef kind;
        if (name == "tt.load")
          kind = "load";
        else if (name == "tt.store")
          kind = "store";
        else if (name == "tt.dot")
          kind = "dot";
        if (kind.empty())
          return;
        op->setAttr(
            "bosc_ame.origin_id",
            IntegerAttr::get(IntegerType::get(&getContext(), 64), originId++));
        op->setAttr("bosc_ame.event_kind",
                    StringAttr::get(&getContext(), kind));
      });
      using namespace buddy::boscame::spmdma;
      Attribute contract = moduleOp->getAttr(kDmaAbiAttrName);
      auto existing = dyn_cast_or_null<StringAttr>(contract);
      if (existing && !config.dmaAbi.empty() &&
          existing.getValue() != config.dmaAbi) {
        moduleOp.emitError(
            "command-line DMA ABI conflicts with the module contract");
        signalPassFailure();
        return;
      }
      if (contract &&
          (!existing || existing.getValue() != kDmaAbiAlignedNonAliasing)) {
        moduleOp.emitError("invalid BOSCAME module DMA ABI contract");
        signalPassFailure();
        return;
      }
      if (!config.dmaAbi.empty() &&
          config.dmaAbi != kDmaAbiAlignedNonAliasing) {
        moduleOp.emitError() << "invalid dma-abi: " << config.dmaAbi;
        signalPassFailure();
        return;
      }
      if (config.dmaAbi.empty() && existing)
        config.dmaAbi = existing.getValue().str();
      if (config.dmaAbi.empty()) {
        moduleOp.emitError(
            "boscame=true requires dma-abi=args_aligned64_nonalias");
        signalPassFailure();
        return;
      }
      moduleOp->setAttr(kDmaAbiAttrName,
                        StringAttr::get(&getContext(), config.dmaAbi));
      auto saveTile = [&](StringRef name, auto &option) -> LogicalResult {
        if (Attribute attr = moduleOp->getAttr(name)) {
          auto value = dyn_cast<IntegerAttr>(attr);
          if (!value || value.getInt() <= 0 ||
              (option.getNumOccurrences() && value.getInt() != option))
            return moduleOp.emitError() << "invalid or conflicting BOSCAME "
                                           "module configuration: "
                                        << name;
          option = value.getInt();
        }
        if (option <= 0)
          return moduleOp.emitError("BOSCAME tile sizes must be positive");
        moduleOp->setAttr(
            name,
            IntegerAttr::get(IntegerType::get(&getContext(), 64), option));
        return success();
      };
      if (failed(saveTile("bosc_ame.bm", bm)) ||
          failed(saveTile("bosc_ame.bn", bn)) ||
          failed(saveTile("bosc_ame.bk", bk))) {
        signalPassFailure();
        return;
      }
      if (Attribute attr = moduleOp->getAttr("bosc_ame.target")) {
        auto value = dyn_cast<StringAttr>(attr);
        if (!value || value.getValue() != "nr-fpga" ||
            (target.getNumOccurrences() && value.getValue() != target)) {
          moduleOp.emitError("invalid or conflicting BOSCAME module target");
          signalPassFailure();
          return;
        }
        target = value.getValue().str();
      }
      if (target != "nr-fpga") {
        moduleOp.emitError("unsupported BOSCAME target; expected nr-fpga");
        signalPassFailure();
        return;
      }
      for (StringRef name :
           {"bosc_ame.enable_spm_dma", "bosc_ame.enable_ame"}) {
        if (Attribute attr = moduleOp->getAttr(name)) {
          auto value = dyn_cast<BoolAttr>(attr);
          if (!value || !value.getValue()) {
            moduleOp.emitError() << "invalid or conflicting BOSCAME module "
                                    "configuration: "
                                 << name;
            signalPassFailure();
            return;
          }
        }
        moduleOp->setAttr(name, BoolAttr::get(&getContext(), true));
      }
      moduleOp->setAttr("bosc_ame.target",
                        StringAttr::get(&getContext(), target));
      // External Triton stores use direct DDR writeback by default. Preserve a
      // valid module policy so an explicitly authored compatibility contract
      // survives the frontend and can be inherited by the standalone backend.
      if (Attribute attr = moduleOp->getAttr("bosc_ame.store_policy")) {
        auto value = dyn_cast<StringAttr>(attr);
        if (!value || (value.getValue() != "direct-ddr" &&
                       value.getValue() != "spm-dma")) {
          moduleOp.emitError("invalid BOSCAME store policy; expected "
                             "direct-ddr or spm-dma");
          signalPassFailure();
          return;
        }
      } else {
        moduleOp->setAttr("bosc_ame.store_policy",
                          StringAttr::get(&getContext(), "direct-ddr"));
      }
      moduleOp->setAttr("bosc_ame.cpu_ddr_policy",
                        UnitAttr::get(&getContext()));
      // The logical placement mode is consumed by the standalone BOSCAME
      // planner after bufferization.  It authorizes direct DDR materialization
      // for the frontend's dense external A/B events; hand-written buffer
      // modules without this marker retain the compatibility SPM path.
      moduleOp->setAttr("bosc_ame.placement",
                        StringAttr::get(&getContext(), "logical"));
    }
    for (const LaunchGridContract &contract : launchGridContracts)
      if (!contract.function->hasAttr("gpu.known_grid_size"))
        contract.function->setAttr("gpu.known_grid_size", contract.grid);
    PassManager pm(&getContext(), moduleOp.getOperationName());
    bool enableTensorFirstVectorCpu = false;
    if (structuredLdstMode == "legacy_memref_bridge") {
      enableTensorFirstVectorCpu = false;
    } else if (structuredLdstMode == "tensor-first-vector-cpu") {
      enableTensorFirstVectorCpu = true;
    } else {
      moduleOp.emitError()
          << "invalid structured-ldst-mode: " << structuredLdstMode
          << ". Expected one of: legacy_memref_bridge, tensor-first-vector-cpu";
      signalPassFailure();
      return;
    }
    if (dumpFacts)
      printSourceFacts(moduleOp, llvm::errs());
    // Preserve each logical memory event for the later buffer-form planner.
    if (config.enableBOSCAME)
      enableTensorFirstVectorCpu = false;

    pm.addPass(createTritonToStructuredPass(enableMakeGatherScatterTensorPtr));

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createTritonToUnstructuredPass());
    // Preserve the source reduction axis. Materializing a transpose merely to
    // force dimension 0 creates large temporary buffers after bufferization
    // and is particularly costly for CPU row reductions.
    pm.addPass(createTritonArithToLinalgPass(
        /*tensorPtrToLinalg=*/true, /*transposeReduceToRank0=*/false, config));

    pm.addPass(
        createStructuredToMemrefPass(enableTensorFirstVectorCpu, config));
    pm.addPass(createUnstructuredToMemrefPass(config));
    pm.addPass(createTritonPtrToMemrefPass());
    pm.addPass(createTritonToPtrPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCastsPass());

    // Now that remove-dead-values fully works with linalg ops, clean up the IR
    // again, particularly unused loop iter-args that were created
    // during triton-to-structured.
    if (config.enableBOSCAME)
      // Preserve private function signatures and their argument/result attrs;
      // only simplify values inside each function in this mode.
      pm.addNestedPass<func::FuncOp>(createRemoveDeadValuesPass());
    else
      pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (enableCollapseShape) {
      // Canonicalizer pass will rewrite tensor.expand_shape(linalg.fill) to
      // linalg.fill(tensor.expand_shape) so we need to run it before
      // collapseShape pass
      pm.addPass(createCollapseShapePass());
    }

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
      return;
    }
    if (config.enableBOSCAME) {
      annotateStaticZeroTailSelects(moduleOp);
      annotateComputeGraph(moduleOp);
    }
    if (dumpFacts)
      printLinalgFacts(moduleOp, llvm::errs());
    if (!config.enableBOSCAME)
      return;
    // The experimental entry uses typed unranked descriptors. Emit the real
    // MLIR C wrapper for public kernels; conversion of arguments/program IDs
    // has already happened once in the common pipeline.
    moduleOp.walk([&](func::FuncOp function) {
      if (!function.isExternal() && function.isPublic())
        function->setAttr("llvm.emit_c_interface",
                          UnitAttr::get(&getContext()));
    });
    int64_t markedMatmuls = 0;
    moduleOp.walk([&](linalg::MatmulOp op) {
      markedMatmuls += op->hasAttr("bosc_ame.lower_to_ame");
    });
    moduleOp->setAttr(
        "bosc_ame.expected_ame_matmuls",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), markedMatmuls));
    // This pass ends at tensor Linalg with ordinary load snapshots and stores.
    // Bufferization and the independent BOSCAME planner are explicit later
    // stages; no SPM memory space or DMA is introduced here.
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgExperimentalPass() {
  return std::make_unique<TritonToLinalgExperimentalPass>();
}
