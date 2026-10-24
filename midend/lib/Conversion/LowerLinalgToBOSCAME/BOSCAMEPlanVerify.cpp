//===- BOSCAMEPlanVerify.cpp - Verify concrete BOSCAME access plans
//--------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/KernelContract.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanIR.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMETargetInfo.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace ::buddy::boscame;

namespace {
DictionaryAttr dict(Operation *op, StringRef name) {
  return op->getAttrOfType<DictionaryAttr>(name);
}
DictionaryAttr dict(DictionaryAttr attr, StringRef name) {
  return attr ? attr.getAs<DictionaryAttr>(name) : DictionaryAttr();
}
StringRef string(DictionaryAttr attr, StringRef name) {
  auto value = attr ? attr.getAs<StringAttr>(name) : StringAttr();
  return value ? value.getValue() : StringRef();
}
int64_t integer(DictionaryAttr attr, StringRef name) {
  auto value = attr ? attr.getAs<IntegerAttr>(name) : IntegerAttr();
  return value ? value.getInt() : -1;
}
int64_t bytes(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() % 8 ? 0 : integer.getWidth() / 8;
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth() % 8 ? 0 : floating.getWidth() / 8;
  return 0;
}
Value root(Value value) {
  auto origin = spmdma::getStaticPointerOrigin(value);
  return origin ? origin->root : Value();
}
bool privateRoot(Value value) {
  return value && value.getDefiningOp() &&
         isa<memref::AllocOp, memref::AllocaOp>(value.getDefiningOp());
}
bool argumentRoot(Value value) {
  auto argument = dyn_cast_or_null<BlockArgument>(value);
  return argument && isa<func::FuncOp>(argument.getOwner()->getParentOp());
}
bool nestedPlan(Attribute attribute) {
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute))
    return llvm::any_of(dictionary, [](NamedAttribute field) {
      return field.getName().strref().starts_with("plan.") ||
             nestedPlan(field.getValue());
    });
  if (auto array = dyn_cast<ArrayAttr>(attribute))
    return llvm::any_of(array, nestedPlan);
  return false;
}
// Offsets are relative to the aligned pointer, in bytes. A view may also
// change that pointer; do not confuse its byte shift with an element offset.
std::optional<std::pair<int64_t, int64_t>> byteBounds(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  auto origin = spmdma::getStaticPointerOrigin(value);
  auto offset = spmdma::getStaticDescriptorOffset(value);
  SmallVector<int64_t> strides;
  int64_t ignored, begin, end;
  int64_t width = type ? bytes(type.getElementType()) : 0;
  if (!type || !type.hasStaticShape() || !width || !origin || !offset ||
      failed(type.getStridesAndOffset(strides, ignored)) ||
      llvm::MulOverflow(*offset, width, begin) ||
      llvm::AddOverflow(begin, origin->byteShift, begin))
    return std::nullopt;
  end = begin;
  for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
    int64_t delta;
    if (dim <= 0 || stride <= 0 || llvm::MulOverflow(dim - 1, stride, delta) ||
        llvm::MulOverflow(delta, width, delta) ||
        llvm::AddOverflow(end, delta, end))
      return std::nullopt;
  }
  if (llvm::AddOverflow(end, width, end))
    return std::nullopt;
  return std::make_pair(begin, end);
}
// Placement replaces fresh logical storage with fixed physical addresses.
// Every alias must remain in the owning scope and complete its memory access
// here. Otherwise a retained address or asynchronous operation can outlive the
// scope or observe a later reuse that was impossible for the original root.
bool localCompletedSPMUses(Value storage, Region *scope) {
  SmallVector<Value> work{storage};
  DenseSet<Value> visited;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (!scope->isAncestor(user->getParentRegion()))
        return false;
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != value || user->getNumRegions() != 0 ||
            !isMemoryEffectFree(user))
          return false;
        for (Value alias : user->getResults()) {
          if (!isa<BaseMemRefType>(alias.getType()))
            continue;
          auto type = dyn_cast<MemRefType>(alias.getType());
          SmallVector<int64_t> strides;
          int64_t offset;
          if (!type || !type.hasStaticShape() ||
              failed(type.getStridesAndOffset(strides, offset)) ||
              llvm::any_of(strides,
                           [](int64_t stride) { return stride <= 0; }) ||
              !spmdma::getStaticDescriptorOffset(alias) ||
              !spmdma::getStaticPointerOrigin(alias))
            return false;
          work.push_back(alias);
        }
        continue;
      }
      if (isa<memref::CopyOp, memref::LoadOp, memref::StoreOp,
              memref::DeallocOp, memref::DimOp, vector::LoadOp,
              vector::StoreOp>(user))
        continue;
      if (isa<linalg::FillOp, linalg::GenericOp, linalg::MatmulOp,
              linalg::ReduceOp>(user)) {
        bool pure = true;
        for (Region &region : user->getRegions())
          region.walk([&](Operation *payload) {
            if (!isMemoryEffectFree(payload))
              pure = false;
          });
        if (pure)
          continue;
      }
      // Calls, yielded pointers, address extraction and asynchronous effects
      // need an independent escape/completion proof absent from this recipe.
      return false;
    }
  }
  return true;
}
Operation *unsupportedSPMAncestor(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<func::FuncOp>(parent))
      return nullptr;
    if (!isa<scf::ForOp, scf::IfOp>(parent))
      return parent;
  }
  return op;
}
bool exclusive(Operation *a, Operation *b) {
  for (Operation *parent = a->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto branch = dyn_cast<scf::IfOp>(parent)) {
      Region *aRegion = nullptr, *bRegion = nullptr;
      for (Region &region : branch->getRegions()) {
        if (region.isAncestor(a->getParentRegion()))
          aRegion = &region;
        if (region.isAncestor(b->getParentRegion()))
          bRegion = &region;
      }
      if (aRegion && bRegion && aRegion != bRegion && a->getNumResults() == 1 &&
          b->getNumResults() == 1 &&
          localCompletedSPMUses(a->getResult(0), aRegion) &&
          localCompletedSPMUses(b->getResult(0), bRegion))
        return true;
    }
  return false;
}

class PlanVerifier {
public:
  explicit PlanVerifier(ModuleOp module) : module(module) {}
  LogicalResult run() {
    if (module->hasAttr("bosc_ame.store_policy"))
      return module.emitError(
          "bosc_ame.store_policy is obsolete; remove it and "
          "rerun the automatic BOSCAME planner");
    module.walk([&](Operation *op) { verifySchema(op); });
    if (invalid)
      return failure();
    if (failed(::buddy::contract::verifyKernelContract(module, true)))
      return failure();
    auto version = module->getAttrOfType<IntegerAttr>("plan.version");
    auto profile = module->getAttrOfType<StringAttr>("plan.target_profile");
    require(module,
            version && version.getInt() == 1 && profile &&
                profile.getValue() == "nr-fpga",
            "requires plan.version=1 and plan.target_profile=nr-fpga");
    for (auto function : module.getOps<func::FuncOp>())
      if (!function.isExternal())
        verifyFunction(function);
    return failure(invalid);
  }

private:
  void require(Operation *op, bool condition, const Twine &message) {
    if (!condition) {
      op->emitError() << "invalid BOSCAME plan: " << message;
      invalid = true;
    }
  }
  void fields(Operation *op, DictionaryAttr value, ArrayRef<StringRef> integers,
              ArrayRef<StringRef> strings = {},
              ArrayRef<StringRef> booleans = {}) {
    if (!value) {
      require(op, false, "execution plan must be a dictionary");
      return;
    }
    for (NamedAttribute field : value) {
      StringRef key = field.getName().strref();
      if (llvm::is_contained(integers, key)) {
        auto number = dyn_cast<IntegerAttr>(field.getValue());
        require(op, number && number.getType().isInteger(64),
                "plan field must have i64 type: " + key);
      } else if (llvm::is_contained(strings, key)) {
        require(op, isa<StringAttr>(field.getValue()),
                "plan field must have string type: " + key);
      } else if (llvm::is_contained(booleans, key)) {
        require(op, isa<BoolAttr>(field.getValue()),
                "plan field must have bool type: " + key);
      }
    }
  }
  void selectionSchema(Operation *op, DictionaryAttr p) {
    fields(op, p, {"operand", "bank", "offset", "bytes"}, {"level", "reason"});
    require(op,
            p && p.getAs<IntegerAttr>("operand") &&
                p.getAs<StringAttr>("level"),
            "operand selection requires operand and level");
  }
  void verifySchema(Operation *op) {
    if (isa<ModuleOp>(op))
      require(op, op == module.getOperation(),
              "nested modules require separate BOSCAME plans");
    auto logicalMemory = [&](Type type) {
      if (auto memref = dyn_cast<BaseMemRefType>(type)) {
        Attribute space = memref.getMemorySpace();
        auto zero = dyn_cast_or_null<IntegerAttr>(space);
        require(
            op, !space || (zero && zero.getInt() == 0),
            "planned memrefs must retain logical memory space until emission");
      }
    };
    for (Type type : op->getOperandTypes())
      logicalMemory(type);
    for (Type type : op->getResultTypes())
      logicalMemory(type);
    if (auto function = dyn_cast<func::FuncOp>(op))
      for (Type type : function.getArgumentTypes())
        logicalMemory(type);
    for (NamedAttribute attr : op->getAttrs()) {
      require(
          op, !nestedPlan(attr.getValue()),
          "plan attributes must be attached directly to supported operations");
      StringRef name = attr.getName().strref();
      if (!name.starts_with("plan."))
        continue;
      bool carrier =
          ((name == "plan.version" || name == "plan.target_profile") &&
           isa<ModuleOp>(op)) ||
          ((name == "plan.budget" || name == "plan.task_map" ||
            name == "plan.source_task_count") &&
           isa<func::FuncOp>(op)) ||
          (name == "plan.buffer" &&
           isa<memref::AllocOp, memref::AllocaOp>(op)) ||
          (name == "plan.transfer" && isa<memref::CopyOp>(op)) ||
          (name == "plan.route" && isa<linalg::MatmulOp>(op)) ||
          (name == "plan.compute" && isa<linalg::MatmulOp>(op)) ||
          (name == "plan.fill" && isa<linalg::FillOp>(op)) ||
          (name == "plan.vector" && isa<linalg::GenericOp>(op)) ||
          (name == "plan.reduce" && isa<linalg::ReduceOp>(op)) ||
          (name == "plan.loop" && isa<scf::ForOp>(op)) ||
          (name == "plan.scope" && isa<scf::IfOp>(op));
      require(op, carrier, "unknown plan or unsupported carrier: " + name);
      if (name == "plan.version") {
        auto version = dyn_cast<IntegerAttr>(attr.getValue());
        require(op, version && version.getType().isInteger(32),
                "plan.version must have i32 type");
        continue;
      }
      if (name == "plan.target_profile") {
        require(op, isa<StringAttr>(attr.getValue()),
                "plan.target_profile must have string type");
        continue;
      }
      if (name == "plan.source_task_count") {
        auto count = dyn_cast<IntegerAttr>(attr.getValue());
        require(op,
                count && count.getType().isInteger(64) && count.getInt() >= 0,
                "plan.source_task_count must be a nonnegative i64");
        continue;
      }
      if (name == "plan.task_map") {
        auto mapping = dyn_cast<ArrayAttr>(attr.getValue());
        require(op, bool(mapping), "plan.task_map must be an array");
        if (mapping)
          for (Attribute entry : mapping)
            fields(op, dyn_cast<DictionaryAttr>(entry), {"old_id", "new_id"},
                   {"action"});
        continue;
      }
      auto p = dyn_cast<DictionaryAttr>(attr.getValue());
      fields(op, p,
             {"bank", "offset", "capacity", "channel", "vector_bytes", "rows",
              "row_bytes", "elements", "axis", "trips", "spm_banks",
              "bank_capacity"},
             {"level", "role", "engine", "from", "to", "guard", "wait", "kind",
              "tail", "fp_policy", "strategy", "reason", "order"},
             {"spm_window_reuse", "share_spm_windows"});
      if (!p)
        continue;
      if (name == "plan.buffer")
        require(op, p.getAs<StringAttr>("role") && p.getAs<StringAttr>("level"),
                "buffer placement requires level and role");
      if (name == "plan.budget")
        require(op, bool(p.getAs<DenseI64ArrayAttr>("high_water")),
                "plan budget high_water must be an i64 array");
      if (name == "plan.transfer")
        fields(op, dict(p, "chunks"),
               {"count", "bytes", "ddr_stride", "spm_stride"});
      if (name == "plan.transfer")
        require(op, bool(p.getAs<StringAttr>("role")),
                "transfer plan requires a role");
      if (name == "plan.route") {
        fields(op, dict(p, "tile"), {"m", "n", "k"});
        fields(op, dict(p, "subtiles"), {"n", "k"});
        for (StringRef key : {"a", "b", "accumulator", "output"})
          selectionSchema(op, dict(p, key));
      }
      if (name == "plan.compute")
        require(op, bool(p.getAs<DenseI64ArrayAttr>("shape")),
                "scalar matmul shape must be an i64 array");
      if (name == "plan.vector" || name == "plan.compute") {
        auto operands = p.getAs<ArrayAttr>("operands");
        require(op, bool(operands), "compute operands must be an array");
        if (operands)
          for (Attribute operand : operands)
            selectionSchema(op, dyn_cast<DictionaryAttr>(operand));
      }
      if (name == "plan.reduce") {
        selectionSchema(op, dict(p, "input"));
        selectionSchema(op, dict(p, "output"));
      }
    }
  }
  DictionaryAttr placement(Value value) {
    auto found = placements.find(root(value));
    return found == placements.end() ? DictionaryAttr() : found->second;
  }
  bool staticType(Value value, SmallVectorImpl<int64_t> &strides) {
    auto type = dyn_cast<MemRefType>(value.getType());
    int64_t offset;
    return type && type.hasStaticShape() &&
           succeeded(type.getStridesAndOffset(strides, offset)) &&
           llvm::all_of(strides, [](int64_t s) { return s > 0; }) &&
           bytes(type.getElementType()) > 0;
  }
  bool aligned(Value value) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || !spmdma::isEndpointPointerAlignmentProven(value, 64))
      return false;
    Value storage = root(value);
    if (privateRoot(storage)) {
      auto p = placement(value);
      return string(p, "level") == "spm" && integer(p, "offset") % 64 == 0;
    }
    auto abi = dict(module, "contract.abi");
    int64_t alignment = integer(abi, "args_aligned_bytes");
    return argumentRoot(storage) && alignment >= 64 && alignment % 64 == 0 &&
           buddy::contract::hasEntryArgumentABI(
               storage.getParentRegion()->getParentOfType<func::FuncOp>());
  }
  void bounds(Operation *op, Value value) {
    auto p = placement(value);
    auto range = byteBounds(value);
    if (string(p, "level") == "spm") {
      require(
          op,
          range && range->first >= 0 && range->second <= integer(p, "capacity"),
          "SPM view exceeds its planned capacity or is not statically bounded");
    } else if (argumentRoot(root(value))) {
      auto rootType = dyn_cast<MemRefType>(root(value).getType());
      // Unranked ABI arguments do not carry allocation extents. The frontend
      // provides a valid effective view; DMA coverage is checked against it.
      if (rootType && rootType.hasStaticShape()) {
        auto storage = byteBounds(root(value));
        require(op,
                range && storage && range->first >= storage->first &&
                    range->second <= storage->second,
                "DDR view exceeds its statically known argument storage");
      }
    }
  }
  void binding(Operation *op, DictionaryAttr selection, unsigned index) {
    if (!selection || index >= op->getNumOperands()) {
      require(op, false, "missing route operand binding");
      return;
    }
    require(op, integer(selection, "operand") == index,
            "route operand index does not match the actual operation");
    Value value = op->getOperand(index);
    auto p = placement(value);
    StringRef level = string(selection, "level");
    if (level == "ddr") {
      require(op, argumentRoot(root(value)),
              "direct DDR operand must refer to an external argument");
      require(op, aligned(value),
              "direct DDR address is not proven 64-byte aligned");
      bounds(op, value);
    } else if (level == "host") {
      auto type = dyn_cast<MemRefType>(value.getType());
      require(op, string(p, "level") == "host" && type && type.getRank() == 0,
              "host route requires a planned rank-zero buffer");
    } else if (level == "spm") {
      require(op,
              string(p, "level") == "spm" &&
                  integer(selection, "bank") == integer(p, "bank") &&
                  integer(selection, "offset") == integer(p, "offset"),
              "SPM route does not match its root placement");
      require(op, aligned(value), "SPM operand is not proven 64-byte aligned");
      bounds(op, value);
    } else {
      require(op, false, "unsupported operand route");
    }
  }
  void verifyFunction(func::FuncOp function) {
    placements.clear();
    SmallVector<Operation *> allocations;
    function.walk([&](Operation *op) {
      if (!isa<memref::AllocOp, memref::AllocaOp>(op))
        return;
      auto p = dict(op, "plan.buffer");
      auto type = dyn_cast<MemRefType>(op->getResult(0).getType());
      require(op, p && type && type.hasStaticShape(),
              "every private allocation needs a static plan.buffer");
      if (!p || !type || !type.hasStaticShape())
        return;
      placements[op->getResult(0)] = p;
      allocations.push_back(op);
      int64_t capacity = integer(p, "capacity");
      SmallVector<int64_t> strides;
      int64_t offset, needed = 1;
      bool legal = bytes(type.getElementType()) &&
                   succeeded(type.getStridesAndOffset(strides, offset)) &&
                   offset == 0;
      for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
        int64_t delta;
        if (dim <= 0 || stride <= 0 ||
            llvm::MulOverflow(dim - 1, stride, delta) ||
            llvm::AddOverflow(needed, delta, needed))
          legal = false;
      }
      legal &= !llvm::MulOverflow(needed, bytes(type.getElementType()), needed);
      require(op, legal && capacity >= needed,
              "planned buffer capacity is smaller than its storage layout");
      StringRef level = string(p, "level");
      if (level == "host") {
        require(op, type.getRank() == 0 && legal,
                "host placement only supports static rank-zero storage");
        for (Operation *parent = op->getParentOp();
             parent && parent != function; parent = parent->getParentOp())
          require(
              op, isa<scf::IfOp>(parent),
              "host allocation in a repeated or unknown scope is unsupported");
        Value storage = op->getResult(0);
        function.walk([&](Operation *user) {
          for (Value operand : user->getOperands()) {
            if (!isa<BaseMemRefType>(operand.getType()) ||
                root(operand) != storage)
              continue;
            bool allowed = isa<memref::LoadOp, memref::DeallocOp>(user);
            if (auto fill = dyn_cast<linalg::FillOp>(user))
              allowed = fill.getOutputs()[0] == operand;
            if (auto reduce = dyn_cast<linalg::ReduceOp>(user))
              allowed = llvm::is_contained(reduce.getInits(), operand);
            require(
                user, allowed,
                "host rank-zero allocation must remain private and not escape");
          }
        });
      } else {
        int64_t bank = integer(p, "bank"), offset = integer(p, "offset");
        require(op,
                level == "spm" && bank >= 0 && bank < 2 && offset >= 0 &&
                    offset % 64 == 0 && capacity > 0 && capacity <= 131072 &&
                    offset <= 131072 - capacity,
                "SPM bank, offset, or capacity exceeds nr-fpga limits");
        require(op,
                localCompletedSPMUses(op->getResult(0), &function.getBody()),
                "SPM root requires non-escaping aliases and completed local "
                "memory accesses");
      }
    });
    // Private logical roots are disjoint before placement, but a callee may
    // use the same physical SPM addresses. V1 has no call-frame/workspace
    // contract, including for opaque callees, so do not turn that absence of
    // logical aliasing into an interprocedural physical ownership proof.
    bool hasSPMWorkspace = llvm::any_of(allocations, [&](Operation *op) {
      return string(dict(op, "plan.buffer"), "level") == "spm";
    });
    if (hasSPMWorkspace) {
      require(function, SymbolTable::symbolKnownUseEmpty(function, module),
              "SPM function has module-local symbol uses; v1 has no "
              "interprocedural workspace ownership proof");
      function.walk([&](Operation *op) {
        if (isa<func::CallOp, func::CallIndirectOp>(op))
          require(op, false,
                  "call may clobber live SPM workspace; v1 has no "
                  "interprocedural workspace proof");
        // A synchronous operation can still race another iteration. Fixed
        // addresses are invocation-local only in the supported serial scopes,
        // including when the allocation itself is outside the concurrent op.
        bool usesSPM = llvm::is_contained(allocations, op) &&
                       string(dict(op, "plan.buffer"), "level") == "spm";
        for (Value operand : op->getOperands())
          if (isa<BaseMemRefType>(operand.getType()) &&
              string(placements.lookup(root(operand)), "level") == "spm")
            usesSPM = true;
        if (usesSPM)
          if (Operation *ancestor = unsupportedSPMAncestor(op))
            require(op, false,
                    "fixed SPM workspace requires serial execution inside "
                    "func.func; unsupported ancestor " +
                        ancestor->getName().getStringRef());
      });
    }
    int64_t highWater[2] = {0, 0};
    for (auto [i, op] : llvm::enumerate(allocations)) {
      auto a = dict(op, "plan.buffer");
      if (string(a, "level") != "spm")
        continue;
      int64_t bank = integer(a, "bank"), start = integer(a, "offset");
      int64_t end;
      if (llvm::AddOverflow(start, integer(a, "capacity"), end)) {
        require(op, false, "SPM window end overflows");
        continue;
      }
      if (bank >= 0 && bank < 2)
        highWater[bank] = std::max(highWater[bank], end);
      for (Operation *other : ArrayRef(allocations).drop_front(i + 1)) {
        auto b = dict(other, "plan.buffer");
        if (string(b, "level") != "spm" || bank != integer(b, "bank"))
          continue;
        int64_t bStart = integer(b, "offset"), bEnd;
        if (llvm::AddOverflow(bStart, integer(b, "capacity"), bEnd)) {
          require(other, false, "SPM window end overflows");
          continue;
        }
        require(op, end <= bStart || bEnd <= start || exclusive(op, other),
                "overlapping SPM windows require mutually exclusive scopes");
      }
    }
    auto budget = dict(function, "plan.budget");
    auto water = budget ? budget.getAs<DenseI64ArrayAttr>("high_water")
                        : DenseI64ArrayAttr();
    require(function,
            integer(budget, "spm_banks") == 2 &&
                integer(budget, "bank_capacity") == 131072 && water &&
                water.asArrayRef() == ArrayRef<int64_t>(highWater),
            "plan.budget does not match the physical windows");
    DenseSet<int64_t> taskIDs, oldIDs;
    function.walk([&](Operation *op) {
      if (auto task = dict(op, "contract.task"))
        taskIDs.insert(integer(task, "id"));
    });
    auto mapping = function->getAttrOfType<ArrayAttr>("plan.task_map");
    auto sourceCount =
        function->getAttrOfType<IntegerAttr>("plan.source_task_count");
    if (mapping || sourceCount)
      require(function,
              mapping && sourceCount && sourceCount.getInt() >= 0 &&
                  uint64_t(sourceCount.getInt()) == mapping.size(),
              "source task mapping coverage mismatch");
    if (mapping)
      for (Attribute entry : mapping) {
        auto item = cast<DictionaryAttr>(entry);
        int64_t oldID = integer(item, "old_id");
        StringRef action = string(item, "action");
        require(function, oldID >= 0 && oldIDs.insert(oldID).second,
                "source task mapping IDs must be nonnegative and unique");
        require(function,
                action == "retained" || action == "folded-load" ||
                    action == "folded-store" || action == "eliminated",
                "unsupported source task mapping action");
        require(function,
                action == "eliminated"
                    ? !item.get("new_id")
                    : taskIDs.contains(integer(item, "new_id")),
                "invalid source task mapping target");
      }
    function.walk([&](Operation *op) { verifyOperation(op); });
  }
  void verifyOperation(Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      return;
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      verifyTransfer(copy);
      return;
    }
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      auto p = dict(op, "plan.fill");
      Value value = fill.getOutputs()[0];
      auto type = cast<MemRefType>(value.getType());
      if (string(p, "level") == "host") {
        require(op,
                type.getRank() == 0 && string(p, "engine") == "scalar" &&
                    string(p, "kind") == "init" &&
                    string(placement(value), "level") == "host",
                "scalar fill requires a planned host rank-zero init");
      } else {
        SmallVector<int64_t> strides;
        int64_t rows = type.getRank() == 2 ? type.getDimSize(0) : 1;
        int64_t rowBytes = 0, pitch = 0;
        bool rowSize =
            type.getRank() &&
            !llvm::MulOverflow(type.getShape().back(),
                               bytes(type.getElementType()), rowBytes);
        require(
            op,
            p && string(p, "level") == "spm" && string(p, "engine") == "rvv" &&
                integer(p, "vector_bytes") == 64 &&
                (string(p, "kind") == "padding" ||
                 string(p, "kind") == "broadcast" ||
                 string(p, "kind") == "overwrite") &&
                type.getRank() >= 1 && type.getRank() <= 2 &&
                integer(p, "rows") == rows && rowSize &&
                integer(p, "row_bytes") == rowBytes && rowBytes > 0 &&
                rowBytes % 64 == 0 && staticType(value, strides) &&
                strides.back() == 1 &&
                (strides.size() == 1 ||
                 (!llvm::MulOverflow(strides[0], bytes(type.getElementType()),
                                     pitch) &&
                  pitch % 64 == 0)) &&
                string(placement(value), "level") == "spm" && aligned(value),
            "unsupported RVV fill shape, layout, or plan");
        bounds(op, value);
      }
      return;
    }
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
      verifyMatmul(matmul);
      return;
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      auto p = dict(op, "plan.vector");
      auto operands = p ? p.getAs<ArrayAttr>("operands") : ArrayAttr();
      int64_t elements = integer(p, "elements");
      Type elementType;
      require(op,
              p && string(p, "engine") == "rvv" &&
                  integer(p, "vector_bytes") == 64 &&
                  string(p, "tail") == "none" &&
                  string(p, "fp_policy") == "preserve" && operands &&
                  operands.size() == op->getNumOperands(),
              "missing or unsupported pointwise vector plan");
      if (operands)
        for (auto [i, selection] : llvm::enumerate(operands))
          if (i < op->getNumOperands()) {
            binding(op, dyn_cast<DictionaryAttr>(selection), i);
            auto type = dyn_cast<MemRefType>(op->getOperand(i).getType());
            SmallVector<int64_t> strides;
            int64_t extentBytes = 0;
            if (!elementType && type)
              elementType = type.getElementType();
            require(
                op,
                type && type.getRank() == 1 && elements > 0 &&
                    type.getDimSize(0) == elements &&
                    type.getElementType() == elementType &&
                    staticType(op->getOperand(i), strides) && strides[0] == 1 &&
                    !llvm::MulOverflow(elements, bytes(type.getElementType()),
                                       extentBytes) &&
                    extentBytes % 64 == 0,
                "pointwise plan requires equal-shaped, same-type contiguous "
                "complete 64-byte vectors");
          }
      Block &body = generic.getRegion().front();
      bool payload =
          body.getOperations().size() == 2 &&
          isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AddFOp,
              arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::NegFOp>(
              body.front());
      require(op,
              payload && generic.getOutputs().size() == 1 &&
                  llvm::all_of(generic.getIndexingMapsArray(),
                               [](AffineMap m) { return m.isIdentity(); }) &&
                  llvm::all_of(generic.getIteratorTypesArray(),
                               [](utils::IteratorType i) {
                                 return i == utils::IteratorType::parallel;
                               }),
              "pointwise emission requires a single arith operation and "
              "identity parallel indexing");
      if (payload) {
        Operation &arithmetic = body.front();
        auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
        require(op,
                arithmetic.getNumResults() == 1 &&
                    arithmetic.getResult(0).getType() == elementType && yield &&
                    yield.getValues().size() == 1 &&
                    yield.getValues()[0] == arithmetic.getResult(0),
                "generic must yield its arithmetic result");
        for (Value operand : arithmetic.getOperands())
          require(
              op,
              isa<BlockArgument>(operand) &&
                  cast<BlockArgument>(operand).getOwner() == &body,
              "generic scalar captures must be passed as explicit operands");
      }
      return;
    }
    if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
      auto p = dict(op, "plan.reduce");
      require(
          op,
          string(p, "engine") == "rvv" && integer(p, "vector_bytes") == 64 &&
              string(p, "strategy") == "ordered-vector-add" &&
              string(p, "fp_policy") == "preserve" && integer(p, "axis") == 0 &&
              reduce.getDimensions() == ArrayRef<int64_t>{0} &&
              reduce.getInputs().size() == 1 && reduce.getInits().size() == 1,
          "unsupported ordered reduction plan");
      if (reduce.getInputs().size() != 1 || reduce.getInits().size() != 1)
        return;
      binding(op, dict(p, "input"), 0);
      binding(op, dict(p, "output"), 1);
      auto input = cast<MemRefType>(reduce.getInputs()[0].getType());
      auto output = cast<MemRefType>(reduce.getInits()[0].getType());
      SmallVector<int64_t> strides;
      require(
          op,
          input.getRank() == 1 && input.getElementType().isF32() &&
              output.getRank() == 0 && output.getElementType().isF32() &&
              integer(p, "elements") == input.getDimSize(0) &&
              input.getDimSize(0) > 0 && input.getDimSize(0) % 16 == 0 &&
              staticType(reduce.getInputs()[0], strides) && strides[0] == 1 &&
              string(placement(reduce.getInits()[0]), "level") == "host",
          "reduction requires contiguous f32 input and host rank-zero output");
      Block &body = reduce.getRegion().front();
      auto add = body.empty() ? arith::AddFOp()
                              : dyn_cast<arith::AddFOp>(body.front());
      auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
      require(op,
              body.getOperations().size() == 2 && body.getNumArguments() == 2 &&
                  add && yield && yield.getValues().size() == 1 &&
                  yield.getValues()[0] == add.getResult() &&
                  add.getFastmath() == arith::FastMathFlags::none &&
                  ((add.getLhs() == body.getArgument(0) &&
                    add.getRhs() == body.getArgument(1)) ||
                   (add.getRhs() == body.getArgument(0) &&
                    add.getLhs() == body.getArgument(1))),
              "reduction payload must yield one ordered add of input and "
              "accumulator");
      return;
    }
    if (isa<linalg::LinalgOp>(op))
      require(op, false, "unplanned Linalg operation");
    if (auto load = dyn_cast<memref::LoadOp>(op))
      require(op,
              cast<MemRefType>(load.getMemRefType()).getRank() == 0 &&
                  string(placement(load.getMemRef()), "level") == "host",
              "scalar load must access planned host rank-zero storage");
    if (auto store = dyn_cast<memref::StoreOp>(op))
      require(op,
              cast<MemRefType>(store.getMemRef().getType()).getRank() == 0 &&
                  string(placement(store.getMemRef()), "level") == "host",
              "v1 has no independent external or SPM scalar-store recipe; "
              "use a supported Linalg output and boundary copy");
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto p = dict(op, "plan.loop");
      auto cf = dict(op, "contract.cf");
      if (!cf || !cf.getAs<IntegerAttr>("trip_count")) {
        auto graph =
            dict(op->getParentOfType<func::FuncOp>(), "contract.compute_graph");
        auto complete = graph ? graph.getAs<BoolAttr>("complete") : BoolAttr();
        require(op, !p && complete && !complete.getValue(),
                "opaque loop requires an incomplete contract and no window "
                "reuse plan");
        return;
      }
      require(op,
              p && p.getAs<BoolAttr>("spm_window_reuse") &&
                  integer(p, "trips") > 0 &&
                  integer(p, "trips") == integer(cf, "trip_count"),
              "loop plan disagrees with its contract summary");
    }
    if (isa<scf::IfOp>(op)) {
      auto p = dict(op, "plan.scope");
      auto sharing = p ? p.getAs<BoolAttr>("share_spm_windows") : BoolAttr();
      require(op, string(p, "kind") == "exclusive-arms" && sharing,
              "if requires an exclusive-arms scope plan");
    }
  }
  void verifyTransfer(memref::CopyOp copy) {
    Operation *op = copy;
    auto p = dict(op, "plan.transfer"), chunks = dict(p, "chunks");
    bool load = string(p, "from") == "ddr" && string(p, "to") == "spm";
    bool store = string(p, "from") == "spm" && string(p, "to") == "ddr";
    int64_t count = integer(chunks, "count"), size = integer(chunks, "bytes");
    int64_t ddrStride = integer(chunks, "ddr_stride"),
            spmStride = integer(chunks, "spm_stride");
    bool legal =
        p && chunks && (load || store) && string(p, "engine") == "dma" &&
        string(p, "wait") == "after-each-chunk" && count > 0 && size > 0 &&
        size % 64 == 0 && ddrStride >= 0 && spmStride >= 0 &&
        (count == 1 || (ddrStride >= size && spmStride >= size &&
                        ddrStride % 64 == 0 && spmStride % 64 == 0));
    require(op, legal,
            "unsupported DMA direction, chunks, stride, or wait policy");
    if (!legal)
      return;
    Value spm = load ? copy.getTarget() : copy.getSource();
    Value ddr = load ? copy.getSource() : copy.getTarget();
    auto window = placement(spm);
    require(op,
            string(window, "level") == "spm" &&
                integer(window, "bank") == integer(p, "channel") &&
                argumentRoot(root(ddr)) && aligned(ddr) && aligned(spm),
            "DMA endpoints, channel/bank, or alignment disagree with the plan");
    bounds(op, spm);
    bounds(op, ddr);
    SmallVector<int64_t> axes;
    for (auto [value, stride] :
         {std::pair<Value, int64_t>{spm, spmStride}, {ddr, ddrStride}}) {
      auto type = dyn_cast<MemRefType>(value.getType());
      SmallVector<int64_t> strides;
      bool legal = type && staticType(value, strides);
      int64_t width = type ? bytes(type.getElementType()) : 0;
      int64_t axis = -1, extent = 0, pitch = 0, last = 0;
      if (legal && type.getRank() == 1) {
        legal = count == 1 && strides[0] == 1 &&
                !llvm::MulOverflow(type.getDimSize(0), width, extent) &&
                extent == size;
        axis = 0;
      } else if (legal && type.getRank() == 2) {
        for (int minor : {0, 1})
          if (strides[minor] == 1 &&
              !llvm::MulOverflow(type.getDimSize(minor), width, extent) &&
              extent == size && count == type.getDimSize(1 - minor) &&
              !llvm::MulOverflow(strides[1 - minor], width, pitch) &&
              (count == 1 || stride == pitch)) {
            axis = minor;
            break;
          }
        legal = axis >= 0;
      } else {
        legal = false;
      }
      legal = legal && width && size % width == 0 && stride % width == 0 &&
              !llvm::MulOverflow(count - 1, stride, last) &&
              !llvm::AddOverflow(last, size, last);
      require(op, legal && width && size % width == 0 && stride % width == 0,
              "DMA chunks do not exactly cover the operand physical layout");
      axes.push_back(axis);
    }
    auto sourceType = cast<MemRefType>(copy.getSource().getType());
    auto targetType = cast<MemRefType>(copy.getTarget().getType());
    require(
        op,
        sourceType.getShape() == targetType.getShape() &&
            sourceType.getElementType() == targetType.getElementType() &&
            axes.size() == 2 && axes[0] >= 0 && axes[0] == axes[1],
        "DMA cannot change the logical shape or transpose its contiguous axis");
  }
  void verifyMatmul(linalg::MatmulOp matmul) {
    Operation *op = matmul;
    if (auto compute = dict(op, "plan.compute")) {
      verifyOrderedMatmul(matmul, compute);
      return;
    }
    auto reference = dict(op, "contract.compute");
    require(op, !reference || !reference.get("order"),
            "an explicit ordered reference computation requires a matching "
            "scalar plan.compute recipe");
    auto p = dict(op, "plan.route"), tile = dict(p, "tile"),
         sub = dict(p, "subtiles");
    int64_t m = integer(tile, "m"), n = integer(tile, "n"),
            k = integer(tile, "k");
    int64_t ns = integer(sub, "n"), ks = integer(sub, "k");
    int64_t fullN = 0, fullK = 0;
    bool geometry = string(p, "engine") == "ame" && m > 0 && m <= 64 && n > 0 &&
                    n <= 64 && k > 0 && k <= 64 && ns > 0 && ks > 0 &&
                    !llvm::MulOverflow(n, ns, fullN) &&
                    !llvm::MulOverflow(k, ks, fullK);
    require(op, geometry, "invalid nr-fpga AME tile or subtile counts");
    if (!geometry || matmul.getNumOperands() != 3)
      return;
    binding(op, dict(p, "a"), 0);
    binding(op, dict(p, "b"), 1);
    binding(op, dict(p, "accumulator"), 2);
    binding(op, dict(p, "output"), 2);
    require(op, matmul.getCast() == linalg::TypeFn::cast_signed,
            "AME requires signed i8 multiplication");
    auto a = cast<MemRefType>(matmul.getInputs()[0].getType());
    auto b = cast<MemRefType>(matmul.getInputs()[1].getType());
    auto c = cast<MemRefType>(matmul.getOutputs()[0].getType());
    for (Value input : matmul.getInputs()) {
      bool disjoint = root(input) != root(matmul.getOutputs()[0]);
      if (!disjoint) {
        auto inputRange = byteBounds(input);
        auto outputRange = byteBounds(matmul.getOutputs()[0]);
        disjoint = inputRange && outputRange &&
                   (inputRange->second <= outputRange->first ||
                    outputRange->second <= inputRange->first);
      }
      require(op, disjoint, "AME accumulator may overlap an input operand");
    }
    require(op,
            a.getRank() == 2 && b.getRank() == 2 && c.getRank() == 2 &&
                a.getElementType().isInteger(8) &&
                b.getElementType().isInteger(8) &&
                c.getElementType().isInteger(32) &&
                a.getShape() == ArrayRef<int64_t>{m, fullK} &&
                b.getShape() == ArrayRef<int64_t>{fullK, fullN} &&
                c.getShape() == ArrayRef<int64_t>{m, fullN},
            "AME tile/subtiles disagree with operand shapes or element types");
    AmeTileLayout layouts[] = {AmeTileLayout::MxKWithKContiguous,
                               AmeTileLayout::KxNWithPhysicalNxK,
                               AmeTileLayout::MxNWithNContiguous};
    for (auto [i, value] : llvm::enumerate(matmul.getOperands())) {
      auto type = cast<MemRefType>(value.getType());
      SmallVector<int64_t> strides;
      bool legal =
          staticType(value, strides) && hasAmeTileLayout(type, layouts[i]);
      if (legal) {
        int64_t pitch = 0;
        legal &= !llvm::MulOverflow(strides[i == 1 ? 1 : 0],
                                    bytes(type.getElementType()), pitch) &&
                 pitch % 64 == 0;
        if (i == 0 && ks > 1)
          legal &= k * bytes(type.getElementType()) % 64 == 0;
        if (i == 1 && ks > 1)
          legal &= k * bytes(type.getElementType()) % 64 == 0;
        if (i == 2 && ns > 1)
          legal &= n * bytes(type.getElementType()) % 64 == 0;
      }
      require(op, legal, "AME tile addresses or parent pitch are not legal");
    }
  }
  void verifyOrderedMatmul(linalg::MatmulOp matmul, DictionaryAttr compute) {
    Operation *op = matmul;
    if (auto reference = dict(op, "contract.compute");
        reference && reference.get("order"))
      require(op,
              string(reference, "kind") == "matmul" &&
                  string(reference, "order") == string(compute, "order"),
              "scalar compute order disagrees with the neutral reference "
              "execution order");
    require(
        op,
        !op->hasAttr("plan.route") && string(compute, "engine") == "scalar" &&
            string(compute, "kind") == "matmul" &&
            string(compute, "order") == "mnk" &&
            string(compute, "fp_policy") == "preserve",
        "ordered matmul requires one scalar mnk plan preserving the payload");
    auto shape = compute.getAs<DenseI64ArrayAttr>("shape");
    auto operands = compute.getAs<ArrayAttr>("operands");
    require(
        op,
        shape && shape.size() == 3 && operands && operands.size() == 3 &&
            op->getNumOperands() == 3 && matmul.hasPureBufferSemantics(),
        "ordered matmul requires three buffer operands and an [m,n,k] shape");
    if (!shape || shape.size() != 3 || !operands || operands.size() != 3 ||
        op->getNumOperands() != 3 || !matmul.hasPureBufferSemantics())
      return;
    int64_t m = shape[0], n = shape[1], k = shape[2];
    auto a = cast<MemRefType>(matmul.getInputs()[0].getType());
    auto b = cast<MemRefType>(matmul.getInputs()[1].getType());
    auto c = cast<MemRefType>(matmul.getOutputs()[0].getType());
    require(op,
            m > 0 && n > 0 && k > 0 &&
                a.getShape() == ArrayRef<int64_t>{m, k} &&
                b.getShape() == ArrayRef<int64_t>{k, n} &&
                c.getShape() == ArrayRef<int64_t>{m, n},
            "ordered matmul shape disagrees with its static operand shapes");
    for (auto [i, value] : llvm::enumerate(matmul.getOperands())) {
      auto selection = dyn_cast<DictionaryAttr>(operands[i]);
      auto type = cast<MemRefType>(value.getType());
      auto p = placement(value);
      SmallVector<int64_t> strides;
      auto range = byteBounds(value);
      int64_t width = bytes(type.getElementType());
      require(
          op,
          selection && integer(selection, "operand") == int64_t(i) &&
              type.getRank() == 2 && staticType(value, strides) && range &&
              range->first >= 0 && width > 0 && range->first % width == 0,
          "ordered matmul operand requires a static naturally aligned view");
      StringRef level = string(selection, "level");
      if (level == "spm") {
        require(op,
                string(p, "level") == "spm" &&
                    integer(selection, "bank") == integer(p, "bank") &&
                    integer(selection, "offset") == integer(p, "offset"),
                "ordered matmul SPM operand does not match its root placement");
        bounds(op, value);
      } else if (level == "ddr") {
        require(
            op, argumentRoot(root(value)),
            "ordered matmul DDR operand must refer to an external argument");
        bounds(op, value);
      } else if (level == "host") {
        auto storage = byteBounds(root(value));
        require(op,
                privateRoot(root(value)) && string(p, "level") == "host" &&
                    range && storage && range->first >= storage->first &&
                    range->second <= storage->second,
                "ordered matmul host operand exceeds its planned private root");
      } else {
        require(op, false, "unsupported ordered matmul operand route");
      }
    }
    Block &body = matmul.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    require(op,
            body.getNumArguments() == 3 && yield &&
                yield.getValues().size() == 1,
            "ordered matmul must yield one scalar accumulator");
    for (Operation &payload : body.without_terminator())
      require(op,
              payload.getName().getDialectNamespace() == "arith" &&
                  payload.getNumRegions() == 0 && isMemoryEffectFree(&payload),
              "ordered matmul payload must contain only scalar arithmetic");
  }
  ModuleOp module;
  bool invalid = false;
  DenseMap<Value, DictionaryAttr> placements;
};

struct PlanVerifyPass : PassWrapper<PlanVerifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanVerifyPass)
  StringRef getArgument() const final { return "bosc-ame-plan-verify"; }
  StringRef getDescription() const final {
    return "Verify concrete BOSCAME physical plans";
  }
  void runOnOperation() final {
    if (failed(verifyBOSCAMEPlan(getOperation())))
      signalPassFailure();
  }
};
} // namespace

LogicalResult buddy::boscame::verifyBOSCAMEPlan(ModuleOp module) {
  return PlanVerifier(module).run();
}
bool buddy::boscame::canShareExclusiveSPMWindow(Value first, Value second) {
  return first && second && first.getDefiningOp() && second.getDefiningOp() &&
         exclusive(first.getDefiningOp(), second.getDefiningOp());
}
std::unique_ptr<Pass> buddy::boscame::createBOSCAMEPlanVerifyPass() {
  return std::make_unique<PlanVerifyPass>();
}
