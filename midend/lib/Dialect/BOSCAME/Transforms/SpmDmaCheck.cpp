//===- SpmDmaCheck.cpp - SPM / DMA lifetime and occupancy verification ----===//
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
//
// `bosc-ame-spm-dma-check` enforces the cross-operation half of the SPM/DMA
// contract; the op verifiers in `BOSCAMEDialect.cpp` cover the per-operation
// half (parameters, endpoint structure, statically derivable lengths).
//
// The pass runs after bufferization, SPM layout and DMA scheduling and before
// LLVM export.  It walks every block in execution order while keeping, for the
// block currently being walked, the set of in-flight DMA requests
// (`active[channel]`):
//
//   start  the channel must be idle, the request must not overlap an in-flight
//          request, and the token must have exactly one user (its wait).
//   op     an access that may alias the source or the destination of an
//          in-flight request is rejected, see the occupancy table below.
//   wait   must match an in-flight token of this block, then frees the channel.
//   end    a block may not end with an in-flight request.  This is also what
//          makes a failure exit drain every started channel: an early
//          `return`/`scf.yield` that still has a request in flight is an error,
//          so the error path has to wait the other channels before it leaves.
//
// Occupancy rules (plan section 8.2):
//
//   | region                | allowed while the request is in flight |
//   |-----------------------|----------------------------------------|
//   | source of the request | reads only                             |
//   | destination           | nothing, not even a read               |
//   | provably disjoint     | anything                               |
//
// A region-bearing operation is only entered when nothing is in flight; a
// request may not outlive the block that started it.  That is a deliberate
// first-version restriction: it keeps the analysis from having to merge
// completion state along control-flow paths.
//
// Conflict decisions use the *request record* ranges for known start/wait
// pairs, not the coarse resource-wide effects that exist to keep the optimizer
// from reordering the MMIO accesses; otherwise legal two-channel concurrency
// would be rejected.  Ordinary memory operations contribute their declared
// effects, their aliasing and their byte range.  When the range of an access
// cannot be traced back to a storage root the access is rejected instead of
// being optimistically assumed disjoint.
//
// Status handling is deliberately not part of this pass: `bosc_ame.dma.wait`
// has no result and the runtime performs the completion barrier and the status
// check itself (a failed wait takes the runtime error path).  There is
// therefore no "status was observed" obligation for the checker to enforce.
//
// Memory effects are a description of an operation, not of the device
// occupancy that lasts from `dma.start` to `dma.wait`, which is why the
// lifetime and range checks above are needed on top of them (plan 8.3).
//
//===----------------------------------------------------------------------===//

#include "Dialect/BOSCAME/BOSCAMEAttrs.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/Transform.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

using namespace buddy::boscame;
using namespace buddy::boscame::spmdma;

//===----------------------------------------------------------------------===//
// Byte-range analysis
//===----------------------------------------------------------------------===//

namespace {

/// Byte footprint of a buffer access, resolved against its storage root.
struct Footprint {
  /// SPM bank when the buffer lives in `#bosc_ame.spm<bank>`.
  std::optional<int64_t> bank;
  /// Storage root of a DDR footprint.  Two accesses rooted at different
  /// buffers are disjoint only under the module ABI promise.
  Value ddrRoot;
  /// Byte range inside the bank (SPM) or inside the root buffer (DDR).
  int64_t begin = 0;
  int64_t end = 0;
  /// False when the footprint contains a *runtime* base-pointer shift, i.e. a
  /// `memref.view` whose byte shift is not a compile-time constant.  That is
  /// how a regular dynamic address (a tile row, a K sub-block) is expressed:
  /// the shift is proven to be a multiple of the DMA granularity by the op
  /// verifier, but its value is unknown here, so the byte range can only be
  /// bounded conservatively.  `begin` is the static part of the chain.
  bool rangeKnown = true;
};

/// The pointer chain of a memref value, up to the storage root it was derived
/// from.  Only `memref.view` moves the base pointer: every other view-like
/// operation keeps the source pointer and encodes its offset in the descriptor,
/// which is read from the queried value's own type instead.
struct PointerChain {
  std::optional<int64_t> bank;
  Value ddrRoot;
  int64_t pointerShift = 0;
  /// True when the chain contains a `memref.view` whose byte shift is not a
  /// compile-time constant.
  bool dynamicShift = false;
};

/// An in-flight request, recorded when `bosc_ame.dma.start` is reached.
struct ActiveRequest {
  int64_t channel = 0;
  Value token;
  Operation *start = nullptr;
  Footprint src;
  Footprint dst;
};

/// Per-block bookkeeping.  A request never outlives its block, so this state
/// never has to be merged with a sibling block's state.
struct BlockAnalysis {
  std::optional<ActiveRequest> active[static_cast<size_t>(kDmaChannelCount)];
};

} // namespace

/// Print an SSA value the way it is written at its definition.
static std::string describeOperand(Value value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.printAsOperand(stream, OpPrintingFlags());
  stream.flush();
  return text;
}

/// Print a byte footprint for a diagnostic.
static std::string describeFootprint(const Footprint &footprint) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  if (footprint.bank) {
    stream << "SPM bank " << *footprint.bank;
  } else {
    stream << "DDR buffer ";
    footprint.ddrRoot.printAsOperand(stream, OpPrintingFlags());
  }
  stream << ", bytes [" << footprint.begin << ", " << footprint.end << ")";
  stream.flush();
  return text;
}

/// Descriptor offset (in elements) and length (in elements) of the region a
/// memref value can address, from its static layout.  The length is an upper
/// bound of any single access, which is what makes it a safe conflict range.
static std::optional<std::pair<int64_t, int64_t>>
getStaticElementSpan(MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  if (getSpmElementByteWidth(type.getElementType()) == 0)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (offset == ShapedType::kDynamic)
    return std::nullopt;
  if (llvm::is_contained(type.getShape(), 0))
    return std::make_pair(offset, int64_t{0});
  int64_t lastElement = 0;
  for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
    if (stride == ShapedType::kDynamic)
      return std::nullopt;
    int64_t reached = 0;
    if (llvm::MulOverflow(dim - 1, stride, reached))
      return std::nullopt;
    if (llvm::AddOverflow(lastElement, reached, lastElement))
      return std::nullopt;
  }
  return std::make_pair(offset, lastElement + 1);
}

/// Walk the defining chain of `value` to the storage root it addresses.
/// Returns std::nullopt for a chain the analysis cannot see through, in which
/// case the caller has to assume the access aliases everything.
static std::optional<PointerChain> tracePointerChain(Value value) {
  PointerChain chain;
  Value current = value;
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (Operation *defining = current.getDefiningOp()) {
    if (!visited.insert(defining).second)
      return std::nullopt;
    // `bosc_ame.spm.view` *is* the SPM anchor: it re-describes a sub-range of a
    // bank, and its offset is already bank-relative.
    if (auto view = dyn_cast<SpmViewOp>(defining)) {
      std::optional<int64_t> bank = getSpmBank(view.getType());
      if (!bank || *bank != view.getBank())
        return std::nullopt;
      chain.bank = bank;
      chain.pointerShift += view.getOffset();
      return chain;
    }
    // Only `memref.view` moves the base pointer, by `byte_shift` bytes.  A
    // constant shift is folded into the footprint; a runtime shift is accepted
    // only when it is provably a multiple of the DMA granularity (the op
    // verifier enforces the same rule) and marks the footprint's range as
    // unknown instead of exact.
    if (auto view = dyn_cast<memref::ViewOp>(defining)) {
      std::optional<int64_t> shift = getConstantIntValue(view.getByteShift());
      if (!shift) {
        if (!isProvablyMultipleOf(view.getByteShift(), kDmaAlignmentBytes))
          return std::nullopt;
        chain.dynamicShift = true;
        current = view.getSource();
        continue;
      }
      if (*shift < 0)
        return std::nullopt;
      chain.pointerShift += *shift;
      current = view.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(defining)) {
      current = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(defining)) {
      current = reinterpret.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(defining)) {
      current = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(defining)) {
      current = expand.getSrc();
      continue;
    }
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    // `memref.reshape`, `memref.transpose`, call results, ... are not traced.
    break;
  }

  auto rootType = dyn_cast<MemRefType>(current.getType());
  if (!rootType)
    return std::nullopt;
  Operation *rootOp = current.getDefiningOp();
  bool isRoot = isa<BlockArgument>(current) ||
                isa_and_nonnull<memref::AllocOp>(rootOp) ||
                isa_and_nonnull<memref::AllocaOp>(rootOp) ||
                isa_and_nonnull<memref::GetGlobalOp>(rootOp);
  if (!isRoot)
    return std::nullopt;
  chain.bank = getSpmBank(rootType);
  if (!chain.bank)
    chain.ddrRoot = current;
  return chain;
}

/// Resolve the byte range touched by an access of `length` bytes into `value`.
static std::optional<Footprint> resolveFootprint(Value value, int64_t length) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType || length <= 0)
    return std::nullopt;
  std::optional<std::pair<int64_t, int64_t>> span =
      getStaticElementSpan(memrefType);
  std::optional<PointerChain> chain = tracePointerChain(value);
  if (!span || !chain)
    return std::nullopt;
  // The address space of the value and of the root it is derived from have to
  // agree, otherwise the byte range would be attributed to the wrong storage.
  if (getSpmBank(memrefType) != chain->bank)
    return std::nullopt;
  unsigned elementBytes = getSpmElementByteWidth(memrefType.getElementType());
  int64_t begin = chain->pointerShift;
  int64_t offsetBytes = 0;
  if (llvm::MulOverflow(span->first, static_cast<int64_t>(elementBytes),
                        offsetBytes))
    return std::nullopt;
  if (llvm::AddOverflow(begin, offsetBytes, begin))
    return std::nullopt;
  int64_t end = 0;
  if (llvm::AddOverflow(begin, length, end))
    return std::nullopt;
  Footprint footprint;
  footprint.bank = chain->bank;
  footprint.ddrRoot = chain->ddrRoot;
  footprint.begin = begin;
  footprint.end = end;
  footprint.rangeKnown = !chain->dynamicShift;
  return footprint;
}

/// Resolve the whole byte range a memref value can address.
static std::optional<Footprint> resolveAccessFootprint(Value value) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return std::nullopt;
  std::optional<std::pair<int64_t, int64_t>> span =
      getStaticElementSpan(memrefType);
  if (!span)
    return std::nullopt;
  unsigned elementBytes = getSpmElementByteWidth(memrefType.getElementType());
  int64_t length = 0;
  if (llvm::MulOverflow(span->second, static_cast<int64_t>(elementBytes),
                        length))
    return std::nullopt;
  if (length == 0)
    return std::nullopt;
  return resolveFootprint(value, length);
}

/// May two footprints overlap?
static bool mayAlias(const Footprint &lhs, const Footprint &rhs,
                     bool abiDisjointDdrRoots) {
  // SPM and DDR are different storage, and two banks are different storage.
  if (lhs.bank.has_value() != rhs.bank.has_value())
    return false;
  if (lhs.bank)
    return *lhs.bank == *rhs.bank && lhs.begin < rhs.end && rhs.begin < lhs.end;
  if (lhs.ddrRoot == rhs.ddrRoot) {
    // A footprint with a runtime base-pointer shift has no exact range, so it
    // has to be taken as covering its whole root.
    if (!lhs.rangeKnown || !rhs.rangeKnown)
      return true;
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
  }
  // Two distinct pointer arguments are disjoint only under the explicit
  // module ABI promise; without it the ranges cannot be compared at all.
  return !abiDisjointDdrRoots;
}

//===----------------------------------------------------------------------===//
// Checker
//===----------------------------------------------------------------------===//

namespace {

class SpmDmaChecker {
public:
  explicit SpmDmaChecker(bool abiDisjointDdrRoots)
      : abiDisjointDdrRoots(abiDisjointDdrRoots) {}

  LogicalResult check(ModuleOp module) {
    WalkResult result = module.walk([&](func::FuncOp function) {
      return failed(checkFunction(function)) ? WalkResult::interrupt()
                                             : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

private:
  bool abiDisjointDdrRoots = false;

  static bool hasActive(const BlockAnalysis &analysis) {
    for (const std::optional<ActiveRequest> &slot : analysis.active)
      if (slot)
        return true;
    return false;
  }

  LogicalResult checkFunction(func::FuncOp function) {
    for (BlockArgument argument : function.getArguments())
      if (isa<DMATokenType>(argument.getType()))
        return function.emitOpError()
               << "takes a DMA token as an argument; a token is a local "
                  "request handle and may not cross a function boundary";
    for (Type resultType : function.getFunctionType().getResults())
      if (isa<DMATokenType>(resultType))
        return function.emitOpError()
               << "returns a DMA token; a token is a local request handle and "
                  "may not cross a function boundary";
    for (Block &block : function.getBody())
      if (failed(checkBlock(block)))
        return failure();
    return success();
  }

  LogicalResult checkBlock(Block &block) {
    BlockAnalysis analysis;
    for (Operation &op : block)
      if (failed(checkOperation(op, analysis)))
        return failure();
    for (std::optional<ActiveRequest> &slot : analysis.active) {
      if (!slot)
        continue;
      return slot->start->emitOpError()
             << "starts a request on DMA channel " << slot->channel
             << " that is never waited before the end of this block; a request "
                "may not outlive the block that started it, which is also what "
                "forces a failure exit to wait (and therefore observe) every "
                "channel it started";
    }
    return success();
  }

  LogicalResult checkOperation(Operation &op, BlockAnalysis &analysis) {
    // Regions are only entered with nothing in flight; the region body is
    // analyzed with its own fresh state.
    if (op.getNumRegions() != 0) {
      for (std::optional<ActiveRequest> &slot : analysis.active) {
        if (!slot)
          continue;
        return op.emitOpError()
               << "is reached while the request started on DMA channel "
               << slot->channel
               << " is still in flight; wait it before entering a region, so "
                  "that completion does not have to be merged along "
                  "control-flow paths";
      }
      for (Region &region : op.getRegions())
        for (Block &bodyBlock : region)
          if (failed(checkBlock(bodyBlock)))
            return failure();
      return success();
    }

    if (auto start = dyn_cast<DmaStartOp>(&op))
      return checkStart(start, analysis);
    if (auto wait = dyn_cast<DmaWaitOp>(&op))
      return checkWait(wait, analysis);

    if (failed(checkSpmStorage(op)))
      return failure();
    return checkAccess(op, analysis);
  }

  /// SPM is a fixed workspace: it is addressed through `bosc_ame.spm.view`,
  /// never allocated, freed or declared as a global.
  LogicalResult checkSpmStorage(Operation &op) {
    auto checkSpace = [&](Type type, StringRef verb) -> LogicalResult {
      std::optional<int64_t> bank = getSpmBank(type);
      if (!bank)
        return success();
      return op.emitOpError()
             << verb << " SPM bank " << *bank
             << "; SPM is a fixed kernel workspace addressed through "
                "`bosc_ame.spm.view`, it is never allocated, freed or declared "
                "as a global";
    };
    if (auto alloc = dyn_cast<memref::AllocOp>(&op))
      return checkSpace(alloc.getType(), "allocates");
    if (auto alloca = dyn_cast<memref::AllocaOp>(&op))
      return checkSpace(alloca.getType(), "allocates");
    if (auto dealloc = dyn_cast<memref::DeallocOp>(&op))
      return checkSpace(dealloc.getMemref().getType(), "frees");
    if (auto global = dyn_cast<memref::GetGlobalOp>(&op))
      return checkSpace(global.getType(), "declares a global in");
    return success();
  }

  LogicalResult checkStart(DmaStartOp start, BlockAnalysis &analysis) {
    int64_t channel = static_cast<int64_t>(start.getChannel());
    if (analysis.active[channel])
      return start.emitOpError()
             << "starts a request on DMA channel " << channel
             << " while an earlier request on the same channel is still in "
                "flight; a channel handles at most one request at a time, wait "
                "the previous one first";

    // A token is a single-use request handle: exactly one `dma.wait`, in the
    // same block, and no other user.
    Value token = start.getToken();
    llvm::SmallVector<Operation *, 2> users(token.getUsers());
    if (users.empty())
      return start.emitOpError()
             << "produces a DMA token that is never waited; a dropped request "
                "stays in flight and leaves the channel busy";
    if (users.size() != 1 || !isa<DmaWaitOp>(users.front()))
      return start.emitOpError()
             << "produces a DMA token that is not used by exactly one "
                "`bosc_ame.dma.wait`; a token is a single-use request handle "
                "and may not be copied, stored or passed to another operation";

    std::optional<int64_t> bytes = getConstantIntValue(start.getBytes());
    if (!bytes)
      return start.emitOpError()
             << "needs a statically known transfer length; the SPM/DMA check "
                "cannot bound an unknown number of bytes";
    std::optional<Footprint> src = resolveFootprint(start.getSrc(), *bytes);
    std::optional<Footprint> dst = resolveFootprint(start.getDst(), *bytes);
    if (!src || !dst)
      return start.emitOpError()
             << "uses a " << (!src ? "source" : "destination")
             << " endpoint whose byte range cannot be traced back to a storage "
                "root; the SPM/DMA analysis needs a view of a function "
                "argument, a `bosc_ame.spm.view` or an allocation";

    // A new request behaves like a read of its source and a write of its
    // destination, so it conflicts with the destination of an in-flight
    // request (write/write or read/write) but not with its source (two
    // channels may read the same staged buffer).
    for (std::optional<ActiveRequest> &slot : analysis.active) {
      if (!slot)
        continue;
      if (mayAlias(*dst, slot->src, abiDisjointDdrRoots) ||
          mayAlias(*dst, slot->dst, abiDisjointDdrRoots) ||
          mayAlias(*src, slot->dst, abiDisjointDdrRoots))
        return start.emitOpError()
               << "requests a transfer between " << describeFootprint(*src)
               << " and " << describeFootprint(*dst)
               << " that overlaps the in-flight request on DMA channel "
               << slot->channel
               << "; concurrent channels need provably disjoint destinations, "
                  "and a source may not be written by another channel";
    }

    ActiveRequest request;
    request.channel = channel;
    request.token = token;
    request.start = start.getOperation();
    request.src = *src;
    request.dst = *dst;
    analysis.active[channel] = request;
    return success();
  }

  LogicalResult checkWait(DmaWaitOp wait, BlockAnalysis &analysis) {
    Value token = wait.getToken();
    if (isa<BlockArgument>(token))
      return wait.emitOpError()
             << "waits on a DMA token that is a block argument; a token must "
                "be the result of a `bosc_ame.dma.start` in the same block";
    auto start = token.getDefiningOp<DmaStartOp>();
    if (!start)
      return wait.emitOpError() << "waits on a value that is not produced by a "
                                   "`bosc_ame.dma.start`";

    for (std::optional<ActiveRequest> &slot : analysis.active) {
      if (!slot || slot->token != token)
        continue;
      // A normal return from `bosc_ame.dma.wait` already means "completed and
      // succeeded": the runtime performs the completion barrier and checks the
      // status itself.  The checker only has to close the occupancy slot.
      slot.reset();
      return success();
    }

    if (analysis.active[start.getChannel()])
      return wait.emitOpError()
             << "does not wait the request that is in flight on DMA channel "
             << start.getChannel()
             << "; each channel holds at most one request and it must be "
                "waited through its own token";
    return wait.emitOpError()
           << "waits a token that is not in flight in this block (it was "
              "already waited, or it belongs to another block); completing a "
              "request may not cross a block boundary";
  }

  LogicalResult checkAccess(Operation &op, BlockAnalysis &analysis) {
    if (!hasActive(analysis))
      return success();

    llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
    bool hasInterface = false;
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(&op)) {
      interface.getEffects(effects);
      hasInterface = true;
    }

    if (!hasInterface) {
      // An operation that does not describe its memory effects could touch the
      // staged source or the destination through state the analysis cannot see.
      if (isa<CallOpInterface>(&op))
        return op.emitOpError()
               << "calls code with unknown memory effects while a DMA request "
                  "is in flight; a call may touch the staged source or the "
                  "destination, so an in-flight request requires inlined, "
                  "effect-annotated code";
      for (Value operand : op.getOperands()) {
        if (!isa<MemRefType>(operand.getType()))
          continue;
        // Conservative: unknown effects are taken as a full read and write of
        // every declared buffer.
        if (failed(checkValueAccess(op, operand, /*isRead=*/true, analysis)))
          return failure();
        if (failed(checkValueAccess(op, operand, /*isRead=*/false, analysis)))
          return failure();
      }
      return success();
    }

    for (const MemoryEffects::EffectInstance &effect : effects) {
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      Value value = effect.getValue();
      if (!value) {
        // A resource-wide effect has no range to compare, so it is only
        // acceptable while nothing is in flight.
        if (!hasActive(analysis))
          continue;
        return op.emitOpError()
               << "declares a " << (isRead ? "read" : "write")
               << " effect on an entire memory resource while a DMA request is "
                  "in flight; a resource-wide effect cannot be proven disjoint "
                  "from the staged source or the destination";
      }
      if (!isa<MemRefType>(value.getType()))
        continue;
      if (failed(checkValueAccess(op, value, isRead, analysis)))
        return failure();
    }
    return success();
  }

  LogicalResult checkValueAccess(Operation &op, Value value, bool isRead,
                                 BlockAnalysis &analysis) {
    std::optional<Footprint> footprint = resolveAccessFootprint(value);

    for (std::optional<ActiveRequest> &slot : analysis.active) {
      if (!slot)
        continue;
      if (!footprint)
        return op.emitOpError()
               << (isRead ? "reads " : "writes ") << describeOperand(value)
               << ", whose byte range cannot be traced back to a storage root, "
                  "while the request started on DMA channel "
               << slot->channel
               << " is in flight; the access cannot be proven disjoint from "
                  "the staged source or the destination";
      bool hitsSrc = mayAlias(*footprint, slot->src, abiDisjointDdrRoots);
      bool hitsDst = mayAlias(*footprint, slot->dst, abiDisjointDdrRoots);
      if (isRead ? !hitsDst : !(hitsSrc || hitsDst))
        continue;
      StringRef endpoints;
      if (hitsSrc && hitsDst)
        endpoints = "source and the destination";
      else if (hitsSrc)
        endpoints = "source";
      else
        endpoints = "destination";
      return op.emitOpError()
             << (isRead ? "reads " : "writes ") << describeOperand(value)
             << " (" << describeFootprint(*footprint)
             << "), which overlaps the " << endpoints
             << " of the in-flight request on DMA channel " << slot->channel
             << "; the destination is unavailable until its "
                "`bosc_ame.dma.wait` and the source may only be read";
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct SpmDmaCheckPass
    : public PassWrapper<SpmDmaCheckPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpmDmaCheckPass)

  StringRef getArgument() const final { return "bosc-ame-spm-dma-check"; }
  StringRef getDescription() const final {
    return "Verify SPM/DMA request lifetime, channel occupancy and aliasing.";
  }

  SpmDmaCheckPass() = default;
  SpmDmaCheckPass(const SpmDmaCheckPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect>();
  }

  void runOnOperation() override {
    if (failed(buddy::boscame::verifySpmDmaContract(getOperation())))
      signalPassFailure();
  }
};
} // namespace

LogicalResult buddy::boscame::verifySpmDmaContract(ModuleOp module) {
  bool abiDisjointDdrRoots = false;
  if (auto abi = module->getAttrOfType<StringAttr>(kDmaAbiAttrName))
    abiDisjointDdrRoots = abi.getValue() == kDmaAbiAlignedNonAliasing;
  SpmDmaChecker checker(abiDisjointDdrRoots);
  return checker.check(module);
}

std::unique_ptr<Pass> buddy::boscame::createSpmDmaCheckPass() {
  return std::make_unique<SpmDmaCheckPass>();
}

namespace mlir {
namespace buddy {
void registerBOSCAMESpmDmaCheckPass() { PassRegistration<SpmDmaCheckPass>(); }
} // namespace buddy
} // namespace mlir
