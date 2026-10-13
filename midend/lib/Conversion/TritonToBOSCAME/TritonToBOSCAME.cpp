//===- TritonToBOSCAME.cpp - TTIR matmul -> SPM/DMA/AME lowering ----------===//
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
// The FPGA AME path of a Triton kernel: `--triton-to-boscame` rewrites the
// statically shaped signed `i8 x i8 -> i32` `tt.dot` regions of a module into
// the BOSCAME three-level datapath and leaves everything else to
// `--triton-to-linalg-experimental`.
//
// The pass works in three steps, in this order:
//
//   1. *Analyze the whole region before rewriting anything.*  Every candidate
//      dot is analyzed together with the address chain of both operand loads,
//      their masks, the initial accumulator, the loop that carries it and the
//      epilogue that consumes the result.  A region is only rewritten when all
//      of it could be described: a half-rewritten region would either lose
//      semantics or leave two incompatible paths behind.
//   2. *Plan the SPM workspace.*  A, B and the accumulator
//      get fixed, disjoint windows with static offsets, sizes and row
//      pitches, so every tile load/store and every DMA endpoint has a static
//      address and the bank budget is checked up front.
//   3. *Rewrite the region.*  DDR -> SPM staging (DMA where the address, the
//      length and the mask allow it, masked DDR packing otherwise), the
//      K-blocked `tile.mma` reduction with the accumulator carried by the
//      loop, and SPM -> DDR write-back through the same staging machinery.
//
// Addresses and masks are *proved*, never assumed:
//
//   * the address tensor of a load/store is described as
//     `base + row*rowStride + column*colStride` with static strides, derived
//     with interval arithmetic over the values the expression is built from
//     (program id, loop induction variables, `make_range`, `splat`,
//     `expand_dims`, `broadcast`, `addptr` and integer arithmetic);
//   * a mask is a conjunction of `lhs < rhs` comparisons that has to separate
//     into a per-row and a per-column condition;
//   * a DMA transfer is only issued for a range whose length is statically
//     known, whose both endpoints are provably 64 B aligned and which lies
//     entirely inside the valid region; everything else - a partial row, an
//     unaligned row, a dynamic tail - is copied into aligned DDR scratch under
//     its real predicate, then transferred to SPM. The CPU never accesses SPM.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToBOSCAME/TritonToBOSCAME.h"

#include "BOSCAME/BOSCAMEAttrs.h"
#include "BOSCAME/BOSCAMEDialect.h"
#include "BOSCAME/BOSCAMEOps.h"
#include "BOSCAME/DmaAlignment.h"
#include "BOSCAME/SpmDma.h"
#include "BOSCAME/TileOps.h"
#include "BOSCAME/Transforms/BOSCAMETileLowering.h"
#include "Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace buddy::boscame;
using namespace buddy::boscame::spmdma;

#define GEN_PASS_DECL
#define GEN_PASS_DEF_TRITONTOBOSCAME
#include "Conversion/TritonToBOSCAME/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

constexpr int64_t kTileDimMax = kAmeMaxTileDim;
constexpr int64_t kDmaGranularity = kDmaAlignmentBytes;
constexpr StringLiteral kDdrReadBegin = "bosc_ame_ddr_read_begin";
constexpr StringLiteral kDdrWriteBegin = "bosc_ame_ddr_write_begin";

/// E6 DDR loads after DMA must bypass cached data, and subsequent stores must
/// restore the write policy. These calls stay outside the AME live range.
void emitDdrPolicy(OpBuilder &builder, Location loc, bool write) {
  func::CallOp::create(builder, loc, write ? kDdrWriteBegin : kDdrReadBegin,
                       TypeRange{}, ValueRange{});
}

LogicalResult declareDdrPolicies(ModuleOp module) {
  OpBuilder builder(module.getContext());
  auto type = builder.getFunctionType({}, {});
  for (StringRef name : {kDdrReadBegin, kDdrWriteBegin}) {
    if (Operation *symbol = SymbolTable::lookupSymbolIn(module, name)) {
      auto function = dyn_cast<func::FuncOp>(symbol);
      if (!function || function.getFunctionType() != type ||
          !function.isExternal())
        return symbol->emitError()
               << "reserved BOSCAME DDR policy helper @" << name
               << " must be an external func.func with type () -> ()";
      continue;
    }
    builder.setInsertionPointToEnd(module.getBody());
    auto function = func::FuncOp::create(builder, module.getLoc(), name, type);
    function.setPrivate();
  }
  return success();
}

int64_t alignTo(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

/// Byte width of an element the AME tiles implement.
int64_t elementByteWidth(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() / 8;
  return 0;
}

AffineExpr constantExpr(MLIRContext *context, int64_t value) {
  return getAffineConstantExpr(value, context);
}

/// Remap the dims of an affine expression positionally.  `dimMapper` returns
/// the replacement of input dim `i`; an expression that is not returned is kept
/// unchanged.
AffineExpr
remapDims(AffineExpr expr, unsigned sourceRank, unsigned resultRank,
          unsigned numSymbols, MLIRContext *context,
          function_ref<std::optional<AffineExpr>(unsigned)> dimMapper) {
  if (isa<AffineConstantExpr>(expr))
    return expr;
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    if (std::optional<AffineExpr> replacement = dimMapper(dim.getPosition()))
      return *replacement;
    return expr;
  }
  if (isa<AffineSymbolExpr>(expr))
    return expr;
  auto binary = cast<AffineBinaryOpExpr>(expr);
  AffineExpr lhs = remapDims(binary.getLHS(), sourceRank, resultRank,
                             numSymbols, context, dimMapper);
  AffineExpr rhs = remapDims(binary.getRHS(), sourceRank, resultRank,
                             numSymbols, context, dimMapper);
  return simplifyAffineExpr(getAffineBinaryOpExpr(binary.getKind(), lhs, rhs),
                            resultRank, numSymbols);
}

/// `tt.expand_dims` inserts a unit dimension: every source dim keeps its order,
/// the inserted axis is not part of the expression.
AffineExpr mapExpandDims(AffineExpr expr, unsigned sourceRank, unsigned axis,
                         unsigned numSymbols, MLIRContext *context) {
  unsigned resultRank = sourceRank + 1;
  return remapDims(expr, sourceRank, resultRank, numSymbols, context,
                   [&](unsigned dim) -> std::optional<AffineExpr> {
                     if (dim < axis)
                       return getAffineDimExpr(dim, context);
                     return getAffineDimExpr(dim + 1, context);
                   });
}

/// `tt.broadcast` keeps the rank and expands the dimensions that were 1: a
/// unit dimension only ever takes the value 0, so it becomes a constant.
AffineExpr mapBroadcast(AffineExpr expr, unsigned rank,
                        ArrayRef<int64_t> sourceShape, unsigned numSymbols,
                        MLIRContext *context) {
  return remapDims(expr, rank, rank, numSymbols, context,
                   [&](unsigned dim) -> std::optional<AffineExpr> {
                     if (dim < sourceShape.size() && sourceShape[dim] == 1)
                       return constantExpr(context, 0);
                     return getAffineDimExpr(dim, context);
                   });
}

/// An expression that is affine and *linear* in two dims:
/// `expr = rest + c0*d0 + c1*d1`.
struct LinearForm {
  int64_t c0 = 0;
  int64_t c1 = 0;
  AffineExpr rest;
};

std::optional<LinearForm> decomposeLinear(AffineExpr expr, unsigned numSymbols,
                                          MLIRContext *context) {
  SmallVector<AffineExpr, 2> dims{constantExpr(context, 0),
                                  constantExpr(context, 0)};
  auto at = [&](int64_t d0, int64_t d1) {
    dims[0] = constantExpr(context, d0);
    dims[1] = constantExpr(context, d1);
    return simplifyAffineExpr(expr.replaceDimsAndSymbols(dims, {}), 0,
                              numSymbols);
  };
  AffineExpr base = at(0, 0);
  AffineExpr step0 = simplifyAffineExpr(at(1, 0) - base, 0, numSymbols);
  AffineExpr step1 = simplifyAffineExpr(at(0, 1) - base, 0, numSymbols);
  auto constant0 = dyn_cast<AffineConstantExpr>(step0);
  auto constant1 = dyn_cast<AffineConstantExpr>(step1);
  if (!constant0 || !constant1)
    return std::nullopt;
  // Linearity has to be verified: the two probes above only describe a linear
  // function, and a product of two indices would satisfy them by accident.
  AffineExpr rebuilt = base +
                       getAffineDimExpr(0, context) *
                           constantExpr(context, constant0.getValue()) +
                       getAffineDimExpr(1, context) *
                           constantExpr(context, constant1.getValue());
  AffineExpr difference = simplifyAffineExpr(expr - rebuilt, 2, numSymbols);
  auto remainder = dyn_cast<AffineConstantExpr>(difference);
  if (!remainder || remainder.getValue() != 0)
    return std::nullopt;
  return LinearForm{constant0.getValue(), constant1.getValue(), base};
}

/// The linear part of every symbol-free term: the coefficients and the
/// constant term of an expression that is linear in the dims.
bool isConstantExpr(AffineExpr expr) { return isa<AffineConstantExpr>(expr); }

//===----------------------------------------------------------------------===//
// Interval arithmetic
//===----------------------------------------------------------------------===//

/// A conservative integer range.  A missing bound means "not known", which is
/// always safe: it can only make a proof fail and send the access to the
/// ordinary copy path.
struct IndexRange {
  std::optional<int64_t> lo;
  std::optional<int64_t> hi;

  static IndexRange unknown() { return IndexRange{}; }
  static IndexRange constant(int64_t value) { return IndexRange{value, value}; }

  bool isConstant() const { return lo && hi && *lo == *hi; }
  std::optional<int64_t> constant() const {
    return isConstant() ? lo : std::nullopt;
  }
  /// Provably < 0?
  bool isProvablyNegative() const { return hi && *hi < 0; }
};

IndexRange addRanges(const IndexRange &lhs, const IndexRange &rhs) {
  IndexRange result;
  if (lhs.lo && rhs.lo)
    result.lo = *lhs.lo + *rhs.lo;
  if (lhs.hi && rhs.hi)
    result.hi = *lhs.hi + *rhs.hi;
  return result;
}

IndexRange mulRanges(const IndexRange &lhs, const IndexRange &rhs) {
  if (!lhs.lo || !lhs.hi || !rhs.lo || !rhs.hi)
    return IndexRange::unknown();
  int64_t values[4] = {*lhs.lo * *rhs.lo, *lhs.lo * *rhs.hi, *lhs.hi * *rhs.lo,
                       *lhs.hi * *rhs.hi};
  IndexRange result{values[0], values[0]};
  for (int64_t value : values) {
    result.lo = std::min(*result.lo, value);
    result.hi = std::max(*result.hi, value);
  }
  return result;
}

/// The range of one index value, derived from the operations that define it.
/// Anything the walk does not understand is unbounded.
IndexRange inferValueRange(Value value) {
  if (!value)
    return IndexRange::unknown();

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto forOp =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!forOp || llvm::is_contained(forOp.getRegionIterArgs(), value))
      return IndexRange::unknown();
    std::optional<int64_t> lower = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> upper = getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
      return IndexRange::unknown();
    return IndexRange{*lower, *upper - 1};
  }

  Operation *defining = value.getDefiningOp();
  if (!defining)
    return IndexRange::unknown();

  if (auto constant = dyn_cast<arith::ConstantOp>(defining)) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return IndexRange::constant(integer.getInt());
    return IndexRange::unknown();
  }
  if (auto add = dyn_cast<arith::AddIOp>(defining))
    return addRanges(inferValueRange(add.getLhs()),
                     inferValueRange(add.getRhs()));
  if (auto sub = dyn_cast<arith::SubIOp>(defining)) {
    IndexRange rhs = inferValueRange(sub.getRhs());
    if (rhs.lo)
      rhs.lo = -*rhs.lo;
    if (rhs.hi)
      rhs.hi = -*rhs.hi;
    std::swap(rhs.lo, rhs.hi);
    return addRanges(inferValueRange(sub.getLhs()), rhs);
  }
  if (auto mul = dyn_cast<arith::MulIOp>(defining))
    return mulRanges(inferValueRange(mul.getLhs()),
                     inferValueRange(mul.getRhs()));
  if (auto cast = dyn_cast<arith::IndexCastOp>(defining))
    return inferValueRange(cast.getIn());
  if (auto ext = dyn_cast<arith::ExtSIOp>(defining))
    return inferValueRange(ext.getIn());
  if (auto ext = dyn_cast<arith::ExtUIOp>(defining))
    return inferValueRange(ext.getIn());
  if (auto min = dyn_cast<arith::MinSIOp>(defining)) {
    IndexRange lhs = inferValueRange(min.getLhs());
    IndexRange rhs = inferValueRange(min.getRhs());
    return IndexRange{
        lhs.lo && rhs.lo ? std::optional<int64_t>(std::min(*lhs.lo, *rhs.lo))
                         : std::nullopt,
        lhs.hi && rhs.hi ? std::optional<int64_t>(std::min(*lhs.hi, *rhs.hi))
                         : std::nullopt};
  }
  if (auto max = dyn_cast<arith::MaxSIOp>(defining)) {
    IndexRange lhs = inferValueRange(max.getLhs());
    IndexRange rhs = inferValueRange(max.getRhs());
    return IndexRange{
        lhs.lo && rhs.lo ? std::optional<int64_t>(std::max(*lhs.lo, *rhs.lo))
                         : std::nullopt,
        lhs.hi && rhs.hi ? std::optional<int64_t>(std::max(*lhs.hi, *rhs.hi))
                         : std::nullopt};
  }
  return IndexRange::unknown();
}

/// Interval arithmetic over an affine expression.  The dims are given by
/// `dimRanges`, the symbol positions refer to `symbols`.
IndexRange inferAffineRange(AffineExpr expr, ArrayRef<IndexRange> dimRanges,
                            ArrayRef<Value> symbols) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expr))
    return IndexRange::constant(constant.getValue());
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    if (symbol.getPosition() >= symbols.size())
      return IndexRange::unknown();
    return inferValueRange(symbols[symbol.getPosition()]);
  }
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    if (dim.getPosition() >= dimRanges.size())
      return IndexRange::unknown();
    return dimRanges[dim.getPosition()];
  }
  auto binary = cast<AffineBinaryOpExpr>(expr);
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    return addRanges(inferAffineRange(binary.getLHS(), dimRanges, symbols),
                     inferAffineRange(binary.getRHS(), dimRanges, symbols));
  case AffineExprKind::Mul:
    return mulRanges(inferAffineRange(binary.getLHS(), dimRanges, symbols),
                     inferAffineRange(binary.getRHS(), dimRanges, symbols));
  default:
    return IndexRange::unknown();
  }
}

//===----------------------------------------------------------------------===//
// Symbolic evaluation of Triton address and mask expressions
//===----------------------------------------------------------------------===//

/// A tensor of integers summarized as an affine function of its own indices.
struct SymbolicTensor {
  int64_t rank = 0;
  AffineExpr expr;
};

/// A tensor of pointers: a root pointer plus an element offset.
struct SymbolicAddress {
  Value base;
  Type elementType;
  int64_t rank = 0;
  AffineExpr offset;
};

/// One `lhs < rhs` term of a regular mask.
struct MaskComparison {
  AffineExpr diff;
};

/// The integer value of a scalar constant or of a splat constant tensor, when
/// `value` is one.
std::optional<int64_t> getConstantValue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  Attribute attribute = constant.getValue();
  if (auto integer = dyn_cast<IntegerAttr>(attribute))
    return integer.getInt();
  if (auto dense = dyn_cast<DenseElementsAttr>(attribute)) {
    if (!dense.isSplat())
      return std::nullopt;
    if (auto integer = dyn_cast<IntegerAttr>(dense.getSplatValue<Attribute>()))
      return integer.getInt();
  }
  return std::nullopt;
}

/// Symbols shared by every expression of one function: the SSA values that are
/// not one of the tensor's own indices.
class SymbolTable {
public:
  unsigned add(Value value) {
    auto it = indexes.find(value);
    if (it != indexes.end())
      return it->second;
    unsigned position = symbols.size();
    symbols.push_back(value);
    indexes[value] = position;
    return position;
  }

  AffineExpr symbol(Value value, MLIRContext *context) {
    return getAffineSymbolExpr(add(value), context);
  }

  ArrayRef<Value> getSymbols() const { return symbols; }

private:
  SmallVector<Value> symbols;
  llvm::DenseMap<Value, unsigned> indexes;
};

/// Evaluate the integer, pointer and mask expressions Triton builds from
/// program ids, ranges, splats, broadcasts and integer arithmetic.
///
/// Every operation the evaluator does not understand produces a failure with a
/// reason, which the pass reports instead of rewriting a region it cannot
/// describe.
class SymbolicEvaluator {
public:
  explicit SymbolicEvaluator(MLIRContext *context) : context(context) {}

  ArrayRef<Value> getSymbols() const { return table.getSymbols(); }
  StringRef getReason() const { return reason; }

  FailureOr<SymbolicTensor> evaluateInteger(Value value) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || !tensor.getElementType().isIntOrIndex())
      return fail(value, "expected an integer tensor");
    Operation *defining = value.getDefiningOp();
    if (!defining)
      return fail(value, "the value is a tensor without a producer inside the "
                         "region");

    if (auto splat = dyn_cast<SplatOp>(defining)) {
      if (std::optional<int64_t> value = getConstantValue(splat.getSrc()))
        return SymbolicTensor{tensor.getRank(), constantExpr(context, *value)};
      return SymbolicTensor{tensor.getRank(),
                            table.symbol(splat.getSrc(), context)};
    }
    if (auto range = dyn_cast<MakeRangeOp>(defining)) {
      if (tensor.getRank() != 1)
        return fail(value, "tt.make_range must produce a 1-D tensor");
      return SymbolicTensor{
          1, constantExpr(context, range.getStartAttr().getInt()) +
                 getAffineDimExpr(0, context)};
    }
    if (auto expand = dyn_cast<ExpandDimsOp>(defining)) {
      FailureOr<SymbolicTensor> source = evaluateInteger(expand.getSrc());
      if (failed(source))
        return failure();
      return SymbolicTensor{tensor.getRank(),
                            mapExpandDims(source->expr, source->rank,
                                          expand.getAxis(),
                                          table.getSymbols().size(), context)};
    }
    if (auto broadcast = dyn_cast<BroadcastOp>(defining)) {
      FailureOr<SymbolicTensor> source = evaluateInteger(broadcast.getSrc());
      if (failed(source))
        return failure();
      auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
      return SymbolicTensor{tensor.getRank(),
                            mapBroadcast(source->expr, source->rank,
                                         sourceType.getShape(),
                                         table.getSymbols().size(), context)};
    }
    if (auto add = dyn_cast<arith::AddIOp>(defining))
      return combine(add.getLhs(), add.getRhs(), value, /*negateRhs=*/false);
    if (auto sub = dyn_cast<arith::SubIOp>(defining))
      return combine(sub.getLhs(), sub.getRhs(), value, /*negateRhs=*/true);
    if (auto mul = dyn_cast<arith::MulIOp>(defining)) {
      FailureOr<SymbolicTensor> lhs = evaluateInteger(mul.getLhs());
      if (failed(lhs))
        return failure();
      FailureOr<SymbolicTensor> rhs = evaluateInteger(mul.getRhs());
      if (failed(rhs))
        return failure();
      if (!isConstantExpr(lhs->expr) && !isConstantExpr(rhs->expr))
        return fail(value,
                    "the product is not affine in the tensor indices; only "
                    "multiplication by a constant is a regular address");
      return SymbolicTensor{
          lhs->rank,
          getAffineBinaryOpExpr(AffineExprKind::Mul, lhs->expr, rhs->expr)};
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(defining)) {
      if (std::optional<int64_t> scalar =
              getConstantValue(constant.getResult()))
        return SymbolicTensor{tensor.getRank(), constantExpr(context, *scalar)};
      return fail(value, "the constant is not an integer splat; a per-element "
                         "constant is not a regular address expression");
    }
    if (auto ext = dyn_cast<arith::ExtSIOp>(defining))
      return evaluateInteger(ext.getIn());
    if (auto ext = dyn_cast<arith::ExtUIOp>(defining))
      return evaluateInteger(ext.getIn());
    if (auto cast = dyn_cast<arith::IndexCastOp>(defining))
      return evaluateInteger(cast.getIn());

    return fail(value,
                "the operand is produced by an operation this pass has no "
                "address model for");
  }

  FailureOr<SymbolicAddress> evaluateAddress(Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !isa<PointerType>(type.getElementType()))
      return fail(value, "expected a tensor of pointers");
    auto pointerType = cast<PointerType>(type.getElementType());
    Operation *defining = value.getDefiningOp();
    if (!defining)
      return fail(value, "the pointer tensor has no producer");

    if (auto splat = dyn_cast<SplatOp>(defining)) {
      Value scalar = splat.getSrc();
      if (!isa<PointerType>(scalar.getType()))
        return fail(value, "tt.splat of a non-pointer scalar into a tensor of "
                           "pointers");
      return SymbolicAddress{scalar, pointerType.getPointeeType(),
                             type.getRank(), constantExpr(context, 0)};
    }
    if (auto addPtr = dyn_cast<AddPtrOp>(defining)) {
      FailureOr<SymbolicAddress> address = evaluateAddress(addPtr.getPtr());
      if (failed(address))
        return failure();
      FailureOr<SymbolicTensor> offset = evaluateInteger(addPtr.getOffset());
      if (failed(offset))
        return failure();
      if (address->rank != offset->rank)
        return fail(value, "tt.addptr combines a pointer tensor and an offset "
                           "tensor of different rank");
      address->offset = getAffineBinaryOpExpr(AffineExprKind::Add,
                                              address->offset, offset->expr);
      return address;
    }
    if (auto expand = dyn_cast<ExpandDimsOp>(defining)) {
      FailureOr<SymbolicAddress> address = evaluateAddress(expand.getSrc());
      if (failed(address))
        return failure();
      address->offset =
          mapExpandDims(address->offset, address->rank, expand.getAxis(),
                        table.getSymbols().size(), context);
      address->rank = type.getRank();
      return address;
    }
    if (auto broadcast = dyn_cast<BroadcastOp>(defining)) {
      FailureOr<SymbolicAddress> address = evaluateAddress(broadcast.getSrc());
      if (failed(address))
        return failure();
      auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
      address->offset =
          mapBroadcast(address->offset, address->rank, sourceType.getShape(),
                       table.getSymbols().size(), context);
      address->rank = type.getRank();
      return address;
    }
    return fail(value,
                "the pointer tensor is produced by an operation this pass has "
                "no address model for");
  }

  FailureOr<SmallVector<MaskComparison>> evaluateMask(Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.getElementType().isInteger(1))
      return fail(value, "expected an i1 tensor mask");
    SmallVector<MaskComparison> comparisons;
    if (failed(collectMask(value, comparisons)))
      return failure();
    return comparisons;
  }

private:
  FailureOr<SymbolicTensor> combine(Value lhs, Value rhs, Value result,
                                    bool negateRhs) {
    FailureOr<SymbolicTensor> lhsTensor = evaluateInteger(lhs);
    if (failed(lhsTensor))
      return failure();
    FailureOr<SymbolicTensor> rhsTensor = evaluateInteger(rhs);
    if (failed(rhsTensor))
      return failure();
    if (lhsTensor->rank != rhsTensor->rank)
      return fail(result, "the operands of the integer operation have "
                          "different ranks");
    AffineExpr rhsExpr = rhsTensor->expr;
    if (negateRhs)
      rhsExpr = getAffineBinaryOpExpr(AffineExprKind::Mul, rhsExpr,
                                      constantExpr(context, -1));
    return SymbolicTensor{
        lhsTensor->rank,
        getAffineBinaryOpExpr(AffineExprKind::Add, lhsTensor->expr, rhsExpr)};
  }

  LogicalResult collectMask(Value value, SmallVectorImpl<MaskComparison> &out) {
    Operation *defining = value.getDefiningOp();
    if (!defining)
      return fail(value, "the mask has no producer");
    if (auto cmpi = dyn_cast<arith::CmpIOp>(defining)) {
      if (cmpi.getPredicate() != arith::CmpIPredicate::slt &&
          cmpi.getPredicate() != arith::CmpIPredicate::ult)
        return fail(value,
                    "the mask comparison is not a '<' comparison; only regular "
                    "upper-bound masks are supported");
      FailureOr<SymbolicTensor> lhs = evaluateInteger(cmpi.getLhs());
      if (failed(lhs))
        return failure();
      FailureOr<SymbolicTensor> rhs = evaluateInteger(cmpi.getRhs());
      if (failed(rhs))
        return failure();
      out.push_back(MaskComparison{getAffineBinaryOpExpr(
          AffineExprKind::Add, lhs->expr,
          getAffineBinaryOpExpr(AffineExprKind::Mul, rhs->expr,
                                constantExpr(context, -1)))});
      return success();
    }
    if (auto andi = dyn_cast<arith::AndIOp>(defining)) {
      if (failed(collectMask(andi.getLhs(), out)))
        return failure();
      return collectMask(andi.getRhs(), out);
    }
    if (auto expand = dyn_cast<ExpandDimsOp>(defining)) {
      SmallVector<MaskComparison> inner;
      if (failed(collectMask(expand.getSrc(), inner)))
        return failure();
      auto sourceType = cast<RankedTensorType>(expand.getSrc().getType());
      return remapComparisons(inner, sourceType.getRank(),
                              /*axis=*/expand.getAxis(),
                              /*broadcast=*/false, {}, out);
    }
    if (auto broadcast = dyn_cast<BroadcastOp>(defining)) {
      SmallVector<MaskComparison> inner;
      if (failed(collectMask(broadcast.getSrc(), inner)))
        return failure();
      auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
      return remapComparisons(inner, sourceType.getRank(), /*axis=*/0,
                              /*broadcast=*/true, sourceType.getShape(), out);
    }
    return fail(value,
                "the mask is produced by an operation this pass cannot turn "
                "into regular row/column conditions");
  }

  LogicalResult remapComparisons(ArrayRef<MaskComparison> inner,
                                 unsigned sourceRank, unsigned axis,
                                 bool broadcast, ArrayRef<int64_t> sourceShape,
                                 SmallVectorImpl<MaskComparison> &out) {
    for (const MaskComparison &comparison : inner) {
      AffineExpr remapped =
          broadcast ? mapBroadcast(comparison.diff, sourceRank, sourceShape,
                                   table.getSymbols().size(), context)
                    : mapExpandDims(comparison.diff, sourceRank, axis,
                                    table.getSymbols().size(), context);
      out.push_back(MaskComparison{remapped});
    }
    return success();
  }

  LogicalResult fail(Value value, StringRef message) {
    if (reason.empty()) {
      std::string text;
      llvm::raw_string_ostream stream(text);
      stream << message << ": " << value;
      reason = stream.str();
    }
    return failure();
  }

  MLIRContext *context;
  SymbolTable table;
  std::string reason;
};

//===----------------------------------------------------------------------===//
// Access description
//===----------------------------------------------------------------------===//

/// `base + row*rowStride + column*colStride` in elements, with static strides.
struct AddressSpec {
  Value base;
  Type elementType;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t rowStride = 0;
  int64_t colStride = 0;
  AffineExpr origin;
};

/// One tile access (`tt.load` / `tt.store`) of a target region.
struct AccessSpec {
  AddressSpec address;
  SmallVector<MaskComparison> mask;
  Value other;
  bool dense() const { return mask.empty(); }
};

FailureOr<AddressSpec> analyzeAddress(SymbolicEvaluator &evaluator,
                                      Value pointerTensor,
                                      std::string &reason) {
  FailureOr<SymbolicAddress> address = evaluator.evaluateAddress(pointerTensor);
  if (failed(address)) {
    reason = evaluator.getReason().str();
    return failure();
  }
  auto tensorType = cast<RankedTensorType>(pointerTensor.getType());
  if (address->rank != 2) {
    reason = "the access is not a 2-D tile; only 2-D tiles are supported";
    return failure();
  }
  std::optional<LinearForm> form =
      decomposeLinear(address->offset, evaluator.getSymbols().size(),
                      pointerTensor.getContext());
  if (!form) {
    reason = "the element offset of the access is not a linear function of the "
             "two tile indices (row and column); only regular addresses with "
             "static strides are supported";
    return failure();
  }
  AddressSpec spec;
  spec.base = address->base;
  spec.elementType = address->elementType;
  spec.rows = tensorType.getShape()[0];
  spec.cols = tensorType.getShape()[1];
  spec.rowStride = form->c0;
  spec.colStride = form->c1;
  spec.origin = form->rest;
  return spec;
}

//===----------------------------------------------------------------------===//
// Region description
//===----------------------------------------------------------------------===//

/// One `tt.dot` and everything the rewrite has to keep intact around it.
struct MatmulRegion {
  DotOp dot;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  /// The reduction extent rounded up to the AME tile width; the extra lanes
  /// are zeroed so that they cannot contribute to the result.
  int64_t kPadded = 0;

  LoadOp aLoad;
  LoadOp bLoad;
  AccessSpec aAccess;
  AccessSpec bAccess;

  /// The loop that carries the accumulator, when the dot is inside one.
  scf::ForOp loop;
  unsigned accArgument = 0;
  Value accInit;
  std::optional<int64_t> accInitValue;

  LoadOp previousLoad;
  AccessSpec previousAccess;
  arith::AddIOp addi;
  bool hasPrevious = false;
  // The accumulator tile has to leave SPM through a transfer: the E6 CPU
  // cannot complete a load from the SPM window, so the DDR-side epilogue
  // combines the tile with `previous` and writes it back through this
  // caller-provided scratch window instead of reading SPM.
  Value scratch;
  Value scratchBytes;
  Value scratchTyped;
  int64_t scratchBytesCount = 0;
  StoreOp store;
  AccessSpec storeAccess;

  /// Staging plan. The K blocks of *every* enclosing-loop iteration are
  /// staged into SPM before the accumulator is loaded, so that the
  /// accumulator stays in the AME register file across a body that contains
  /// no call at all: the DMA runtime helpers are calls, and an AME register
  /// value that is live across a call cannot be preserved (the AME register
  /// files have no spill support, and a spill would also have to save the
  /// matrix engine configuration).
  ///
  /// `stagedIterations` is the number of staged K blocks, `stagedK` the
  /// reduction extent the A/B windows have to hold per row, and
  /// `loopIvSymbol` the position of the enclosing loop's induction variable
  /// in `symbols` when the staged addresses depend on it.
  int64_t stagedIterations = 1;
  int64_t stagedK = 0;
  std::optional<int64_t> loopLowerBound;
  std::optional<int64_t> loopUpperBound;
  std::optional<int64_t> loopStep;
  std::optional<unsigned> loopIvSymbol;

  SmallVector<Value> symbols;
};

/// The scalar of a splat tensor, when the tensor is one.
std::optional<int64_t> getSplatValue(Value tensor) {
  if (auto constant = tensor.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (dense && dense.isSplat())
      if (auto integer =
              dyn_cast<IntegerAttr>(dense.getSplatValue<Attribute>()))
        return integer.getInt();
  }
  return std::nullopt;
}

FailureOr<MatmulRegion> analyzeRegion(DotOp dot, std::string &reason) {
  MLIRContext *context = dot.getContext();
  MatmulRegion region;
  region.dot = dot;

  auto resultType = dyn_cast<RankedTensorType>(dot.getType());
  auto lhsType = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto rhsType = dyn_cast<RankedTensorType>(dot.getB().getType());
  if (!resultType || !lhsType || !rhsType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || resultType.getRank() != 2) {
    reason = "the dot operands and result must be statically shaped 2-D "
             "tensors";
    return failure();
  }
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !resultType.hasStaticShape()) {
    reason = "a target dot must have a static tile shape";
    return failure();
  }
  region.m = lhsType.getShape()[0];
  region.k = lhsType.getShape()[1];
  region.n = rhsType.getShape()[1];
  if (rhsType.getShape()[0] != region.k ||
      resultType.getShape()[0] != region.m ||
      resultType.getShape()[1] != region.n) {
    reason = "the dot operand shapes do not agree: A is [" +
             std::to_string(lhsType.getShape()[0]) + ", " +
             std::to_string(lhsType.getShape()[1]) + "], B is [" +
             std::to_string(rhsType.getShape()[0]) + ", " +
             std::to_string(rhsType.getShape()[1]) +
             "] and the accumulator is [" +
             std::to_string(resultType.getShape()[0]) + ", " +
             std::to_string(resultType.getShape()[1]) + "]";
    return failure();
  }
  if (region.m < 1 || region.n < 1 || region.k < 1) {
    reason = "a target dot must have positive M, N and K";
    return failure();
  }
  if (region.m > kTileDimMax || region.n > kTileDimMax) {
    reason = "the AME accumulator tile is at most " +
             std::to_string(kTileDimMax) + "x" + std::to_string(kTileDimMax) +
             ", but this dot has M = " + std::to_string(region.m) +
             " and N = " + std::to_string(region.n) +
             "; split the dot into tiles that fit the register file";
    return failure();
  }
  // The tile registers are the full tile width in K, so a narrower reduction
  // is padded with zero lanes; the padding is a property of the staging, not
  // of the mathematical reduction.
  region.kPadded = alignTo(region.k, kTileDimMax);

  region.aLoad = dot.getA().getDefiningOp<LoadOp>();
  region.bLoad = dot.getB().getDefiningOp<LoadOp>();
  if (!region.aLoad || !region.bLoad) {
    reason = "the A and B operands of a target dot must come directly from a "
             "tt.load; an operand computed from a dot result or a block "
             "argument has no DDR description this pass can stage";
    return failure();
  }

  SymbolicEvaluator evaluator(context);
  FailureOr<AddressSpec> aAddress =
      analyzeAddress(evaluator, region.aLoad.getPtr(), reason);
  if (failed(aAddress))
    return failure();
  FailureOr<AddressSpec> bAddress =
      analyzeAddress(evaluator, region.bLoad.getPtr(), reason);
  if (failed(bAddress))
    return failure();
  region.aAccess.address = *aAddress;
  region.bAccess.address = *bAddress;

  // The AME A load wants K contiguous, the B load wants the physical N x K
  // buffer with K contiguous; DMA never transposes, so a DDR layout that is
  // not already in that order is rejected instead of being silently loaded
  // transposed.
  if (region.aAccess.address.colStride != 1) {
    reason = "A must have the reduction axis contiguous in DDR (element stride "
             "1 along K), but its K stride is " +
             std::to_string(region.aAccess.address.colStride);
    return failure();
  }
  if (region.bAccess.address.rowStride != 1) {
    reason = "B must be stored as the physical N x K tile the AME B load "
             "reads (K contiguous, element stride 1), but its K stride is " +
             std::to_string(region.bAccess.address.rowStride) +
             "; transpose B before staging it, DMA cannot";
    return failure();
  }

  // Like `other`, an absent mask is passed as a plain (null) `Value`: a null
  // Value inside `std::optional` would be engaged, and the evaluator would be
  // asked to evaluate it.
  auto collectMask = [&](Operation *op, Value mask,
                         AccessSpec &access) -> LogicalResult {
    (void)op;
    if (!mask)
      return success();
    FailureOr<SmallVector<MaskComparison>> comparisons =
        evaluator.evaluateMask(mask);
    if (failed(comparisons)) {
      reason = evaluator.getReason().str();
      return failure();
    }
    access.mask = *comparisons;
    return success();
  };
  if (failed(collectMask(region.aLoad.getOperation(), region.aLoad.getMask(),
                         region.aAccess)) ||
      failed(collectMask(region.bLoad.getOperation(), region.bLoad.getMask(),
                         region.bAccess)))
    return failure();

  // `other` is absent for a dense load. It is passed as a plain `Value` on
  // purpose: a null Value wrapped in `std::optional` would be "engaged", and
  // the splat check below would dereference it.
  auto collectOther = [&](Operation *op, Value other, AccessSpec &access,
                          StringRef what) -> LogicalResult {
    if (!other)
      return success();
    std::optional<int64_t> value = getSplatValue(other);
    if (!value) {
      reason = std::string(what) +
               ": the `other` value is not a splat constant; only a scalar or "
               "splat `other` is representable in the staging buffer";
      return failure();
    }
    access.other = other;
    (void)op;
    return success();
  };
  if (failed(collectOther(region.aLoad.getOperation(), region.aLoad.getOther(),
                          region.aAccess, "the A load")) ||
      failed(collectOther(region.bLoad.getOperation(), region.bLoad.getOther(),
                          region.bAccess, "the B load")))
    return failure();

  // The accumulator chain: the dot's `c` operand is either the initial value
  // itself or the loop-carried value of an enclosing `scf.for`.
  Value accumulator = dot.getC();
  if (auto argument = dyn_cast<BlockArgument>(accumulator)) {
    region.loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!region.loop ||
        !llvm::is_contained(region.loop.getRegionIterArgs(), accumulator)) {
      reason = "the accumulator operand of the dot is a block argument that is "
               "not an `scf.for` iter-argument";
      return failure();
    }
    region.accArgument = argument.getArgNumber() - 1;
    region.accInit = region.loop.getInitArgs()[region.accArgument];
  } else {
    region.accInit = accumulator;
  }
  if (auto argument = dyn_cast<BlockArgument>(region.accInit)) {
    reason = "the initial accumulator is a block argument; only a splat "
             "constant initial accumulator is supported";
    return failure();
  }
  region.accInitValue = getSplatValue(region.accInit);
  if (!region.accInitValue) {
    reason = "the initial accumulator is not a splat constant; only a splat "
             "initial value is supported";
    return failure();
  }

  // What consumes the dot result: either the store directly, or an
  // `arith.addi` of the dot result with a `previous` load.
  Value result =
      region.loop ? region.loop.getResult(region.accArgument) : dot.getResult();
  Value stored = result;
  for (Operation *user : result.getUsers()) {
    if (auto store = dyn_cast<StoreOp>(user)) {
      region.store = store;
      continue;
    }
    if (auto add = dyn_cast<arith::AddIOp>(user)) {
      Value other = add.getLhs() == result ? add.getRhs() : add.getLhs();
      auto previous = other.getDefiningOp<LoadOp>();
      if (!previous) {
        reason = "the dot result is added to a value that is not a tt.load; "
                 "only the `previous + A x B` epilogue is supported";
        return failure();
      }
      for (Operation *addUser : add.getResult().getUsers()) {
        auto store = dyn_cast<StoreOp>(addUser);
        if (!store) {
          reason = "the value computed from the dot result is used by an "
                   "operation other than a tt.store; only a direct store of "
                   "`previous + A x B` is supported";
          return failure();
        }
        region.store = store;
      }
      region.previousLoad = previous;
      region.addi = add;
      region.hasPrevious = true;
      stored = add.getResult();
      continue;
    }
    reason = "the dot result is consumed by an operation this pass does not "
             "lower; a target dot result must be stored directly or be added "
             "to a `previous` load and then stored";
    return failure();
  }
  if (!region.store) {
    reason = "the dot result is never stored; there is no write-back for the "
             "SPM staging to target";
    return failure();
  }
  (void)stored;
  if (region.hasPrevious) {
    if (region.previousLoad.getPtr() != region.store.getPtr()) {
      reason = "the `previous` load and the write-back store use different "
               "addresses; only the `previous + A x B` epilogue over the same "
               "address is supported";
      return failure();
    }
    if (region.previousLoad.getMask() != region.store.getMask()) {
      reason = "the `previous` load and the write-back store use different "
               "masks; only matching masks are supported";
      return failure();
    }
  }

  FailureOr<AddressSpec> previousAddress;
  if (region.hasPrevious) {
    previousAddress =
        analyzeAddress(evaluator, region.previousLoad.getPtr(), reason);
    if (failed(previousAddress))
      return failure();
    region.previousAccess.address = *previousAddress;
    if (failed(collectMask(region.previousLoad.getOperation(),
                           region.previousLoad.getMask(),
                           region.previousAccess)))
      return failure();
    if (failed(collectOther(region.previousLoad.getOperation(),
                            region.previousLoad.getOther(),
                            region.previousAccess, "the previous load")))
      return failure();
    if (region.previousAccess.dense() == false &&
        !region.previousLoad.getOther()) {
      reason = "the `previous` load has a mask but no `other` value; the "
               "masked-out lanes are undefined in TTIR and the SPM staging "
               "would have to invent a value";
      return failure();
    }
  }

  FailureOr<AddressSpec> storeAddress =
      analyzeAddress(evaluator, region.store.getPtr(), reason);
  if (failed(storeAddress))
    return failure();
  region.storeAccess.address = *storeAddress;
  if (failed(collectMask(region.store.getOperation(), region.store.getMask(),
                         region.storeAccess)))
    return failure();
  if (region.storeAccess.address.colStride != 1) {
    reason = "the write-back address must have N contiguous in DDR (element "
             "stride 1 along N)";
    return failure();
  }
  if (region.hasPrevious && region.previousAccess.address.colStride != 1) {
    reason = "the `previous` address must have N contiguous in DDR (element "
             "stride 1 along N)";
    return failure();
  }

  region.symbols.assign(evaluator.getSymbols().begin(),
                        evaluator.getSymbols().end());

  // Plan the staging of the whole accumulation up front. Every K block the
  // enclosing loop walks over is staged before the accumulator is loaded, so
  // the loop body contains only AME operations. That needs the loop to be
  // enumerable at compile time and the staged blocks to fit in SPM.
  region.stagedK = region.kPadded;
  if (region.loop) {
    std::optional<int64_t> lower =
        getConstantIntValue(region.loop.getLowerBound());
    std::optional<int64_t> upper =
        getConstantIntValue(region.loop.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(region.loop.getStep());
    if (!lower || !upper || !step || *step <= 0 ||
        (*upper - *lower) % *step != 0) {
      reason = "the loop that carries the accumulator must have constant "
               "bounds and step so that its K blocks can be staged before the "
               "accumulator is loaded; the AME register files cannot hold an "
               "accumulator across a call";
      return failure();
    }
    region.loopLowerBound = *lower;
    region.loopUpperBound = *upper;
    region.loopStep = *step;
    region.stagedIterations = (*upper - *lower) / *step;
    if (region.stagedIterations <= 0) {
      reason = "the loop that carries the accumulator has no iteration, but "
               "the dot still has to produce the initial accumulator";
      return failure();
    }
    region.stagedK = region.stagedIterations * region.kPadded;
    for (auto [position, symbol] : llvm::enumerate(region.symbols))
      if (symbol == region.loop.getInductionVar())
        region.loopIvSymbol = position;
  }
  return region;
}

//===----------------------------------------------------------------------===//
// SPM workspace
//===----------------------------------------------------------------------===//

/// One fixed window of an SPM bank.
struct SpmWindow {
  int64_t bank = 0;
  int64_t offset = 0;
  int64_t sizeBytes = 0;
};

/// The fixed workspace layout of one function.
///
/// The layout is deterministic: windows are handed out front to back inside a
/// bank, 64 B aligned, and a request that does not fit is an error.  Two
/// concurrently running programs of the same kernel would share these
/// addresses, so a caller that launches the kernel concurrently has to provide
/// disjoint banks or serialize the programs; the pass documents the layout it
/// produced instead of pretending the hardware has a per-program window.
class SpmWorkspace {
public:
  SpmWorkspace() { nextOffset.assign(kSpmBankCount, 0); }

  FailureOr<SpmWindow> allocate(int64_t bank, int64_t sizeBytes, StringRef what,
                                std::string &reason) {
    if (bank < 0 || bank >= kSpmBankCount) {
      reason = "unknown SPM bank " + std::to_string(bank);
      return failure();
    }
    int64_t offset = alignTo(nextOffset[bank], kDmaGranularity);
    if (offset + sizeBytes > kSpmBankCapacityBytes) {
      reason = "the " + std::string(what) + " window of " +
               std::to_string(sizeBytes) + " bytes does not fit in SPM bank " +
               std::to_string(bank) + ": " + std::to_string(offset) +
               " bytes are already allocated and the bank holds " +
               std::to_string(kSpmBankCapacityBytes) +
               " bytes; reduce the tile or split the region";
      return failure();
    }
    nextOffset[bank] = offset + sizeBytes;
    return SpmWindow{bank, offset, sizeBytes};
  }

  /// The `bosc_ame.spm.view` of one bank, created once per bank and function.
  Value getBankView(OpBuilder &builder, Location loc, int64_t bank,
                    ModuleOp module) {
    if (bankViews[bank])
      return bankViews[bank];
    auto type = MemRefType::get({kSpmBankCapacityBytes}, builder.getI8Type(),
                                MemRefLayoutAttrInterface(),
                                SPMAttr::get(builder.getContext(), bank));
    auto view =
        SpmViewOp::create(builder, loc, type, builder.getI32IntegerAttr(bank),
                          builder.getI64IntegerAttr(0));
    bankViews[bank] = view.getView();
    (void)module;
    return bankViews[bank];
  }

  int64_t used(int64_t bank) const { return nextOffset[bank]; }

private:
  SmallVector<int64_t> nextOffset;
  Value bankViews[kSpmBankCount] = {Value(), Value()};
};

//===----------------------------------------------------------------------===//
// Mask plan
//===----------------------------------------------------------------------===//

/// The mask of one staged tile, split into the staging row axis and the fast
/// (contiguous) axis.
///
/// A row is valid when `coeff * row + rest < 0` holds for every row term, and
/// the valid positions of a valid row are `[0, count)` where `count` is the
/// minimum of the column bounds clamped to the row width.  This is the shape
/// of every mask the reference kernels use - a rectangle clipped by a row
/// limit and a reduction limit - and anything else is rejected instead of
/// being approximated.
struct MaskPlan {
  struct RowTerm {
    int64_t coeff;
    AffineExpr rest;
  };
  SmallVector<RowTerm> rowTerms;
  SmallVector<AffineExpr> colBounds;
  bool dense = true;
};

FailureOr<MaskPlan> buildMaskPlan(ArrayRef<MaskComparison> comparisons,
                                  unsigned rowAxis, unsigned colAxis,
                                  unsigned numSymbols, MLIRContext *context,
                                  std::string &reason) {
  MaskPlan plan;
  plan.dense = comparisons.empty();
  for (const MaskComparison &comparison : comparisons) {
    std::optional<LinearForm> form =
        decomposeLinear(comparison.diff, numSymbols, context);
    if (!form) {
      reason = "a mask term is not linear in the two tile indices; only "
               "regular row/column masks are supported";
      return failure();
    }
    int64_t rowCoefficient = rowAxis == 0 ? form->c0 : form->c1;
    int64_t colCoefficient = colAxis == 0 ? form->c0 : form->c1;
    if (rowCoefficient == 0 && colCoefficient == 0) {
      IndexRange range = inferAffineRange(form->rest, {}, {});
      if (!range.isProvablyNegative()) {
        reason = "a mask term does not depend on either tile index and is not "
                 "provably true; the first version only supports regular "
                 "row/column masks";
        return failure();
      }
      continue;
    }
    if (rowCoefficient != 0 && colCoefficient != 0) {
      reason = "a mask term couples the row and the column index; only a "
               "product of a row condition and a column condition is a "
               "regular mask";
      return failure();
    }
    if (rowCoefficient != 0) {
      if (rowCoefficient < 0) {
        reason = "a row condition has a negative coefficient (a lower bound); "
                 "only an upper row bound is supported";
        return failure();
      }
      plan.rowTerms.push_back(MaskPlan::RowTerm{rowCoefficient, form->rest});
      continue;
    }
    if (colCoefficient != 1) {
      reason = "a column condition has a coefficient other than 1; only a "
               "unit-stride upper column bound is supported";
      return failure();
    }
    plan.colBounds.push_back(simplifyAffineExpr(
        getAffineBinaryOpExpr(AffineExprKind::Mul, form->rest,
                              constantExpr(context, -1)),
        0, numSymbols));
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// Region lowering
//===----------------------------------------------------------------------===//

/// The four fixed SPM windows a rewritten region uses.
struct RegionWindows {
  SpmWindow a;
  SpmWindow b;
  SpmWindow acc;
  SpmWindow prev;
  bool hasPrev = false;

  Value bank0View;
  Value bank1View;
  Value aBytes;
  Value bBytes;
  Value accBytes;
  Value prevBytes;
  Value aTyped;
  Value bTyped;
  Value accTyped;
  Value prevTyped;
  int64_t aPitch = 0;
  int64_t bPitch = 0;
  int64_t accPitch = 0;
  int64_t prevPitch = 0;
};

class RegionLowering;
void eraseIfDead(Operation *op);

class RegionLowering {
public:
  RegionLowering(MatmulRegion &region, SpmWorkspace &workspace,
                 bool dmaAbiPresent, SmallVectorImpl<Value> &convertedRoots)
      : region(region), workspace(workspace), context(region.dot.getContext()),
        dmaAbiPresent(dmaAbiPresent), convertedRoots(convertedRoots) {}

  LogicalResult run(std::string &reason);

private:
  //===--- builders ---===//
  Value indexConstant(OpBuilder &builder, Location loc, int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value);
  }

  /// The index form of a symbol.  The cast is created at the use site, never
  /// cached: the same symbol can be used in a loop body and after the loop,
  /// and a value created in the body would not dominate the later use.
  Value toIndex(OpBuilder &builder, Location loc, Value value) {
    if (value.getType().isIndex())
      return value;
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      value);
  }

  /// Materialize an affine expression as index arithmetic.
  Value buildIndexValue(OpBuilder &builder, Location loc, AffineExpr expr) {
    if (auto constant = dyn_cast<AffineConstantExpr>(expr))
      return indexConstant(builder, loc, constant.getValue());
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
      if (symbol.getPosition() >= region.symbols.size())
        return Value();
      return toIndex(builder, loc, region.symbols[symbol.getPosition()]);
    }
    if (isa<AffineDimExpr>(expr))
      return Value();
    auto binary = cast<AffineBinaryOpExpr>(expr);
    switch (binary.getKind()) {
    case AffineExprKind::Add: {
      Value lhs = buildIndexValue(builder, loc, binary.getLHS());
      Value rhs = buildIndexValue(builder, loc, binary.getRHS());
      if (!lhs || !rhs)
        return Value();
      return arith::AddIOp::create(builder, loc, lhs, rhs);
    }
    case AffineExprKind::Mul: {
      auto lhsConstant = dyn_cast<AffineConstantExpr>(binary.getLHS());
      auto rhsConstant = dyn_cast<AffineConstantExpr>(binary.getRHS());
      if (rhsConstant) {
        Value lhs = buildIndexValue(builder, loc, binary.getLHS());
        if (!lhs)
          return Value();
        return arith::MulIOp::create(
            builder, loc, lhs,
            indexConstant(builder, loc, rhsConstant.getValue()));
      }
      if (lhsConstant) {
        Value rhs = buildIndexValue(builder, loc, binary.getRHS());
        if (!rhs)
          return Value();
        return arith::MulIOp::create(
            builder, loc, indexConstant(builder, loc, lhsConstant.getValue()),
            rhs);
      }
      return Value();
    }
    default:
      return Value();
    }
  }

  bool hasIndexModel(const AccessSpec &access) const {
    return elementByteWidth(access.address.elementType) != 0;
  }

  //===--- types ---===//
  MemRefType byteWindowType(int64_t bytes, const SpmWindow &window) {
    return MemRefType::get({bytes}, IntegerType::get(context, 8),
                           MemRefLayoutAttrInterface(),
                           SPMAttr::get(context, window.bank));
  }

  MemRefType typedWindowType(int64_t rows, int64_t cols, Type elementType,
                             int64_t pitchBytes, const SpmWindow &window) {
    int64_t elementBytes = elementByteWidth(elementType);
    SmallVector<int64_t, 2> strides{pitchBytes / elementBytes, 1};
    return MemRefType::get({rows, cols}, elementType,
                           StridedLayoutAttr::get(context, 0, strides),
                           SPMAttr::get(context, window.bank));
  }

  //===--- staging ---===//
  FailureOr<Value> getRoot(Value base, std::string &reason);

  LogicalResult createWindows(OpBuilder &builder, Location loc,
                              std::string &reason);

  /// Stage an operand with direct DMA when the whole tile is valid/aligned,
  /// otherwise assemble it in DDR scratch before DMA. Only DDR is CPU-accessed.
  LogicalResult stageIn(OpBuilder &builder, Location loc,
                        const AccessSpec &access, unsigned rowAxis, Value root,
                        const SpmWindow &window, Value windowBytes,
                        Value windowTyped, int64_t pitchBytes, int64_t rows,
                        int64_t cols, int64_t columnOffset,
                        std::string &reason);

  /// Stage the K block of every enclosing-loop iteration before loading ACC.
  LogicalResult stageAllBlocks(OpBuilder &builder, Location loc,
                               std::string &reason);

  /// Copy masked elements between an original DDR buffer and a DDR tile.
  /// `addPrevious` implements the matching load/add/store epilogue in DDR.
  LogicalResult copyDdrTile(OpBuilder &builder, Location loc, bool toTile,
                            const AccessSpec &access, const MaskPlan &plan,
                            unsigned rowAxis, Value root, Value tile,
                            std::string &reason, bool addPrevious = false);

  LogicalResult emitReduction(OpBuilder &builder, Location loc,
                              const Value &accumulator, Value &result,
                              Value blockOffset, std::string &reason);

  Value createAccumulatorSeed(OpBuilder &builder, Location loc);

  LogicalResult rewrite(std::string &reason);

  MatmulRegion &region;
  SpmWorkspace &workspace;
  MLIRContext *context;
  bool dmaAbiPresent = false;
  SmallVectorImpl<Value> &convertedRoots;

  triton::FuncOp func;
  ModuleOp module;

  DenseMap<Value, Value> rootCache;
  RegionWindows windows;
  Value accVector;
};

FailureOr<Value> RegionLowering::getRoot(Value base, std::string &reason) {
  auto it = rootCache.find(base);
  if (it != rootCache.end())
    return it->second;

  auto argument = dyn_cast<BlockArgument>(base);
  if (!argument || argument.getOwner() != &func.getBody().front()) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "the DDR address of a target access is rooted at " << base
           << ", which is not a function argument; only a root pointer "
              "argument can become the byte-addressable root the staging and "
              "the DMA endpoints need";
    reason = stream.str();
    return failure();
  }
  if (!isa<PointerType>(argument.getType())) {
    reason = "the DDR root of a target access is not a `!tt.ptr` argument";
    return failure();
  }
  auto memrefType =
      MemRefType::get({ShapedType::kDynamic}, IntegerType::get(context, 8));
  argument.setType(memrefType);
  rootCache[base] = argument;
  convertedRoots.push_back(argument);
  return argument;
}

LogicalResult RegionLowering::createWindows(OpBuilder &builder, Location loc,
                                            std::string &reason) {
  const int64_t elementBytesA =
      elementByteWidth(region.aAccess.address.elementType);
  const int64_t elementBytesB =
      elementByteWidth(region.bAccess.address.elementType);
  const int64_t elementBytesC =
      elementByteWidth(region.storeAccess.address.elementType);

  // The A and B windows hold every staged K block of the accumulation, one
  // next to the other, so the loop body can address them with a dynamic (but
  // assuredly 64 B aligned) byte shift.
  int64_t aPitch = alignTo(region.stagedK * elementBytesA, kDmaGranularity);
  int64_t bPitch = alignTo(region.stagedK * elementBytesB, kDmaGranularity);
  int64_t accPitch = alignTo(region.n * elementBytesC, kDmaGranularity);

  FailureOr<SpmWindow> a =
      workspace.allocate(0, region.m * aPitch, "A staging", reason);
  if (failed(a))
    return failure();
  FailureOr<SpmWindow> b =
      workspace.allocate(1, region.n * bPitch, "B staging", reason);
  if (failed(b))
    return failure();
  FailureOr<SpmWindow> acc =
      workspace.allocate(0, region.m * accPitch, "accumulator", reason);
  if (failed(acc))
    return failure();

  auto advance = [&](Operation *op) { builder.setInsertionPointAfter(op); };
  windows.bank0View = workspace.getBankView(builder, loc, 0, module);
  advance(windows.bank0View.getDefiningOp());
  windows.bank1View = workspace.getBankView(builder, loc, 1, module);
  advance(windows.bank1View.getDefiningOp());

  auto makeWindows = [&](const SpmWindow &window, int64_t bytes, int64_t rows,
                         int64_t cols, Type elementType, int64_t pitch,
                         Value bankView, Value &byteView, Value &typedView) {
    auto byteOp = memref::ViewOp::create(
        builder, loc, byteWindowType(bytes, window), bankView,
        indexConstant(builder, loc, window.offset), ValueRange{});
    byteView = byteOp.getResult();
    advance(byteOp.getOperation());
    // `memref.view` is the one view operation that can change the element
    // type, and `memref.reinterpret_cast` is the one that can give the window
    // its row pitch; the two together describe the same bytes as the 1-D i8
    // view the DMA endpoints use.
    int64_t elementBytes = elementByteWidth(elementType);
    auto elementView = memref::ViewOp::create(
        builder, loc,
        MemRefType::get({bytes / elementBytes}, elementType,
                        MemRefLayoutAttrInterface(),
                        SPMAttr::get(context, window.bank)),
        bankView, indexConstant(builder, loc, window.offset), ValueRange{});
    advance(elementView.getOperation());
    SmallVector<OpFoldResult, 2> sizes{builder.getIndexAttr(rows),
                                       builder.getIndexAttr(cols)};
    SmallVector<OpFoldResult, 2> strides{
        builder.getIndexAttr(pitch / elementBytes), builder.getIndexAttr(1)};
    auto typedOp = memref::ReinterpretCastOp::create(
        builder, loc, typedWindowType(rows, cols, elementType, pitch, window),
        elementView.getResult(), builder.getIndexAttr(0), sizes, strides);
    typedView = typedOp.getResult();
    advance(typedOp.getOperation());
  };

  makeWindows(*a, region.m * aPitch, region.m, region.stagedK,
              region.aAccess.address.elementType, aPitch, windows.bank0View,
              windows.aBytes, windows.aTyped);
  makeWindows(*b, region.n * bPitch, region.n, region.stagedK,
              region.bAccess.address.elementType, bPitch, windows.bank1View,
              windows.bBytes, windows.bTyped);
  makeWindows(*acc, region.m * accPitch, region.m, region.n,
              region.storeAccess.address.elementType, accPitch,
              windows.bank0View, windows.accBytes, windows.accTyped);

  // Scratch is reused for masked A/B packing, the ACC seed, and the result.
  // Each DMA is waited for before the next reuse; no input needs padding.
  int64_t scratchBytes = region.m * accPitch;
  region.scratchBytesCount = scratchBytes;
  int64_t packingBytes = std::max(
      region.m * alignTo(region.kPadded * elementBytesA, kDmaGranularity),
      region.n * alignTo(region.kPadded * elementBytesB, kDmaGranularity));
  func->setAttr("bosc_ame.scratch_bytes", builder.getI64IntegerAttr(std::max(
                                              scratchBytes, packingBytes)));
  auto scratchByteOp = memref::ViewOp::create(
      builder, loc,
      MemRefType::get({scratchBytes}, IntegerType::get(context, 8)),
      region.scratch, indexConstant(builder, loc, 0), ValueRange{});
  region.scratchBytes = scratchByteOp.getResult();
  advance(scratchByteOp.getOperation());
  int64_t elementBytes =
      elementByteWidth(region.storeAccess.address.elementType);
  auto scratchElementOp = memref::ViewOp::create(
      builder, loc,
      MemRefType::get({scratchBytes / elementBytes},
                      region.storeAccess.address.elementType),
      region.scratch, indexConstant(builder, loc, 0), ValueRange{});
  advance(scratchElementOp.getOperation());
  SmallVector<OpFoldResult, 2> scratchSizes{builder.getIndexAttr(region.m),
                                            builder.getIndexAttr(region.n)};
  SmallVector<OpFoldResult, 2> scratchStrides{
      builder.getIndexAttr(accPitch / elementBytes), builder.getIndexAttr(1)};
  auto scratchTypedOp = memref::ReinterpretCastOp::create(
      builder, loc,
      MemRefType::get(
          {region.m, region.n}, region.storeAccess.address.elementType,
          StridedLayoutAttr::get(context, 0, {accPitch / elementBytes, 1})),
      scratchElementOp.getResult(), builder.getIndexAttr(0), scratchSizes,
      scratchStrides);
  region.scratchTyped = scratchTypedOp.getResult();
  advance(scratchTypedOp.getOperation());

  windows.a = *a;
  windows.b = *b;
  windows.acc = *acc;
  windows.aPitch = aPitch;
  windows.bPitch = bPitch;
  windows.accPitch = accPitch;
  return success();
}

Value RegionLowering::createAccumulatorSeed(OpBuilder &builder, Location loc) {
  Type elementType = region.storeAccess.address.elementType;

  // Reinitialize on every program, including when scratch held a prior result.
  // Keep the original dot initializer; `previous` is added under its mask in
  // the epilogue, so neither a nonzero initializer nor a boundary lane is lost.
  Value initial = arith::ConstantOp::create(
      builder, loc, elementType,
      builder.getIntegerAttr(elementType, *region.accInitValue));
  emitDdrPolicy(builder, loc, /*write=*/true);
  linalg::FillOp::create(builder, loc, ValueRange{initial},
                         ValueRange{region.scratchTyped});
  auto start = DmaStartOp::create(
      builder, loc, DMATokenType::get(context, /*channel=*/0),
      region.scratchBytes, windows.accBytes,
      indexConstant(builder, loc, region.scratchBytesCount),
      builder.getI32IntegerAttr(0));
  DmaWaitOp::create(builder, loc, start.getToken());
  return TileLoadAccOp::create(
      builder, loc, VectorType::get({region.m, region.n}, elementType),
      windows.accTyped);
}

LogicalResult RegionLowering::stageAllBlocks(OpBuilder &builder, Location loc,
                                             std::string &reason) {
  // Symbols whose defining operation lives inside the loop have to be rebuilt
  // in the prologue with the loop counter substituted: the staged addresses
  // and masks are affine in those symbols, and the loop body is not where the
  // staging happens any more.
  const SmallVector<Value> symbols{region.symbols};

  for (int64_t iteration = 0; iteration < region.stagedIterations;
       ++iteration) {
    // The staged addresses are affine in the enclosing loop's induction
    // variable; enumerating the iterations means materializing each one as a
    // constant. The substitution only feeds the address and mask
    // materialization below. The constant carries the type of the original
    // induction variable so that the rebuilt scalar chain type-checks.
    IRMapping hoisted;
    Value counter;
    if (region.loop) {
      Type ivType = region.loop.getInductionVar().getType();
      counter = arith::ConstantOp::create(
          builder, loc, ivType,
          builder.getIntegerAttr(ivType, *region.loopLowerBound +
                                             iteration * *region.loopStep));
      hoisted.map(region.loop.getInductionVar(), counter);
    }
    if (region.loopIvSymbol)
      region.symbols[*region.loopIvSymbol] = counter;
    if (region.loop) {
      std::function<Value(Value)> rebuild = [&](Value value) -> Value {
        if (Value known = hoisted.lookupOrNull(value))
          return known;
        Operation *defining = value.getDefiningOp();
        if (!defining || !region.loop->isProperAncestor(defining))
          return value;
        for (Value operand : defining->getOperands())
          hoisted.map(operand, rebuild(operand));
        Operation *clone = builder.clone(*defining, hoisted);
        hoisted.map(value, clone->getResult(0));
        return clone->getResult(0);
      };
      for (auto [position, symbol] : llvm::enumerate(symbols)) {
        Operation *defining = symbol.getDefiningOp();
        if (defining && region.loop->isProperAncestor(defining))
          region.symbols[position] = rebuild(symbol);
      }
    }
    const int64_t column = iteration * region.kPadded;
    if (failed(stageIn(builder, loc, region.aAccess, /*rowAxis=*/0,
                       region.aAccess.address.base, windows.a, windows.aBytes,
                       windows.aTyped, windows.aPitch, region.m, region.kPadded,
                       column, reason)) ||
        failed(stageIn(builder, loc, region.bAccess, /*rowAxis=*/1,
                       region.bAccess.address.base, windows.b, windows.bBytes,
                       windows.bTyped, windows.bPitch, region.n, region.kPadded,
                       column, reason)))
      return failure();
  }
  region.symbols.assign(symbols.begin(), symbols.end());

  // The `previous` tile stays in DDR and is read only under the store mask.
  return success();
}

LogicalResult RegionLowering::stageIn(
    OpBuilder &builder, Location loc, const AccessSpec &access,
    unsigned rowAxis, Value root, const SpmWindow &window, Value windowBytes,
    Value windowTyped, int64_t pitchBytes, int64_t rows, int64_t cols,
    int64_t columnOffset, std::string &reason) {
  FailureOr<MaskPlan> plan =
      buildMaskPlan(access.mask, rowAxis, 1 - rowAxis, region.symbols.size(),
                    context, reason);
  if (failed(plan))
    return failure();
  const int64_t elementBytes = elementByteWidth(access.address.elementType);
  const int64_t logicalCols =
      rowAxis == 0 ? access.address.cols : access.address.rows;
  const int64_t rowStride =
      rowAxis == 0 ? access.address.rowStride : access.address.colStride;
  const int64_t rowBytes = alignTo(cols * elementBytes, kDmaGranularity);
  const int64_t paddedCols = rowBytes / elementBytes;

  // Direct DMA requires every logical lane to be valid, no K padding, and
  // aligned original rows. A runtime predicate selects the boundary fallback.
  Value origin = buildIndexValue(builder, loc, access.address.origin);
  if (!origin) {
    reason = "the operand origin could not be materialized";
    return failure();
  }
  Value byteOrigin = arith::MulIOp::create(
      builder, loc, origin, indexConstant(builder, loc, elementBytes));
  bool direct = dmaAbiPresent && logicalCols == paddedCols &&
                (rowStride * elementBytes) % kDmaGranularity == 0 &&
                isProvablyMultipleOf(byteOrigin, kDmaGranularity);
  Value fullTile;
  auto addGuard = [&](Value guard) {
    fullTile =
        fullTile ? arith::AndIOp::create(builder, loc, fullTile, guard) : guard;
  };
  if (direct) {
    for (const MaskPlan::RowTerm &term : plan->rowTerms) {
      AffineExpr last = simplifyAffineExpr(
          term.rest + constantExpr(context, term.coeff * (rows - 1)), 0,
          region.symbols.size());
      IndexRange range = inferAffineRange(last, {}, region.symbols);
      if (range.isProvablyNegative())
        continue;
      if (range.lo && *range.lo >= 0) {
        direct = false;
        break;
      }
      addGuard(arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                     buildIndexValue(builder, loc, last),
                                     indexConstant(builder, loc, 0)));
    }
    for (AffineExpr bound : plan->colBounds) {
      IndexRange range = inferAffineRange(bound, {}, region.symbols);
      if (range.lo && *range.lo >= logicalCols)
        continue;
      if (range.hi && *range.hi < logicalCols) {
        direct = false;
        break;
      }
      addGuard(arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                     buildIndexValue(builder, loc, bound),
                                     indexConstant(builder, loc, logicalCols)));
    }
  }

  auto transfer = [&](OpBuilder &nested, Location nestedLoc, Value source,
                      int64_t row) {
    Value destination = memref::SubViewOp::create(
        nested, nestedLoc, windowBytes,
        ArrayRef<OpFoldResult>{nested.getIndexAttr(columnOffset * elementBytes +
                                                   row * pitchBytes)},
        ArrayRef<OpFoldResult>{nested.getIndexAttr(rowBytes)},
        ArrayRef<OpFoldResult>{nested.getIndexAttr(1)});
    auto start = DmaStartOp::create(
        nested, nestedLoc, DMATokenType::get(context, /*channel=*/0), source,
        destination, indexConstant(nested, nestedLoc, rowBytes),
        nested.getI32IntegerAttr(0));
    DmaWaitOp::create(nested, nestedLoc, start.getToken());
  };
  auto emitDirect = [&](OpBuilder &nested, Location nestedLoc) {
    for (int64_t row = 0; row < rows; ++row) {
      Value offset = buildIndexValue(
          nested, nestedLoc,
          simplifyAffineExpr(access.address.origin +
                                 constantExpr(context, row * rowStride),
                             0, region.symbols.size()));
      Value bytes =
          arith::MulIOp::create(nested, nestedLoc, offset,
                                indexConstant(nested, nestedLoc, elementBytes));
      Value source = memref::ViewOp::create(
          nested, nestedLoc, MemRefType::get({rowBytes}, nested.getI8Type()),
          root, bytes, ValueRange{});
      transfer(nested, nestedLoc, source, row);
    }
  };
  LogicalResult packedStatus = success();
  auto emitPacked = [&](OpBuilder &nested, Location nestedLoc) {
    // Only the private DDR tile is padded. A masked original address is never
    // loaded, and lanes beyond the logical K stay zero even for other != 0.
    Value packed = memref::ViewOp::create(
        nested, nestedLoc,
        MemRefType::get({rows, paddedCols}, access.address.elementType),
        region.scratch, indexConstant(nested, nestedLoc, 0), ValueRange{});
    Value zero = arith::ConstantOp::create(
        nested, nestedLoc, access.address.elementType,
        nested.getIntegerAttr(access.address.elementType, 0));
    emitDdrPolicy(nested, nestedLoc, /*write=*/true);
    linalg::FillOp::create(nested, nestedLoc, ValueRange{zero},
                           ValueRange{packed});
    int64_t other = access.other ? *getSplatValue(access.other) : 0;
    if (other != 0) {
      Value logical = memref::SubViewOp::create(
          nested, nestedLoc, packed,
          ArrayRef<OpFoldResult>{nested.getIndexAttr(0),
                                 nested.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{nested.getIndexAttr(rows),
                                 nested.getIndexAttr(logicalCols)},
          ArrayRef<OpFoldResult>{nested.getIndexAttr(1),
                                 nested.getIndexAttr(1)});
      Value fill = arith::ConstantOp::create(
          nested, nestedLoc, access.address.elementType,
          nested.getIntegerAttr(access.address.elementType, other));
      linalg::FillOp::create(nested, nestedLoc, ValueRange{fill},
                             ValueRange{logical});
    }
    packedStatus = copyDdrTile(nested, nestedLoc, /*toTile=*/true, access,
                               *plan, rowAxis, root, packed, reason);
    if (failed(packedStatus))
      return;
    for (int64_t row = 0; row < rows; ++row) {
      Value source = memref::ViewOp::create(
          nested, nestedLoc, MemRefType::get({rowBytes}, nested.getI8Type()),
          region.scratch, indexConstant(nested, nestedLoc, row * rowBytes),
          ValueRange{});
      transfer(nested, nestedLoc, source, row);
    }
  };
  if (!direct) {
    emitPacked(builder, loc);
  } else if (!fullTile) {
    emitDirect(builder, loc);
  } else {
    scf::IfOp::create(
        builder, loc, fullTile,
        [&](OpBuilder &nested, Location nestedLoc) {
          emitDirect(nested, nestedLoc);
          scf::YieldOp::create(nested, nestedLoc);
        },
        [&](OpBuilder &nested, Location nestedLoc) {
          emitPacked(nested, nestedLoc);
          scf::YieldOp::create(nested, nestedLoc);
        });
  }
  (void)window;
  (void)windowTyped;
  return packedStatus;
}

LogicalResult RegionLowering::copyDdrTile(OpBuilder &builder, Location loc,
                                          bool toTile, const AccessSpec &access,
                                          const MaskPlan &plan,
                                          unsigned rowAxis, Value root,
                                          Value tile, std::string &reason,
                                          bool addPrevious) {
  const AddressSpec &address = access.address;
  const int64_t rows = rowAxis == 0 ? address.rows : address.cols;
  const int64_t cols = rowAxis == 0 ? address.cols : address.rows;
  const int64_t elementBytes = elementByteWidth(address.elementType);
  const int64_t rowStride =
      rowAxis == 0 ? address.rowStride : address.colStride;
  assert(!cast<MemRefType>(tile.getType()).getMemorySpace() &&
         "CPU copies must only access DDR");
  for (int64_t row = 0; row < rows; ++row) {
    bool invalid = false;
    Value rowGuard;
    for (const MaskPlan::RowTerm &term : plan.rowTerms) {
      AffineExpr substituted = simplifyAffineExpr(
          term.rest + constantExpr(context, term.coeff * row), 0,
          region.symbols.size());
      IndexRange range = inferAffineRange(substituted, {}, region.symbols);
      if (range.isProvablyNegative())
        continue;
      if (range.lo && *range.lo >= 0) {
        invalid = true;
        break;
      }
      Value condition = buildIndexValue(builder, loc, substituted);
      if (!condition) {
        reason = "a row condition of a target access could not be materialized";
        return failure();
      }
      Value guard =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                condition, indexConstant(builder, loc, 0));
      rowGuard = rowGuard ? arith::AndIOp::create(builder, loc, rowGuard, guard)
                          : guard;
    }
    if (invalid)
      continue;
    Value count = indexConstant(builder, loc, cols);
    for (AffineExpr bound : plan.colBounds) {
      IndexRange range = inferAffineRange(bound, {}, region.symbols);
      if (range.lo && *range.lo >= cols)
        continue;
      Value value = buildIndexValue(builder, loc, bound);
      if (!value) {
        reason = "a column bound of a target access could not be materialized";
        return failure();
      }
      count = arith::MinSIOp::create(builder, loc, count, value);
    }
    count = arith::MaxSIOp::create(builder, loc, count,
                                   indexConstant(builder, loc, 0));
    Value origin = buildIndexValue(
        builder, loc,
        simplifyAffineExpr(address.origin +
                               constantExpr(context, row * rowStride),
                           0, region.symbols.size()));
    if (!origin) {
      reason = "the row address of a target access could not be materialized";
      return failure();
    }
    Value byteOffset = arith::MulIOp::create(
        builder, loc, origin, indexConstant(builder, loc, elementBytes));
    auto emitRow = [&](OpBuilder &nested, Location nestedLoc) {
      Value ddrRow = memref::ViewOp::create(
          nested, nestedLoc, MemRefType::get({cols}, address.elementType), root,
          byteOffset, ValueRange{});
      scf::ForOp::create(
          nested, nestedLoc, indexConstant(nested, nestedLoc, 0), count,
          indexConstant(nested, nestedLoc, 1), ValueRange{},
          [&](OpBuilder &body, Location bodyLoc, Value column, ValueRange) {
            SmallVector<Value> indices{indexConstant(body, bodyLoc, row),
                                       column};
            emitDdrPolicy(body, bodyLoc, /*write=*/false);
            Value loaded =
                toTile ? memref::LoadOp::create(body, bodyLoc, ddrRow, column)
                             .getResult()
                       : memref::LoadOp::create(body, bodyLoc, tile, indices)
                             .getResult();
            if (!toTile && addPrevious)
              loaded = arith::AddIOp::create(
                  body, bodyLoc, loaded,
                  memref::LoadOp::create(body, bodyLoc, ddrRow, column));
            emitDdrPolicy(body, bodyLoc, /*write=*/true);
            if (toTile)
              memref::StoreOp::create(body, bodyLoc, loaded, tile, indices);
            else
              memref::StoreOp::create(body, bodyLoc, loaded, ddrRow, column);
            scf::YieldOp::create(body, bodyLoc);
          });
    };
    if (!rowGuard) {
      emitRow(builder, loc);
    } else {
      scf::IfOp::create(builder, loc, rowGuard,
                        [&](OpBuilder &nested, Location nestedLoc) {
                          emitRow(nested, nestedLoc);
                          scf::YieldOp::create(nested, nestedLoc);
                        });
    }
  }
  return success();
}

LogicalResult RegionLowering::emitReduction(OpBuilder &builder, Location loc,
                                            const Value &accumulator,
                                            Value &result, Value blockOffset,
                                            std::string &reason) {
  (void)reason;
  Type elementTypeA = region.aAccess.address.elementType;
  Type elementTypeB = region.bAccess.address.elementType;
  Type elementTypeC = region.storeAccess.address.elementType;
  VectorType accType = VectorType::get({region.m, region.n}, elementTypeC);

  auto makeATile = [&](OpBuilder &nested, Location nestedLoc, Value base,
                       int64_t kExtent) {
    return memref::ReinterpretCastOp::create(
        nested, nestedLoc,
        MemRefType::get(
            {region.m, kExtent}, elementTypeA,
            StridedLayoutAttr::get(context, 0,
                                   SmallVector<int64_t, 2>{windows.aPitch, 1}),
            SPMAttr::get(context, windows.a.bank)),
        base, nested.getIndexAttr(0),
        ArrayRef<OpFoldResult>{nested.getIndexAttr(region.m),
                               nested.getIndexAttr(kExtent)},
        ArrayRef<OpFoldResult>{nested.getIndexAttr(windows.aPitch),
                               nested.getIndexAttr(1)});
  };
  auto makeBTile = [&](OpBuilder &nested, Location nestedLoc, Value base,
                       int64_t kExtent) {
    return memref::ReinterpretCastOp::create(
        nested, nestedLoc,
        MemRefType::get(
            {kExtent, region.n}, elementTypeB,
            StridedLayoutAttr::get(context, 0,
                                   SmallVector<int64_t, 2>{1, windows.bPitch}),
            SPMAttr::get(context, windows.b.bank)),
        base, nested.getIndexAttr(0),
        ArrayRef<OpFoldResult>{nested.getIndexAttr(kExtent),
                               nested.getIndexAttr(region.n)},
        ArrayRef<OpFoldResult>{nested.getIndexAttr(1),
                               nested.getIndexAttr(windows.bPitch)});
  };
  auto emitSingleBlock = [&](OpBuilder &nested, Location nestedLoc, Value base,
                             Value inner, int64_t kExtent, Value aOffset,
                             Value bOffset) {
    Value aBytes = memref::ViewOp::create(
        nested, nestedLoc, byteWindowType(region.m * windows.aPitch, windows.a),
        windows.bank0View, aOffset, ValueRange{});
    Value bBytes = memref::ViewOp::create(
        nested, nestedLoc, byteWindowType(region.n * windows.bPitch, windows.b),
        windows.bank1View, bOffset, ValueRange{});
    Value ta = TileLoadAOp::create(
        nested, nestedLoc, VectorType::get({region.m, kExtent}, elementTypeA),
        makeATile(nested, nestedLoc, aBytes, kExtent));
    Value tb = TileLoadBOp::create(
        nested, nestedLoc, VectorType::get({kExtent, region.n}, elementTypeB),
        makeBTile(nested, nestedLoc, bBytes, kExtent));
    (void)base;
    return Value(TileMmaOp::create(nested, nestedLoc, accType, inner, ta, tb));
  };

  // The staged K block of the current loop iteration sits `blockOffset` bytes
  // (elements for the i8 operands) into the A/B windows.
  Value blockShift = toIndex(builder, loc, blockOffset);
  Value aBlock = arith::AddIOp::create(
      builder, loc, blockShift, indexConstant(builder, loc, windows.a.offset));
  Value bBlock = arith::AddIOp::create(
      builder, loc, blockShift, indexConstant(builder, loc, windows.b.offset));

  if (region.kPadded <= kTileDimMax) {
    result = emitSingleBlock(builder, loc, Value(), accumulator, region.k,
                             aBlock, bBlock);
    return success();
  }

  // K blocking: a real `scf.for` over the reduction that carries the
  // accumulator, so the chain stays in the register file instead of
  // round-tripping through memory between sub-blocks.
  auto loop = scf::ForOp::create(builder, loc, indexConstant(builder, loc, 0),
                                 indexConstant(builder, loc, region.kPadded),
                                 indexConstant(builder, loc, kTileDimMax),
                                 ValueRange{accumulator});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value induction = loop.getInductionVar();
    Value inner = loop.getRegionIterArg(0);
    // `blockOffset` is a multiple of the 64 B staging granularity, so the
    // shift stays a proved multiple of 64 even though it is dynamic.
    Value aShift = arith::AddIOp::create(
        builder, loc,
        arith::AddIOp::create(builder, loc, induction, blockShift),
        indexConstant(builder, loc, windows.a.offset));
    Value bShift = arith::AddIOp::create(
        builder, loc,
        arith::AddIOp::create(builder, loc, induction, blockShift),
        indexConstant(builder, loc, windows.b.offset));
    Value aBytes = memref::ViewOp::create(
        builder, loc, byteWindowType(region.m * windows.aPitch, windows.a),
        windows.bank0View, aShift, ValueRange{});
    Value bBytes = memref::ViewOp::create(
        builder, loc, byteWindowType(region.n * windows.bPitch, windows.b),
        windows.bank1View, bShift, ValueRange{});
    Value ta = TileLoadAOp::create(
        builder, loc, VectorType::get({region.m, kTileDimMax}, elementTypeA),
        makeATile(builder, loc, aBytes, kTileDimMax));
    Value tb = TileLoadBOp::create(
        builder, loc, VectorType::get({kTileDimMax, region.n}, elementTypeB),
        makeBTile(builder, loc, bBytes, kTileDimMax));
    Value next = TileMmaOp::create(builder, loc, accType, inner, ta, tb);
    scf::YieldOp::create(builder, loc, next);
  }
  result = loop.getResult(0);
  return success();
}

/// Erase an operation when nothing uses it any more.
void eraseIfDead(Operation *op) {
  if (op && op->use_empty())
    op->erase();
}

/// Erase every trivially dead operation of a function.
void eraseDeadOps(triton::FuncOp func) {
  for (int round = 0; round < 4; ++round) {
    SmallVector<Operation *> ops;
    func.walk([&](Operation *op) { ops.push_back(op); });
    bool changed = false;
    for (Operation *op : llvm::reverse(ops)) {
      if (op == func.getOperation())
        continue;
      if (isOpTriviallyDead(op)) {
        op->erase();
        changed = true;
      }
    }
    if (!changed)
      break;
  }
}

LogicalResult RegionLowering::rewrite(std::string &reason) {
  func = region.dot->getParentOfType<triton::FuncOp>();
  if (!func) {
    reason = "the target dot is not inside a `tt.func`";
    return failure();
  }
  module = func->getParentOfType<ModuleOp>();
  Location loc = region.dot.getLoc();

  // 1. DDR roots become byte-addressable memref arguments, so that the staging
  //    views and the DMA endpoints describe the same bytes.
  for (Value base : {region.aAccess.address.base, region.bAccess.address.base,
                     region.storeAccess.address.base}) {
    if (failed(getRoot(base, reason)))
      return failure();
  }
  if (region.hasPrevious)
    if (failed(getRoot(region.previousAccess.address.base, reason)))
      return failure();
  // The scratch window the accumulator tile is transferred into.  It is a
  // normal function argument, so the caller states its alignment and size
  // through the same `bosc_ame.dma.abi` contract as the data buffers.
  region.scratch = func.getBody().front().addArgument(
      MemRefType::get({ShapedType::kDynamic}, IntegerType::get(context, 8)),
      loc);
  // The argument types changed with the roots; the function signature has to
  // follow, and its DAC ABI is otherwise untouched (the program information
  // arguments are appended once, by the same rule the ordinary TTIR path
  // uses).
  {
    SmallVector<Type> inputs;
    for (BlockArgument argument : func.getArguments())
      inputs.push_back(argument.getType());
    func.setFunctionType(FunctionType::get(
        context, inputs, func.getFunctionType().getResults()));
  }

  // 2. The fixed SPM workspace in the entry block, so every window dominates
  //    the loop and the epilogue.
  OpBuilder entry = OpBuilder::atBlockBegin(&func.getBody().front());
  if (failed(createWindows(entry, loc, reason)))
    return failure();

  // 3. Stage the whole accumulation, and the `previous` tile, before the
  //    accumulator is loaded, so that the accumulator's live range holds no
  //    call at all (see `stageAllBlocks`).
  OpBuilder prologue(region.loop ? region.loop.getOperation()
                                 : region.dot.getOperation());
  if (failed(stageAllBlocks(prologue, loc, reason)))
    return failure();

  // 4. The accumulator seed.
  OpBuilder builder(region.loop ? region.loop.getOperation()
                                : region.dot.getOperation());
  accVector = createAccumulatorSeed(builder, loc);

  // 5. The reduction.
  Value accMatrix;
  if (region.loop) {
    scf::ForOp loop = region.loop;
    SmallVector<Value> inits(loop.getInitArgs().begin(),
                             loop.getInitArgs().end());
    inits[region.accArgument] = accVector;
    auto newLoop =
        scf::ForOp::create(builder, loc, loop.getLowerBound(),
                           loop.getUpperBound(), loop.getStep(), inits);
    newLoop.getRegion().takeBody(loop.getRegion());
    Block &body = newLoop.getRegion().front();
    body.getArgument(region.accArgument + 1).setType(accVector.getType());

    // The staged K block of this iteration starts `iteration * kPadded` bytes
    // into the A/B windows. `kPadded` is a multiple of the 64 B staging
    // granularity, so the shift is provably aligned even though the iteration
    // is only known at run time.
    OpBuilder bodyBuilder(region.dot);
    Value induction = newLoop.getInductionVar();
    Type ivType = induction.getType();
    auto ivConstant = [&](int64_t value) {
      return arith::ConstantOp::create(
          bodyBuilder, loc, ivType, bodyBuilder.getIntegerAttr(ivType, value));
    };
    Value iteration = induction;
    if (*region.loopLowerBound != 0)
      iteration = arith::SubIOp::create(bodyBuilder, loc, iteration,
                                        ivConstant(*region.loopLowerBound));
    if (*region.loopStep != 1)
      iteration = arith::DivUIOp::create(bodyBuilder, loc, iteration,
                                         ivConstant(*region.loopStep));
    Value blockOffset = arith::MulIOp::create(bodyBuilder, loc, iteration,
                                              ivConstant(region.kPadded));

    Value reduced;
    if (failed(emitReduction(bodyBuilder, loc,
                             body.getArgument(region.accArgument + 1), reduced,
                             blockOffset, reason)))
      return failure();
    region.dot.getResult().replaceAllUsesWith(reduced);
    accMatrix = newLoop.getResult(region.accArgument);
    eraseIfDead(region.dot.getOperation());
    eraseIfDead(region.aLoad.getOperation());
    eraseIfDead(region.bLoad.getOperation());
    for (auto [oldResult, newResult] :
         llvm::zip(loop.getResults(), newLoop.getResults()))
      oldResult.replaceAllUsesWith(newResult);
    eraseIfDead(loop.getOperation());
  } else {
    OpBuilder reductionBuilder(region.dot);
    Value reduced;
    if (failed(emitReduction(reductionBuilder, loc, accVector, reduced,
                             indexConstant(reductionBuilder, loc, 0), reason)))
      return failure();
    region.dot.getResult().replaceAllUsesWith(reduced);
    accMatrix = reduced;
    eraseIfDead(region.dot.getOperation());
    eraseIfDead(region.aLoad.getOperation());
    eraseIfDead(region.bLoad.getOperation());
  }

  // 6. Materialize ACC in SPM, DMA it to DDR scratch, then add `previous` and
  //    write back under the original mask. No CPU instruction touches SPM.
  OpBuilder epilogue(region.store);

  // The accumulator leaves SPM through a transfer, never through a CPU load:
  // the E6 cannot complete a load from the SPM window, so the tile is copied
  // into the caller's scratch window and every arithmetic step that follows
  // happens in DDR.
  TileStoreAccOp::create(epilogue, loc, accMatrix, windows.accTyped);
  {
    Value length = indexConstant(epilogue, loc, region.scratchBytesCount);
    auto start = DmaStartOp::create(epilogue, loc,
                                    DMATokenType::get(context, /*channel=*/0),
                                    windows.accBytes, region.scratchBytes,
                                    length, epilogue.getI32IntegerAttr(0));
    DmaWaitOp::create(epilogue, loc, start.getToken());
  }

  FailureOr<MaskPlan> storePlan =
      buildMaskPlan(region.storeAccess.mask, /*rowAxis=*/0, /*colAxis=*/1,
                    region.symbols.size(), context, reason);
  if (failed(storePlan))
    return failure();
  if (failed(copyDdrTile(epilogue, loc, /*toTile=*/false, region.storeAccess,
                         *storePlan, /*rowAxis=*/0,
                         region.storeAccess.address.base, region.scratchTyped,
                         reason, /*addPrevious=*/region.hasPrevious)))
    return failure();

  // 6. The original scalar chain is dead now: the loads the region described,
  //    the arithmetic that only existed to address them and the store whose
  //    place the write-back took.
  eraseIfDead(region.store.getOperation());
  if (region.addi)
    eraseIfDead(region.addi.getOperation());
  if (region.hasPrevious)
    eraseIfDead(region.previousLoad.getOperation());
  eraseDeadOps(func);
  return success();
}

LogicalResult RegionLowering::run(std::string &reason) {
  return rewrite(reason);
}

//===----------------------------------------------------------------------===//
// Function finishing: the kernel ABI of the rewritten function
//===----------------------------------------------------------------------===//

/// The launch grid rank the program information arguments carry: three
/// `num_programs` followed by three `program_id` values, appended after the
/// kernel's own arguments.  This is the ABI the ordinary TTIR path produces,
/// and the two must agree: a rewritten function is already a `func.func`, so
/// the ordinary conversion leaves it (and its argument list) alone.
constexpr unsigned kLaunchGridRank = 3;

/// Turn a rewritten `tt.func` into the `func.func` the rest of the pipeline
/// expects, with the program information arguments appended exactly once.
LogicalResult finishFunction(triton::FuncOp function) {
  OpBuilder builder(function);
  ModuleOp module = function->getParentOfType<ModuleOp>();

  if (function->hasAttr("tt.noinline"))
    function->removeAttr("tt.noinline");

  // 1. The program information arguments.
  auto originalType = function.getFunctionType();
  SmallVector<Type> inputs(originalType.getInputs());
  inputs.append(2 * kLaunchGridRank, builder.getI32Type());
  auto newType = builder.getFunctionType(inputs, originalType.getResults());
  function.setFunctionType(newType);
  if (function.getAllArgAttrs()) {
    SmallVector<DictionaryAttr> argAttrs;
    function.getAllArgAttrs(argAttrs);
    argAttrs.append(2 * kLaunchGridRank, DictionaryAttr());
    function.setAllArgAttrs(argAttrs);
  }
  for (unsigned index = 0; index < 2 * kLaunchGridRank; ++index)
    function.getBody().front().addArgument(builder.getI32Type(),
                                           function.getLoc());

  // 2. `tt.get_program_id` / `tt.get_num_programs` read those arguments.
  //    The operations are collected first: a walk that rewrites its own
  //    operations would visit freed nodes.
  SmallVector<Operation *> programIds, numProgramsOps;
  function.walk([&](Operation *op) {
    if (isa<GetProgramIdOp>(op))
      programIds.push_back(op);
    else if (isa<GetNumProgramsOp>(op))
      numProgramsOps.push_back(op);
  });
  for (Operation *op : programIds) {
    auto programId = cast<GetProgramIdOp>(op);
    unsigned axis = static_cast<unsigned>(programId.getAxis());
    Value replacement = function.getArgument(function.getNumArguments() -
                                             kLaunchGridRank + axis);
    programId.getResult().replaceAllUsesWith(replacement);
    programId.erase();
  }
  for (Operation *op : numProgramsOps) {
    auto numPrograms = cast<GetNumProgramsOp>(op);
    unsigned axis = static_cast<unsigned>(numPrograms.getAxis());
    Value replacement = function.getArgument(function.getNumArguments() -
                                             2 * kLaunchGridRank + axis);
    numPrograms.getResult().replaceAllUsesWith(replacement);
    numPrograms.erase();
  }

  // 3. Nothing else of the Triton dialect may be left: this pass lowers the
  //    target matmul regions only.
  bool failed = false;
  function.walk([&](Operation *op) {
    if (isa<triton::FuncOp, triton::ReturnOp>(op))
      return;
    if (op->getDialect()->getNamespace() !=
        TritonDialect::getDialectNamespace())
      return;
    op->emitError()
        << "'" << op->getName()
        << "' is Triton code outside a target matmul region; this pass lowers "
           "the target matmul regions only, so the function cannot be turned "
           "into the `func.func` the rest of the FPGA pipeline needs";
    failed = true;
  });
  if (failed)
    return failure();

  // 4. `tt.func` / `tt.return` become their `func` counterparts.
  SmallVector<DictionaryAttr> argAttrs, resAttrs;
  function.getAllArgAttrs(argAttrs);
  function.getAllResultAttrs(resAttrs);
  auto funcFunc = builder.create<func::FuncOp>(function.getLoc(),
                                               function.getName(), newType);
  funcFunc.setAllArgAttrs(argAttrs);
  funcFunc.setAllResultAttrs(resAttrs);
  funcFunc.setVisibility(function.getVisibility());
  funcFunc->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  if (Attribute scratchBytes = function->getAttr("bosc_ame.scratch_bytes"))
    funcFunc->setAttr("bosc_ame.scratch_bytes", scratchBytes);
  IRMapping mapping;
  function.getBody().cloneInto(&funcFunc.getBody(), mapping);
  for (Block &block : funcFunc.getBody().getBlocks()) {
    Operation *terminator = block.getTerminator();
    builder.setInsertionPoint(terminator);
    builder.create<func::ReturnOp>(function.getLoc(),
                                   terminator->getOperands());
    terminator->erase();
  }
  function.erase();
  (void)module;
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TritonToBOSCAMEPass
    : public ::impl::TritonToBOSCAMEBase<TritonToBOSCAMEPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    SmallVector<DotOp> dots;
    module.walk([&](DotOp dot) { dots.push_back(dot); });

    // Which dots are targeted at all: the AME datapath implements signed
    // `i8 x i8 -> i32`.  A floating point dot is not silently turned into an
    // integer one - it is left to the ordinary TTIR -> Linalg conversion.
    SmallVector<DotOp> targets;
    for (DotOp dot : dots) {
      auto lhsType = dyn_cast<RankedTensorType>(dot.getA().getType());
      auto rhsType = dyn_cast<RankedTensorType>(dot.getB().getType());
      auto resultType = dyn_cast<RankedTensorType>(dot.getType());
      if (!lhsType || !rhsType || !resultType)
        continue;
      if (!lhsType.getElementType().isInteger(8) ||
          !rhsType.getElementType().isInteger(8) ||
          !resultType.getElementType().isInteger(32))
        continue;
      targets.push_back(dot);
    }
    if (targets.empty())
      return;

    // SPM is only accessible through DMA/AME, so aligned, nonaliasing DDR
    // scratch is required even when the original operand rows need packing.
    // `getAttrOfType` returns a *null* attribute when the module does not
    // carry one, which is not the same as `std::nullopt`.
    StringAttr abi = module->getAttrOfType<StringAttr>(kDmaAbiAttrName);
    bool abiPresent = static_cast<bool>(abi);
    if (!dmaAbi.empty()) {
      if (dmaAbi != kDmaAbiAlignedNonAliasing.str()) {
        module.emitError() << "`" << dmaAbi
                           << "` is not a DMA caller contract this pass "
                              "implements; the only contract it records is `"
                           << kDmaAbiAlignedNonAliasing << "`";
        signalPassFailure();
        return;
      }
      abi = StringAttr::get(context, dmaAbi);
      module->setAttr(kDmaAbiAttrName, abi);
      abiPresent = true;
    }
    if (!abiPresent || abi.getValue() != kDmaAbiAlignedNonAliasing) {
      module.emitError()
          << "SPM cannot be accessed by the CPU; the BOSCAME path requires "
             "64 B aligned, non-overlapping DDR buffers and scratch; pass "
             "--triton-to-boscame=dma-abi="
          << kDmaAbiAlignedNonAliasing;
      signalPassFailure();
      return;
    }

    if (failed(declareDdrPolicies(module))) {
      signalPassFailure();
      return;
    }

    SpmWorkspace workspace;
    SmallVector<Value> convertedRoots;
    SmallPtrSet<Operation *, 4> rewrittenFunctions;
    for (DotOp dot : targets) {
      std::string reason;
      FailureOr<MatmulRegion> region = analyzeRegion(dot, reason);
      if (failed(region)) {
        dot.emitError()
            << "cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: "
            << reason;
        signalPassFailure();
        return;
      }
      RegionLowering lowering(*region, workspace, abiPresent, convertedRoots);
      rewrittenFunctions.insert(dot->getParentOfType<triton::FuncOp>());
      if (failed(lowering.run(reason))) {
        dot.emitError()
            << "cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: "
            << reason;
        signalPassFailure();
        return;
      }
    }

    // A root argument that is still used by Triton code after the rewrite is
    // used by a region the pass does not lower, and its type has already
    // changed: the argument cannot be a pointer for one use and a
    // byte-addressable staging root for another.
    for (Value root : convertedRoots) {
      for (Operation *user : root.getUsers()) {
        if (user->getDialect()->getNamespace() !=
            TritonDialect::getDialectNamespace())
          continue;
        user->emitError()
            << "uses the pointer argument " << root
            << " outside the matmul region that made it byte addressable; the "
               "argument cannot be a pointer for one use and an SPM staging "
               "root for another";
        signalPassFailure();
        return;
      }
    }

    // A rewritten function is done with TTIR: it is turned into a `func.func`
    // with the program information the kernel ABI carries, exactly once, and
    // it must not keep Triton operations the pass does not lower.
    SmallVector<triton::FuncOp> finished;
    for (triton::FuncOp function : module.getOps<triton::FuncOp>())
      if (rewrittenFunctions.contains(function.getOperation()))
        finished.push_back(function);
    for (triton::FuncOp function : finished) {
      if (failed(finishFunction(function)))
        return;
    }

    // The rewritten module runs the FPGA AME datapath, so it has to say so:
    // tile operations are only lowered against an explicit contract.
    if (!module->getAttrOfType<StringAttr>(kAmeTargetAttrName))
      module->setAttr(kAmeTargetAttrName,
                      StringAttr::get(context, kAmeTargetNrFpga));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToBOSCAMEPass() {
  return std::make_unique<TritonToBOSCAMEPass>();
}
