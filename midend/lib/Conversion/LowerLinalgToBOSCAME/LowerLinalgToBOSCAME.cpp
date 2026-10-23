//=== LowerLinalgToBOSCAME.cpp - Linalg to BOSCAME Dialect Lowering Pass --===//
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
// This file defines Linalg dialect lowering pass to BOSCAME dialect.
//
//===----------------------------------------------------------------------===//

#include "Analysis/LinalgComputationFactsPrinter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "Analysis/BOSCAMELaunchFacts.h"
#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEAmeEmitter.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEDmaEmitter.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanner.h"
#include "Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/Transform.h"

using namespace mlir;
using namespace ::buddy::boscame;

namespace {

enum class ElementwiseKind {
  Add,
  Sub,
  Mul,
  MinS,
  MinU,
  MaxS,
  MaxU,
  FAdd,
  FSub,
  FMul,
  FMax,
};

enum class BroadcastKind {
  Row,
  Column,
  Element,
};

enum class BroadcastRegisterKind {
  TileA,
  TileB,
  Accumulation,
};

static unsigned getElementBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return 0;
}

static unsigned getElementByteWidth(Type type) {
  unsigned bitWidth = getElementBitWidth(type);
  return bitWidth == 0 ? 0 : (bitWidth + 7) / 8;
}

static LogicalResult getMsetTypeImm(Type elementType, int64_t &imm) {
  if (elementType.isInteger(4)) {
    imm = 4;
    return success();
  }
  if (elementType.isInteger(8)) {
    imm = 8;
    return success();
  }
  if (elementType.isInteger(16) || elementType.isF16() ||
      elementType.isBF16()) {
    imm = 16;
    return success();
  }
  if (elementType.isInteger(32) || elementType.isF32()) {
    imm = 32;
    return success();
  }
  if (elementType.isInteger(64) || elementType.isF64()) {
    imm = 64;
    return success();
  }
  return failure();
}

static bool isParallel(utils::IteratorType type) {
  return type == utils::IteratorType::parallel;
}

static bool isReduction(utils::IteratorType type) {
  return type == utils::IteratorType::reduction;
}

static bool hasPureBufferOperands(linalg::GenericOp op) {
  return op.hasPureBufferSemantics() && op.getNumDpsInits() == 1;
}

static bool isAffineMap(AffineMap map, unsigned numDims,
                        ArrayRef<unsigned> dimPositions) {
  if (map.getNumDims() != numDims || map.getNumSymbols() != 0 ||
      map.getNumResults() != dimPositions.size())
    return false;

  MLIRContext *ctx = map.getContext();
  for (auto [result, dimPos] : llvm::zip_equal(map.getResults(), dimPositions))
    if (result != getAffineDimExpr(dimPos, ctx))
      return false;
  return true;
}

static bool isIdentity2D(AffineMap map) {
  return isAffineMap(map, 2, ArrayRef<unsigned>{0, 1});
}

static bool isIdentityND(AffineMap map, unsigned rank) {
  SmallVector<unsigned> dims;
  dims.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    dims.push_back(i);
  return isAffineMap(map, rank, dims);
}

static bool isGenericMatmulLike(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 2)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 3 || !isParallel(iterators[0]) ||
      !isParallel(iterators[1]) || !isReduction(iterators[2]))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !isAffineMap(maps[0], 3, ArrayRef<unsigned>{0, 2}) ||
      !isAffineMap(maps[1], 3, ArrayRef<unsigned>{2, 1}) ||
      !isAffineMap(maps[2], 3, ArrayRef<unsigned>{0, 1}))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *addOp = yieldOp.getValues()[0].getDefiningOp();
  if (!addOp)
    return false;

  StringRef addName = addOp->getName().getStringRef();
  if (addName != "arith.addi" && addName != "arith.addf")
    return false;

  Value lhs = addOp->getOperand(0);
  Value rhs = addOp->getOperand(1);
  Operation *mulOp = lhs.getDefiningOp();
  Value accumulator = rhs;
  if (!mulOp) {
    mulOp = rhs.getDefiningOp();
    accumulator = lhs;
  }
  if (!mulOp || accumulator != body.getArgument(2))
    return false;

  StringRef mulName = mulOp->getName().getStringRef();
  if (mulName != "arith.muli" && mulName != "arith.mulf")
    return false;

  return (mulOp->getOperand(0) == body.getArgument(0) &&
          mulOp->getOperand(1) == body.getArgument(1)) ||
         (mulOp->getOperand(0) == body.getArgument(1) &&
          mulOp->getOperand(1) == body.getArgument(0));
}

static bool matchElementwiseKind(linalg::GenericOp op, ElementwiseKind &kind) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 2)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !llvm::all_of(maps, isIdentity2D))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *opDef = yieldOp.getValues()[0].getDefiningOp();
  if (!opDef || opDef->getNumOperands() != 2)
    return false;

  bool directOrder = opDef->getOperand(0) == body.getArgument(0) &&
                     opDef->getOperand(1) == body.getArgument(1);
  bool swappedOrder = opDef->getOperand(0) == body.getArgument(1) &&
                      opDef->getOperand(1) == body.getArgument(0);
  if (!directOrder && !swappedOrder)
    return false;

  StringRef name = opDef->getName().getStringRef();
  if (name == "arith.addi")
    kind = ElementwiseKind::Add;
  else if (name == "arith.subi" && directOrder)
    kind = ElementwiseKind::Sub;
  else if (name == "arith.muli")
    kind = ElementwiseKind::Mul;
  else if (name == "arith.minsi")
    kind = ElementwiseKind::MinS;
  else if (name == "arith.minui")
    kind = ElementwiseKind::MinU;
  else if (name == "arith.maxsi")
    kind = ElementwiseKind::MaxS;
  else if (name == "arith.maxui")
    kind = ElementwiseKind::MaxU;
  else if (name == "arith.addf")
    kind = ElementwiseKind::FAdd;
  else if (name == "arith.subf" && directOrder)
    kind = ElementwiseKind::FSub;
  else if (name == "arith.mulf")
    kind = ElementwiseKind::FMul;
  else if (name == "arith.maximumf" || name == "arith.maxnumf")
    kind = ElementwiseKind::FMax;
  else
    return false;

  return true;
}

static bool matchUnarySquare(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() < 2 || !llvm::all_of(iterators, isParallel))
    return false;

  unsigned rank = iterators.size();
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isIdentityND(maps[0], rank) ||
      !isIdentityND(maps[1], rank))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 2)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *opDef = yieldOp.getValues()[0].getDefiningOp();
  if (!opDef || opDef->getName().getStringRef() != "math.fpowi" ||
      opDef->getNumOperands() != 2 ||
      opDef->getOperand(0) != body.getArgument(0))
    return false;

  APInt exponent;
  if (!matchPattern(opDef->getOperand(1), m_ConstantInt(&exponent)))
    return false;
  return exponent.getSExtValue() == 2;
}

static bool matchTranspose(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isAffineMap(maps[0], 2, ArrayRef<unsigned>{1, 0}) ||
      !isIdentity2D(maps[1]))
    return false;

  Block &body = op.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues()[0] == body.getArgument(0);
}

static bool matchBroadcast(linalg::GenericOp op, BroadcastKind &kind) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isIdentity2D(maps[1]))
    return false;

  AffineMap inputMap = maps[0];
  if (inputMap.getNumDims() != 2 || inputMap.getNumSymbols() != 0)
    return false;

  if (inputMap.getNumResults() == 0) {
    kind = BroadcastKind::Element;
  } else if (isAffineMap(inputMap, 2, ArrayRef<unsigned>{1})) {
    kind = BroadcastKind::Row;
  } else if (isAffineMap(inputMap, 2, ArrayRef<unsigned>{0})) {
    kind = BroadcastKind::Column;
  } else {
    return false;
  }

  Block &body = op.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues()[0] == body.getArgument(0);
}

static bool isLowerableGeneric(linalg::GenericOp op) {
  ElementwiseKind elementwiseKind;
  BroadcastKind broadcastKind;
  return isGenericMatmulLike(op) || matchElementwiseKind(op, elementwiseKind) ||
         matchUnarySquare(op) || matchTranspose(op) ||
         matchBroadcast(op, broadcastKind);
}

static Value createDim(PatternRewriter &rewriter, Location loc, Value memref,
                       int64_t dim) {
  return memref::DimOp::create(rewriter, loc, memref, dim);
}

static Value createIndexMin(PatternRewriter &rewriter, Location loc,
                            Value bound, Value iv, int64_t step) {
  Value remain = arith::SubIOp::create(rewriter, loc, bound, iv);
  Value stepVal = arith::ConstantIndexOp::create(rewriter, loc, step);
  Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                    remain, stepVal);
  return arith::SelectOp::create(rewriter, loc, cmp, remain, stepVal);
}

static Value castIndexToI64(PatternRewriter &rewriter, Location loc,
                            Value value) {
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    value);
}

static Value createByteStride(PatternRewriter &rewriter, Location loc,
                              Value memref, unsigned dim = 0) {
  auto memrefType = cast<MemRefType>(memref.getType());
  unsigned rank = memrefType.getRank();
  unsigned bytesPerElem = getElementByteWidth(memrefType.getElementType());
  Value bytes = arith::ConstantIndexOp::create(rewriter, loc, bytesPerElem);

  if (rank == 0)
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                      bytes);

  auto metadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
  Value strideElem = metadata.getResult(2 + rank + dim);
  Value strideBytes = arith::MulIOp::create(rewriter, loc, strideElem, bytes);
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    strideBytes);
}

static Value createSubView(PatternRewriter &rewriter, Location loc,
                           Value source, ArrayRef<Value> offsets,
                           ArrayRef<Value> sizes) {
  SmallVector<OpFoldResult> offsetResults;
  SmallVector<OpFoldResult> sizeResults;
  for (Value offset : offsets)
    offsetResults.push_back(offset);
  for (Value size : sizes)
    sizeResults.push_back(size);
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  return memref::SubViewOp::create(rewriter, loc, source, offsetResults,
                                   sizeResults, strides);
}

static LogicalResult createMSetTypeAndTiles(PatternRewriter &rewriter,
                                            Operation *anchor, Location loc,
                                            Type elementType, Value currM,
                                            Value currN, Value currK = {}) {
  int64_t msetTypeImm = 0;
  if (failed(getMsetTypeImm(elementType, msetTypeImm)))
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME element type");

  MSettypeiOp::create(rewriter, loc, rewriter.getI64Type(), msetTypeImm);
  MSettilemOp::create(rewriter, loc, rewriter.getI64Type(),
                      castIndexToI64(rewriter, loc, currM));
  MSettilenOp::create(rewriter, loc, rewriter.getI64Type(),
                      castIndexToI64(rewriter, loc, currN));
  if (currK)
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(),
                        castIndexToI64(rewriter, loc, currK));
  return success();
}

static FailureOr<Value> createLoadA(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlae8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlae16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlae32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlae64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME A load type");
}

static FailureOr<Value> createLoadB(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlbe8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlbe16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlbe32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlbe64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME B load type");
}

static FailureOr<Value> createLoadC(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlce8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlce16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlce32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlce64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME C load type");
}

static LogicalResult createStoreC(PatternRewriter &rewriter, Operation *anchor,
                                  Location loc, Type elementType, Value src,
                                  Value dest, Value stride) {
  if (elementType.isInteger(8))
    Msce8mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    Msce16mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(32) || elementType.isF32())
    Msce32mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(64) || elementType.isF64())
    Msce64mOp::create(rewriter, loc, src, dest, stride);
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME C store type");
  return success();
}

static FailureOr<Value> createMatmul(PatternRewriter &rewriter,
                                     Operation *anchor, Location loc,
                                     Type lhsType, Type resultType, Value acc,
                                     Value lhs, Value rhs) {
  Type tileType = VectorType::get({4, 4}, resultType);
  if (resultType.isInteger(32) && lhsType.isInteger(32))
    return MmaWmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(32) && lhsType.isInteger(16))
    return MwmaHmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(32) && lhsType.isInteger(8))
    return MqmaBmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(16) && lhsType.isInteger(16))
    return MmaHmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(64) && lhsType.isInteger(64))
    return MmaDwmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if ((lhsType.isF16() || lhsType.isBF16()) && resultType.isF32())
    return MfwmaHfmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF32() && resultType.isF32())
    return MfmaFmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF32() && resultType.isF64())
    return MfwmaFmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF64() && resultType.isF64())
    return MfmaDmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else
    return rewriter.notifyMatchFailure(
        anchor, "unsupported BOSCAME matmul instruction type");
}

static Value createMatrixOp(PatternRewriter &rewriter, Location loc,
                            StringRef name, Type resultType,
                            ValueRange operands) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultType);
  return rewriter.create(state)->getResult(0);
}

static FailureOr<Value> createElementwise(PatternRewriter &rewriter,
                                          Operation *anchor, Location loc,
                                          ElementwiseKind kind,
                                          Type elementType, Value lhs,
                                          Value rhs) {
  StringRef suffix;
  if (elementType.isInteger(8))
    suffix = ".b.mm";
  else if (elementType.isInteger(16))
    suffix = ".h.mm";
  else if (elementType.isInteger(32))
    suffix = ".w.mm";
  else if (elementType.isInteger(64))
    suffix = ".dw.mm";
  else if (elementType.isF16() || elementType.isBF16())
    suffix = ".hf.mm";
  else if (elementType.isF32())
    suffix = ".f.mm";
  else if (elementType.isF64())
    suffix = ".d.mm";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME elementwise type");

  StringRef mnemonic;
  switch (kind) {
  case ElementwiseKind::Add:
    mnemonic = "madd";
    break;
  case ElementwiseKind::Sub:
    mnemonic = "msub";
    break;
  case ElementwiseKind::Mul:
    mnemonic = "mmul";
    break;
  case ElementwiseKind::MinS:
    mnemonic = "mmin";
    break;
  case ElementwiseKind::MinU:
    mnemonic = "mminu";
    break;
  case ElementwiseKind::MaxS:
    mnemonic = "mmax";
    break;
  case ElementwiseKind::MaxU:
    mnemonic = "mmaxu";
    break;
  case ElementwiseKind::FAdd:
    mnemonic = "mfadd";
    break;
  case ElementwiseKind::FSub:
    mnemonic = "mfsub";
    break;
  case ElementwiseKind::FMul:
    mnemonic = "mfmul";
    break;
  case ElementwiseKind::FMax:
    mnemonic = "mfmax";
    break;
  }
  std::string name = "bosc_ame." + mnemonic.str() + suffix.str();
  return createMatrixOp(rewriter, loc, name, lhs.getType(), {lhs, rhs});
}

static FailureOr<Value> createBroadcast(PatternRewriter &rewriter,
                                        Operation *anchor, Location loc,
                                        BroadcastKind kind,
                                        BroadcastRegisterKind regKind,
                                        Type elementType, Value src) {
  StringRef registerSuffix = regKind == BroadcastRegisterKind::TileA   ? "a"
                             : regKind == BroadcastRegisterKind::TileB ? "b"
                                                                       : "c";
  if (kind == BroadcastKind::Row) {
    std::string name = "bosc_ame.mbc" + registerSuffix.str() + "r.m";
    return createMatrixOp(rewriter, loc, name, src.getType(), {src});
  }

  StringRef bitWidth;
  if (elementType.isInteger(8))
    bitWidth = "8";
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    bitWidth = "16";
  else if (elementType.isInteger(32) || elementType.isF32())
    bitWidth = "32";
  else if (elementType.isInteger(64) || elementType.isF64())
    bitWidth = "64";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME broadcast type");

  std::string name = "bosc_ame.mbc" + registerSuffix.str() +
                     (kind == BroadcastKind::Column ? "ce" : "ee") +
                     bitWidth.str() + ".m";
  return createMatrixOp(rewriter, loc, name, src.getType(), {src});
}

static FailureOr<Value> createTranspose(PatternRewriter &rewriter,
                                        Operation *anchor, Location loc,
                                        BroadcastRegisterKind regKind,
                                        Type elementType, Value src) {
  StringRef registerSuffix = regKind == BroadcastRegisterKind::TileA   ? "a"
                             : regKind == BroadcastRegisterKind::TileB ? "b"
                                                                       : "c";
  StringRef bitWidth;
  if (elementType.isInteger(8))
    bitWidth = "8";
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    bitWidth = "16";
  else if (elementType.isInteger(32) || elementType.isF32())
    bitWidth = "32";
  else if (elementType.isInteger(64) || elementType.isF64())
    bitWidth = "64";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME transpose type");
  std::string name =
      "bosc_ame.mt" + registerSuffix.str() + "e" + bitWidth.str() + ".m";
  return createMatrixOp(rewriter, loc, name, src.getType(), {src});
}

static LogicalResult lowerMatmulLike(Operation *anchor,
                                     PatternRewriter &rewriter, Location loc,
                                     Value A, Value B, Value C) {
  auto AType = dyn_cast<MemRefType>(A.getType());
  auto BType = dyn_cast<MemRefType>(B.getType());
  auto CType = dyn_cast<MemRefType>(C.getType());
  if (!AType || !BType || !CType)
    return rewriter.notifyMatchFailure(anchor, "expected memref operands");

  Type elemTypeA = AType.getElementType();
  Type elemTypeB = BType.getElementType();
  Type elemTypeC = CType.getElementType();
  if (elemTypeA != elemTypeB)
    return rewriter.notifyMatchFailure(
        anchor, "matmul lhs and rhs element types differ");

  int64_t tileM = 4, tileN = 4, tileK = 4;
  if (elemTypeA.isInteger(8) && elemTypeC.isInteger(32))
    tileK = 16;
  else if ((elemTypeA.isF16() || elemTypeA.isBF16()) && elemTypeC.isF32())
    tileK = 8;
  else if (elemTypeA.isInteger(16) && elemTypeC.isInteger(32))
    tileK = 8;
  else if (elemTypeA.isInteger(32) && elemTypeC.isInteger(32))
    tileK = 4;
  else if (elemTypeA.isF32() && (elemTypeC.isF32() || elemTypeC.isF64()))
    tileK = 4;
  else if (elemTypeA.isInteger(64) && elemTypeC.isInteger(64))
    tileK = 2;
  else if (elemTypeA.isF64() && elemTypeC.isF64())
    tileK = 2;
  else
    return rewriter.notifyMatchFailure(anchor, "unsupported matmul precision");

  Value dimM = createDim(rewriter, loc, A, 0);
  Value dimK = createDim(rewriter, loc, A, 1);
  Value dimN = createDim(rewriter, loc, B, 1);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
  Value stepK = arith::ConstantIndexOp::create(rewriter, loc, tileK);
  Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

  auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
  rewriter.setInsertionPointToStart(loopM.getBody());
  Value ivM = loopM.getInductionVar();

  auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
  rewriter.setInsertionPointToStart(loopN.getBody());
  Value ivN = loopN.getInductionVar();

  auto loopK = scf::ForOp::create(rewriter, loc, c0, dimK, stepK);
  rewriter.setInsertionPointToStart(loopK.getBody());
  Value ivK = loopK.getInductionVar();

  Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
  Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);
  Value currK = createIndexMin(rewriter, loc, dimK, ivK, tileK);

  Value subA = createSubView(rewriter, loc, A, {ivM, ivK}, {currM, currK});
  Value subB = createSubView(rewriter, loc, B, {ivK, ivN}, {currK, currN});
  Value subC = createSubView(rewriter, loc, C, {ivM, ivN}, {currM, currN});

  Value strideA = createByteStride(rewriter, loc, subA);
  Value strideB = createByteStride(rewriter, loc, subB);
  Value strideC = createByteStride(rewriter, loc, subC);

  if (failed(createMSetTypeAndTiles(rewriter, anchor, loc, elemTypeA, currM,
                                    currN, currK)))
    return failure();
  FailureOr<Value> tileA =
      createLoadA(rewriter, anchor, loc, elemTypeA, subA, strideA);
  FailureOr<Value> tileB =
      createLoadB(rewriter, anchor, loc, elemTypeB, subB, strideB);
  FailureOr<Value> acc =
      createLoadC(rewriter, anchor, loc, elemTypeC, subC, strideC);
  if (failed(tileA) || failed(tileB) || failed(acc))
    return failure();
  FailureOr<Value> result = createMatmul(rewriter, anchor, loc, elemTypeA,
                                         elemTypeC, *acc, *tileA, *tileB);
  if (failed(result) || failed(createStoreC(rewriter, anchor, loc, elemTypeC,
                                            *result, subC, strideC)))
    return failure();

  rewriter.setInsertionPointAfter(loopM);
  return success();
}

/// Lower remaining ordinary buffer Linalg to scalar memory/arithmetic
/// loops.  The independent backend deliberately keeps this path free of AME
/// operations; a later RVV/vector pass can recognize the regular loops.  The
/// legacy pass does not install this pattern and therefore retains its old
/// matrix instruction behaviour byte for byte.
static LogicalResult lowerOrdinaryLinalg(linalg::LinalgOp op,
                                         PatternRewriter &rewriter,
                                         bool preserveGraph = false) {
  if (!op.hasPureBufferSemantics() ||
      op->hasAttr(::buddy::boscame::lowering::kLowerToAme))
    return failure();
  Operation *previous = op->getPrevNode();
  if (failed(linalg::linalgOpToLoops(rewriter, op)))
    return failure();
  if (preserveGraph) {
    Operation *completion = op->getPrevNode();
    if (!completion || completion == previous)
      return op.emitError(
          "BOSCAME task lowering produced no completion operation");
    for (NamedAttribute attribute : op->getAttrs())
      if (attribute.getName().getValue().starts_with("bosc_ame."))
        completion->setAttr(attribute.getName(), attribute.getValue());
  }
  rewriter.eraseOp(op);
  return success();
}

class OrdinaryLinalgToLoops
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return lowerOrdinaryLinalg(op, rewriter);
  }
};

class MatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics())
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();

    if (failed(lowerMatmulLike(op, rewriter, op.getLoc(), A, B, C)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class GenericMatmulToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isGenericMatmulLike(op))
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();

    if (failed(lowerMatmulLike(op, rewriter, op.getLoc(), A, B, C)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class GenericElementwiseToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    ElementwiseKind kind;
    if (!matchElementwiseKind(op, kind))
      return failure();

    Location loc = op.getLoc();
    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    Type elementType = outType.getElementType();
    if (lhsType.getElementType() != rhsType.getElementType() ||
        lhsType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op,
                                         "elementwise operand types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value subLhs =
        createSubView(rewriter, loc, lhs, {ivM, ivN}, {currM, currN});
    Value subRhs =
        createSubView(rewriter, loc, rhs, {ivM, ivN}, {currM, currN});
    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});

    Value strideLhs = createByteStride(rewriter, loc, subLhs);
    Value strideRhs = createByteStride(rewriter, loc, subRhs);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> lhsTile =
        createLoadC(rewriter, op, loc, elementType, subLhs, strideLhs);
    FailureOr<Value> rhsTile =
        createLoadC(rewriter, op, loc, elementType, subRhs, strideRhs);
    if (failed(lhsTile) || failed(rhsTile))
      return failure();
    FailureOr<Value> result = createElementwise(
        rewriter, op, loc, kind, elementType, *lhsTile, *rhsTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericUnarySquareToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchUnarySquare(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || inputType.getRank() != outType.getRank() ||
        inputType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "expected same-rank memrefs with rank >= 2");

    if (inputType.getElementType() != outType.getElementType())
      return rewriter.notifyMatchFailure(op, "square operand types differ");

    Type elementType = outType.getElementType();
    if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32() &&
        !elementType.isF64())
      return rewriter.notifyMatchFailure(
          op, "math.fpowi square expects floating point element type");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;
    unsigned rank = outType.getRank();
    unsigned matrixDimM = rank - 2;
    unsigned matrixDimN = rank - 1;

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    SmallVector<Value> outerIvs;
    SmallVector<scf::ForOp> outerLoops;
    for (unsigned dim = 0; dim < matrixDimM; ++dim) {
      Value dimSize = createDim(rewriter, loc, out, dim);
      auto loop = scf::ForOp::create(rewriter, loc, c0, dimSize, c1);
      outerLoops.push_back(loop);
      rewriter.setInsertionPointToStart(loop.getBody());
      outerIvs.push_back(loop.getInductionVar());
    }

    Value dimM = createDim(rewriter, loc, out, matrixDimM);
    Value dimN = createDim(rewriter, loc, out, matrixDimN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    SmallVector<Value> offsets(outerIvs.begin(), outerIvs.end());
    offsets.push_back(ivM);
    offsets.push_back(ivN);

    SmallVector<Value> sizes(outerIvs.size(), c1);
    sizes.push_back(currM);
    sizes.push_back(currN);

    Value subInput = createSubView(rewriter, loc, input, offsets, sizes);
    Value subOut = createSubView(rewriter, loc, out, offsets, sizes);

    Value strideInput = createByteStride(rewriter, loc, subInput, matrixDimM);
    Value strideOut = createByteStride(rewriter, loc, subOut, matrixDimM);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile =
        createLoadC(rewriter, op, loc, elementType, subInput, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result =
        createElementwise(rewriter, op, loc, ElementwiseKind::FMul, elementType,
                          *inputTile, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    if (!outerLoops.empty())
      rewriter.setInsertionPointAfter(outerLoops.front());
    else
      rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericTransposeToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchTranspose(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || inputType.getRank() != 2 ||
        outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    Type elementType = outType.getElementType();
    if (inputType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "transpose element types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;
    for (int64_t dim : outType.getShape()) {
      if (ShapedType::isDynamic(dim) || dim % tileM != 0)
        return rewriter.notifyMatchFailure(
            op, "BOSCAME transpose requires full 4x4 square tiles");
    }

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value subInput =
        createSubView(rewriter, loc, input, {ivN, ivM}, {currN, currM});
    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});

    Value strideInput = createByteStride(rewriter, loc, subInput);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile =
        createLoadC(rewriter, op, loc, elementType, subInput, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result =
        createTranspose(rewriter, op, loc, BroadcastRegisterKind::Accumulation,
                        elementType, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericBroadcastToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    BroadcastKind kind;
    if (!matchBroadcast(op, kind))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected memref operands");

    Type elementType = outType.getElementType();
    if (inputType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "broadcast element types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value broadcastSource = input;
    if (kind == BroadcastKind::Row) {
      if (inputType.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "row broadcast expects rank-1 input");
      broadcastSource = createSubView(rewriter, loc, input, {ivN}, {currN});
    } else if (kind == BroadcastKind::Column) {
      if (inputType.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "column broadcast expects rank-1 input");
      broadcastSource = createSubView(rewriter, loc, input, {ivM}, {currM});
    } else if (inputType.getRank() != 0) {
      return rewriter.notifyMatchFailure(
          op, "element broadcast expects rank-0 input");
    }

    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});
    Value strideInput = createByteStride(rewriter, loc, broadcastSource);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile = createLoadC(rewriter, op, loc, elementType,
                                             broadcastSource, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result = createBroadcast(
        rewriter, op, loc, kind, BroadcastRegisterKind::Accumulation,
        elementType, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
static LogicalResult verifySpmDescriptor(Operation *anchor, Value value) {
  using namespace ::buddy::boscame::spmdma;
  if (!getSpmBank(value.getType()))
    return success();
  auto type = dyn_cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t ignored;
  auto offset = getStaticDescriptorOffset(value);
  if (!type || !type.hasStaticShape() || !offset || *offset < 0 ||
      failed(type.getStridesAndOffset(strides, ignored)))
    return anchor->emitError("BOSCAME SPM view requires static shape and "
                             "a known nonnegative descriptor offset");
  int64_t span = 1;
  for (auto [dimension, stride] : llvm::zip(type.getShape(), strides)) {
    int64_t extent;
    if (dimension <= 0 || stride <= 0 ||
        llvm::MulOverflow(dimension - 1, stride, extent) ||
        llvm::AddOverflow(span, extent, span))
      return anchor->emitError("BOSCAME SPM view has an invalid physical span");
  }
  int64_t end;
  unsigned width = getSpmElementByteWidth(type.getElementType());
  if (!width || llvm::AddOverflow(*offset, span, end) ||
      llvm::MulOverflow(end, int64_t{width}, end))
    return anchor->emitError("BOSCAME SPM view physical span overflows");
  Value root = value;
  while (Operation *definition = root.getDefiningOp()) {
    if (auto op = dyn_cast<memref::ReinterpretCastOp>(definition))
      root = op.getSource();
    else if (auto op = dyn_cast<memref::SubViewOp>(definition))
      root = op.getSource();
    else if (auto op = dyn_cast<memref::TransposeOp>(definition))
      root = op.getIn();
    else if (auto op = dyn_cast<memref::CastOp>(definition))
      root = op.getSource();
    else if (auto op = dyn_cast<memref::MemorySpaceCastOp>(definition))
      root = op.getSource();
    else if (auto op = dyn_cast<memref::CollapseShapeOp>(definition))
      root = op.getSrc();
    else if (auto op = dyn_cast<memref::ExpandShapeOp>(definition))
      root = op.getSrc();
    else if (auto op = dyn_cast<memref::ViewOp>(definition)) {
      auto shift = getConstantIntValue(op.getByteShift());
      if (!shift || *shift < 0 || llvm::AddOverflow(end, *shift, end))
        return anchor->emitError("BOSCAME SPM view needs a known nonnegative "
                                 "byte shift");
      root = op.getSource();
    } else
      break;
  }
  auto view = root.getDefiningOp<SpmViewOp>();
  auto bytes =
      view ? getContiguousZeroOffsetByteSize(cast<MemRefType>(view.getType()))
           : std::nullopt;
  if (!bytes || end > *bytes)
    return anchor->emitError("BOSCAME SPM view leaves its spm.view allocation");
  return success();
}

// Run before rewriting as well as at the exit: dead unsupported operations
// must not disappear during canonicalization and silently satisfy the contract.
static LogicalResult verifyStandaloneSurface(ModuleOp module,
                                             bool verifyBounds = true) {
  WalkResult result = module.walk([&](Operation *op) {
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "tt" || dialect == "tts" || dialect == "bufferization") {
      op->emitError("incomplete BOSCAME lowering: residual Triton or "
                    "bufferization bridge operation");
      return WalkResult::interrupt();
    }
    if (isa<vector::GatherOp, vector::ScatterOp>(op)) {
      op->emitError("unsupported BOSCAME gather/scatter memory access");
      return WalkResult::interrupt();
    }
    if (auto global = dyn_cast<memref::GlobalOp>(op)) {
      if (::buddy::boscame::spmdma::getSpmBank(global.getType())) {
        op->emitError("BOSCAME SPM globals are unsupported; use explicit "
                      "bosc_ame.spm.view windows");
        return WalkResult::interrupt();
      }
    }
    if (!verifyBounds)
      return WalkResult::advance();
    for (Value value : op->getOperands())
      if (failed(verifySpmDescriptor(op, value)))
        return WalkResult::interrupt();
    for (Value value : op->getResults())
      if (failed(verifySpmDescriptor(op, value)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

} // namespace

namespace buddy::boscame {
LogicalResult verifyBOSCAMEPlanningSurface(ModuleOp module) {
  return verifyStandaloneSurface(module);
}
} // namespace buddy::boscame

namespace {

static LogicalResult promoteStaticTemporaries(ModuleOp module) {
  SmallVector<memref::AllocOp> allocations;
  module.walk([&](memref::AllocOp alloc) {
    if (alloc.getType().hasStaticShape() &&
        !::buddy::boscame::spmdma::getSpmBank(alloc.getType()))
      allocations.push_back(alloc);
  });
  for (memref::AllocOp alloc : allocations) {
    MemRefType type = alloc.getType();
    if (type.getLayout().isIdentity())
      continue;
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(strides, offset)) || offset < 0)
      return alloc.emitError("BOSCAME temporary requires static positive "
                             "physical strides and offset");
    int64_t span = offset;
    bool empty = false;
    for (auto [dimension, stride] : llvm::zip(type.getShape(), strides)) {
      int64_t extent;
      if (stride <= 0 ||
          llvm::MulOverflow(std::max<int64_t>(dimension - 1, 0), stride,
                            extent) ||
          llvm::AddOverflow(span, extent, span))
        return alloc.emitError("BOSCAME temporary physical span overflows");
      empty |= dimension == 0;
    }
    if (!empty && llvm::AddOverflow(span, int64_t{1}, span))
      return alloc.emitError("BOSCAME temporary physical span overflows");
    if (empty)
      span = 0;
    OpBuilder builder(alloc);
    auto flatType = MemRefType::get({span}, type.getElementType());
    auto flat =
        memref::AllocOp::create(builder, alloc.getLoc(), flatType, ValueRange{},
                                builder.getI64IntegerAttr(std::max<int64_t>(
                                    64, alloc.getAlignment().value_or(0))));
    SmallVector<OpFoldResult> sizes, physicalStrides;
    for (int64_t size : type.getShape())
      sizes.push_back(builder.getIndexAttr(size));
    for (int64_t stride : strides)
      physicalStrides.push_back(builder.getIndexAttr(stride));
    Value view = memref::ReinterpretCastOp::create(
        builder, alloc.getLoc(), type, flat, builder.getIndexAttr(offset),
        sizes, physicalStrides);
    SmallVector<memref::DeallocOp> deallocations;
    for (Operation *user : alloc->getUsers())
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
        deallocations.push_back(dealloc);
    alloc.replaceAllUsesWith(view);
    for (memref::DeallocOp dealloc : deallocations)
      dealloc.getMemrefMutable().assign(flat);
    alloc.erase();
  }
  PassManager hoisting(module.getContext());
  hoisting.addNestedPass<func::FuncOp>(
      bufferization::createBufferLoopHoistingPass());
  if (failed(hoisting.run(module)))
    return failure();
  // The generic promotion pass deliberately excludes allocations with an
  // explicit dealloc. At function scope their lifetime can safely extend to
  // the return; convert those temporaries and remove the matching deallocs.
  allocations.clear();
  module.walk([&](memref::AllocOp alloc) {
    if (alloc.getType().hasStaticShape() &&
        isa<func::FuncOp>(alloc->getParentOp()) &&
        !::buddy::boscame::spmdma::getSpmBank(alloc.getType()))
      allocations.push_back(alloc);
  });
  for (memref::AllocOp alloc : allocations) {
    SmallVector<memref::DeallocOp> deallocations;
    for (Operation *user : alloc->getUsers())
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
        deallocations.push_back(dealloc);
    if (deallocations.empty())
      continue;
    OpBuilder builder(alloc);
    auto stack =
        memref::AllocaOp::create(builder, alloc.getLoc(), alloc.getType(),
                                 ValueRange{}, alloc.getAlignmentAttr());
    for (memref::DeallocOp dealloc : deallocations)
      dealloc.erase();
    alloc.replaceAllUsesWith(stack.getResult());
    alloc.erase();
  }
  PassManager manager(module.getContext());
  manager.addNestedPass<func::FuncOp>(
      bufferization::createPromoteBuffersToStackPass([](Value value) {
        return cast<MemRefType>(value.getType()).hasStaticShape();
      }));
  return manager.run(module);
}

static StringRef getTransferRoleName(TransferRole role) {
  switch (role) {
  case TransferRole::InputA:
    return "input-a";
  case TransferRole::InputB:
    return "input-b";
  case TransferRole::OutputC:
    return "output-c";
  case TransferRole::Temporary:
    return "temporary";
  }
  llvm_unreachable("unknown BOSCAME transfer role");
}

static void printPlanShape(llvm::raw_ostream &stream,
                           ArrayRef<int64_t> values) {
  stream << '[';
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index)
      stream << ',';
    stream << value;
  }
  stream << ']';
}

static StringRef getComputeGraphStatusName(ComputeGraphStatus status) {
  switch (status) {
  case ComputeGraphStatus::Unavailable:
    return "unavailable";
  case ComputeGraphStatus::Incomplete:
    return "incomplete";
  case ComputeGraphStatus::Complete:
    return "complete";
  }
  llvm_unreachable("unknown compute graph status");
}

static StringRef getBufferAccessEffectName(BufferAccessEffect effect) {
  switch (effect) {
  case BufferAccessEffect::Read:
    return "read";
  case BufferAccessEffect::Write:
    return "write";
  case BufferAccessEffect::Allocate:
    return "allocate";
  case BufferAccessEffect::Free:
    return "free";
  case BufferAccessEffect::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown buffer access effect");
}

static StringRef getBufferAliasRelationName(BufferAliasRelation relation) {
  switch (relation) {
  case BufferAliasRelation::MayAlias:
    return "may-alias";
  case BufferAliasRelation::Disjoint:
    return "disjoint";
  }
  llvm_unreachable("unknown buffer alias relation");
}

static StringRef getBufferAliasProofName(BufferAliasProof proof) {
  switch (proof) {
  case BufferAliasProof::SameRoot:
    return "same-root";
  case BufferAliasProof::SameRootByteIntervals:
    return "same-root-byte-intervals";
  case BufferAliasProof::SameRootSymbolicByteIntervals:
    return "same-root-symbolic-byte-intervals";
  case BufferAliasProof::SameRootPitchedRowIntervals:
    return "same-root-pitched-row-intervals";
  case BufferAliasProof::AbiNonAliasingEntryArguments:
    return "abi-nonalias-entry-arguments";
  case BufferAliasProof::DistinctGlobalSymbols:
    return "distinct-global-symbols";
  case BufferAliasProof::FreshAllocation:
    return "fresh-allocation";
  }
  llvm_unreachable("unknown buffer alias proof");
}

static StringRef getMemoryDependencyKindName(MemoryDependencyKind kind) {
  switch (kind) {
  case MemoryDependencyKind::RAW:
    return "RAW";
  case MemoryDependencyKind::WAR:
    return "WAR";
  case MemoryDependencyKind::WAW:
    return "WAW";
  case MemoryDependencyKind::UnknownEffect:
    return "unknown-effect";
  case MemoryDependencyKind::Lifetime:
    return "lifetime";
  }
  llvm_unreachable("unknown memory dependency kind");
}

struct LaunchGridAxisArg {
  int64_t axis;
  unsigned argument;
};

static void printLaunchGrid(llvm::ArrayRef<LaunchFunctionFact> launches,
                            llvm::raw_ostream &stream) {
  for (const LaunchFunctionFact &function : launches) {
    const auto &knownGrid = function.knownGrid;
    SmallVector<LaunchGridAxisArg, 3> axes;
    SmallVector<unsigned, 2> missingAxisArguments;
    for (const LaunchProgramArgumentFact &argument :
         function.programArguments) {
      if (argument.role != LaunchProgramRole::NumPrograms)
        continue;
      if (!argument.axis) {
        missingAxisArguments.push_back(argument.argumentIndex);
        continue;
      }
      axes.push_back({*argument.axis, argument.argumentIndex});
    }
    llvm::sort(axes,
               [](const LaunchGridAxisArg &lhs, const LaunchGridAxisArg &rhs) {
                 return lhs.axis != rhs.axis ? lhs.axis < rhs.axis
                                             : lhs.argument < rhs.argument;
               });

    stream << "  launch-grid: function=@" << function.functionName;
    if (axes.empty() && missingAxisArguments.empty() && !knownGrid) {
      stream << " unavailable\n";
      continue;
    }
    for (const LaunchGridAxisArg &entry : axes) {
      stream << " axis[" << entry.axis << "]=runtime(arg#" << entry.argument
             << ") extent=";
      if (knownGrid && entry.axis >= 0 && entry.axis < 3)
        stream << (*knownGrid)[entry.axis];
      else
        stream << "<unknown>";
    }
    for (unsigned argument : missingAxisArguments)
      stream << " axis=<unknown>:runtime(arg#" << argument
             << ") extent=<unknown>";
    if (knownGrid && axes.empty()) {
      stream << " known-grid=[";
      for (auto [index, extent] : llvm::enumerate(*knownGrid)) {
        if (index)
          stream << ',';
        stream << extent;
      }
      stream << ']';
    }
    stream << '\n';
  }
}

static LogicalResult verifyKnownLaunchGrid(ModuleOp module) {
  return failed(analyzeBOSCAMELaunchFacts(module)) ? failure() : success();
}

static const TransferPlan *findPlannedTransfer(const WorkspacePlan &workspace,
                                               Value buffer) {
  auto transfer =
      llvm::find_if(workspace.transfers, [&](const TransferPlan &candidate) {
        return candidate.buffer == buffer;
      });
  return transfer == workspace.transfers.end() ? nullptr : &*transfer;
}

static StringRef getDirectDecision(bool isDirect,
                                   const TransferPlan *transfer) {
  if (isDirect)
    return "direct-ddr (proven)";
  if (transfer)
    return "spm-dma (planned)";
  return "not inferred";
}

static StringRef getResidencyDecisionReason(ResidencyDecisionReason reason) {
  switch (reason) {
  case ResidencyDecisionReason::Selected:
    return "traffic-benefit-and-capacity";
  case ResidencyDecisionReason::Capacity:
    return "spm-capacity";
  case ResidencyDecisionReason::NoTrafficBenefit:
    return "no-traffic-benefit";
  }
  llvm_unreachable("unknown residency decision reason");
}

static void printBOSCAMEPlanReport(ModuleOp module,
                                   const LinalgToBOSCAMEConfig &config,
                                   const DataflowPlan &workspace) {
  llvm::raw_ostream &stream = llvm::errs();
  stream << "BOSCAME plan report:\n"
         << "  target="
         << (config.target.empty() ? "<unspecified>" : config.target)
         << " tile=[" << config.bm << ',' << config.bn << ',' << config.bk
         << "] routing=automatic bank-capacity=" << workspace.bankCapacity
         << " bytes\n";

  const TargetInfo &target = workspace.target;
  auto printAmeRole = [&](StringRef name, const AmeRoleCapability &role) {
    stream << name << "={bits=" << role.elementBits
           << ",layout=" << stringifyAmeTileLayout(role.layout)
           << ",spm=" << (role.supportsSpm ? "true" : "false")
           << ",direct-ddr=" << (role.supportsDirectDdr ? "true" : "false")
           << '}';
  };
  stream << "  target-capabilities: name=" << target.name << " dma={ddr-to-spm="
         << (target.dma.supports(StorageKind::DDR, StorageKind::SPM) ? "true"
                                                                     : "false")
         << ",spm-to-ddr="
         << (target.dma.supports(StorageKind::SPM, StorageKind::DDR) ? "true"
                                                                     : "false")
         << ",spm-to-spm="
         << (target.dma.supports(StorageKind::SPM, StorageKind::SPM) ? "true"
                                                                     : "false")
         << ",endpoint-alignment=" << target.dma.endpointAlignmentBytes
         << ",packet-bytes=" << target.dma.packetGranularityBytes
         << ",max-rank=" << target.dma.maxStaticRank << ",pitched-rows="
         << (target.dma.packetizesPitchedRows ? "packetized" : "unsupported")
         << "} banks=" << target.spmBankCount
         << " channels=" << target.dmaChannelCount << " ame={signed-multiply="
         << (target.ame.signedIntegerMultiply ? "true" : "false")
         << ",tile-mnk=" << target.ame.maxTileM << 'x' << target.ame.maxTileN
         << 'x' << target.ame.maxTileK << ',';
  printAmeRole("a", target.ame.a);
  stream << ',';
  printAmeRole("b", target.ame.b);
  stream << ',';
  printAmeRole("accumulator", target.ame.accumulator);
  stream << "} rvv-vector-bytes=";
  if (target.rvvVectorBytes)
    stream << *target.rvvVectorBytes;
  else
    stream << "<unknown>";
  stream << '\n';

  printLaunchGrid(workspace.launches, stream);

  stream << "  linalg-facts: scope=post-planning\n";
  printLinalgComputationFacts(workspace.linalg, stream, "    ");

  if (config.bufferPlacement && !workspace.bufferPlacements.empty()) {
    auto kindName = [](BufferPlacementKind kind) -> StringRef {
      switch (kind) {
      case BufferPlacementKind::Spare:
        return "Spare";
      case BufferPlacementKind::SpmResident:
        return "SpmResident";
      case BufferPlacementKind::StagedFallback:
        return "StagedFallback";
      }
      llvm_unreachable("unknown placement kind");
    };
    auto reasonName = [](BufferPlacementReason reason) -> StringRef {
      switch (reason) {
      case BufferPlacementReason::Selected:
        return "group-legality-and-capacity";
      case BufferPlacementReason::AmeDirectRoute:
        return "ame-direct-route";
      case BufferPlacementReason::Auxiliary:
        return "predicate-auxiliary";
      case BufferPlacementReason::Unused:
        return "no-computation-use";
      case BufferPlacementReason::UnsupportedLayout:
        return "unsupported-layout";
      case BufferPlacementReason::UnsupportedUse:
        return "unsupported-use";
      case BufferPlacementReason::ControlFlow:
        return "control-flow-or-call";
      case BufferPlacementReason::AliasUnknown:
        return "alias-unknown";
      case BufferPlacementReason::MissingStaging:
        return "missing-staging-capacity";
      case BufferPlacementReason::Capacity:
        return "spm-capacity";
      case BufferPlacementReason::ComputationLegality:
        return "computation-legality";
      }
      llvm_unreachable("unknown placement reason");
    };
    size_t resident = 0, fallback = 0;
    bool applied = false;
    for (const BufferPlacementDecision &decision : workspace.bufferPlacements) {
      resident += decision.kind == BufferPlacementKind::SpmResident;
      fallback += decision.kind == BufferPlacementKind::StagedFallback;
      applied |= decision.applied;
    }
    stream << "  buffer-placement: resident=" << resident << " ddr=" << fallback
           << " applied=" << (applied ? "true" : "false")
           << " scope=serial-scf fixed-windows=true interval-reuse=false"
              " exclusive-branch-reuse=true\n";
    for (const BufferPlacementDecision &decision : workspace.bufferPlacements) {
      const RootStorageFact &root =
          workspace.rootStorageFacts[decision.rootIndex];
      stream << "  root[" << decision.rootIndex
             << "]: kind=" << kindName(decision.kind)
             << " bytes=" << root.capacityBytes << " bank=" << decision.bank
             << " offset=" << decision.offset << " group=" << decision.group
             << " reason=" << reasonName(decision.reason)
             << " applied=" << (decision.applied ? "true" : "false")
             << " lexical-range=" << root.firstPosition << ".."
             << root.lastPosition << " potential-read-savings-bytes="
             << root.potentialReadSavingsBytes << " structured-scope="
             << (root.structuredScope ? "true" : "false");
      if (!decision.detail.empty())
        stream << " detail=" << decision.detail;
      stream << '\n';
    }
    for (auto [index, version] : llvm::enumerate(workspace.contentVersions)) {
      stream << "  content-version[" << index << "]: root=" << version.rootIndex
             << " writer=" << version.writer->getName()
             << " position=" << version.position << " validShape=";
      printPlanShape(stream, version.validShape);
      stream << " padding="
             << (version.paddingInitialization ? "true" : "false")
             << " rematerializable="
             << (version.rematerializable ? "true" : "false") << " consumers=";
      if (workspace.rootStorageFacts[version.rootIndex].structuredScope)
        stream << "unknown";
      else
        stream << version.consumers.size();
      stream << '\n';
    }
  }

  for (auto [index, transfer] : llvm::enumerate(workspace.transfers)) {
    stream << "  transfer[" << index << "]: consumer=";
    if (transfer.firstConsumer)
      stream << transfer.firstConsumer->getName().getStringRef();
    else
      stream << "<unknown>";
    stream << " residency="
           << (transfer.crossTaskResident ? "cross-task" : "operation-local")
           << " consumers=[";
    llvm::interleaveComma(transfer.consumerOperations, stream,
                          [&](Operation *consumer) {
                            stream << consumer->getName().getStringRef();
                          });
    stream << "]";
    if (transfer.sourceOriginId)
      stream << " source-origin-id=" << *transfer.sourceOriginId;
    stream << " role=" << getTransferRoleName(transfer.role) << " shape=";
    printPlanShape(stream, transfer.shape);
    stream << " strides=";
    printPlanShape(stream, transfer.strides);
    stream << " validShape=";
    printPlanShape(stream, transfer.validShape);
    stream << " bank=" << transfer.spmBank << " offset=" << transfer.spmOffset
           << " capacity=" << transfer.spmCapacity
           << " effective-byte-offset-bytes=";
    if (transfer.effectiveByteOffsetBytes)
      stream << *transfer.effectiveByteOffsetBytes;
    else
      stream << "<unknown>";
    stream << " endpoint-alignment-proven="
           << (transfer.endpointAlignmentProven ? "true" : "false")
           << " packets=" << transfer.packets.size();
    if (transfer.guardedPrefix)
      stream << " guarded-prefix=true packet-count=upper-bound extent="
             << transfer.validElements;
    if (transfer.directExternalBoundary) {
      stream << " boundary=external source=" << transfer.source
             << " target=" << transfer.target
             << " provider=" << transfer.provider->getName()
             << " completion=wait-at-provider offsets-and-bytes=[";
      llvm::interleaveComma(
          transfer.packets, stream, [&](const DmaPacket &packet) {
            stream << '{' << packet.ddrOffsetBytes << ','
                   << packet.spmOffsetBytes << ',' << packet.bytes << '}';
          });
      stream << ']';
    }
    stream << " read=" << (transfer.read ? "true" : "false")
           << " write=" << (transfer.write ? "true" : "false") << '\n';
  }

  stream << "  compute-graph: status="
         << getComputeGraphStatusName(workspace.computeGraphStatus)
         << " tasks=" << workspace.tasks.size()
         << " dependencies=" << workspace.dependencies.size()
         << " logical-values=" << workspace.logicalValues.size() << '\n';
  for (const PlannedTask &task : workspace.tasks) {
    stream << "  task[" << task.id << "]: op=";
    if (task.operation)
      stream << task.operation->getName().getStringRef();
    else
      stream << "<unknown>";
    stream << " kind=";
    if (task.kind)
      stream << task.kind.getValue();
    else
      stream << "<unknown>";
    stream << " stage=" << task.stage << " predecessors=";
    printPlanShape(stream, task.predecessors);
    stream << " inputs=";
    printPlanShape(stream, task.inputValues);
    stream << " outputs=";
    printPlanShape(stream, task.outputValues);
    stream << " varying-axes=";
    printPlanShape(stream, task.varyingProgramAxes);
    stream << " barrier-after=" << (task.barrierAfter ? "true" : "false")
           << " rematerializable=" << (task.rematerializable ? "true" : "false")
           << '\n';
  }
  for (const TaskDependency &dependency : workspace.dependencies)
    stream << "  dependency: " << dependency.predecessor << " -> "
           << dependency.consumer
           << " resolved=" << (dependency.resolved ? "true" : "false") << '\n';
  for (const LogicalValuePlan &value : workspace.logicalValues) {
    stream << "  logical-value[" << value.id << "]: producer=";
    if (value.producerTask)
      stream << *value.producerTask;
    else if (value.producerIsFunctionArgument)
      stream << "function-argument";
    else
      stream << "<unknown>";
    stream << " consumers=";
    printPlanShape(stream, value.consumerTasks);
    stream << " lifetime=" << (value.lifetimeComplete ? "known" : "unknown")
           << '\n';
  }
  for (const BufferRequirement &buffer : workspace.buffers) {
    stream << "  buffer[" << buffer.transferIndex
           << "]: root=known root-id=" << buffer.storageRootId << " views=[";
    llvm::interleaveComma(buffer.viewChain, stream, [&](Operation *view) {
      stream << view->getName().getStringRef();
    });
    stream << "] shape=";
    printPlanShape(stream, buffer.shape);
    stream << " strides=";
    printPlanShape(stream, buffer.strides);
    stream << " effective-byte-offset-bytes=";
    if (buffer.effectiveByteOffsetBytes)
      stream << *buffer.effectiveByteOffsetBytes;
    else
      stream << "<unknown>";
    stream << " capacity=" << buffer.capacityBytes << " spm-lifetime=";
    if (buffer.spmFirstTask && buffer.spmLastTask)
      stream << (buffer.crossTaskResident ? "cross-task task="
                                          : "operation-local task=")
             << *buffer.spmFirstTask << ".." << *buffer.spmLastTask;
    else if (buffer.operationLocalLifetime)
      stream << "operation-local task=<unannotated>";
    else
      stream << "unknown";
    stream << " physical-users=";
    printPlanShape(stream, buffer.physicalUseTasks);
    stream << " physical-lifetime="
           << (buffer.physicalLifetimeKnown ? "known" : "unknown");
    if (buffer.physicalFirstTask && buffer.physicalLastTask)
      stream << " tasks=" << *buffer.physicalFirstTask << ".."
             << *buffer.physicalLastTask;
    stream << '\n';
  }
  stream << "  buffer-alias-pairs: count=" << workspace.bufferAliasPairs.size()
         << " scope=known-roots-within-function unproven-pairs=unknown status="
         << (workspace.bufferAliasPairsComplete ? "complete" : "bounded")
         << '\n';
  for (auto [index, pair] : llvm::enumerate(workspace.bufferAliasPairs)) {
    stream << "  buffer-alias-pair[" << index << "]: function=@"
           << (pair.functionName ? pair.functionName.getValue() : "<unknown>")
           << " accesses=[" << pair.firstAccessIndex << ','
           << pair.secondAccessIndex
           << "] relation=" << getBufferAliasRelationName(pair.relation)
           << " proof=" << getBufferAliasProofName(pair.proof) << '\n';
  }
  stream << "  buffer-source-events: count="
         << workspace.bufferSourceEvents.size()
         << " scope=pre-accumulator-fusion\n";
  for (auto [index, event] : llvm::enumerate(workspace.bufferSourceEvents)) {
    stream << "  buffer-source-event[" << index << "]: function=@"
           << (event.functionName ? event.functionName.getValue()
                                  : StringRef("<unknown>"))
           << " source-origin-id=" << event.sourceOriginId
           << " op=" << event.operationName
           << " effect=" << getBufferAccessEffectName(event.effect)
           << " effect-index=" << event.effectIndex << " operand=";
    if (event.operandNumber >= 0)
      stream << event.operandNumber;
    else
      stream << "<unknown>";
    stream << " type=";
    if (event.accessedType)
      stream << event.accessedType;
    else
      stream << "<unknown>";
    stream << " memory-space=";
    if (!event.accessedType)
      stream << "<unknown>";
    else if (event.memorySpace)
      stream << event.memorySpace;
    else
      stream << "default";
    stream << " views=[";
    llvm::interleaveComma(event.viewChain, stream,
                          [&](StringRef view) { stream << view; });
    stream << "] shape=";
    printPlanShape(stream, event.shape);
    stream << " logical-shape=";
    printPlanShape(stream, event.logicalShape);
    stream << " strides=";
    printPlanShape(stream, event.strides);
    stream << " valid-shape=";
    printPlanShape(stream, event.validShape);
    stream << " effective-byte-offset-bytes=";
    if (event.effectiveByteOffsetBytes)
      stream << *event.effectiveByteOffsetBytes;
    else
      stream << "<unknown>";
    stream << " valid-footprint-bytes=";
    if (event.logicalFootprintBytes)
      stream << *event.logicalFootprintBytes;
    else
      stream << "<unknown>";
    stream << " physical-span-bytes=";
    if (event.physicalSpanBytes)
      stream << *event.physicalSpanBytes;
    else
      stream << "<unknown>";
    stream << '\n';
  }
  stream << "  buffer-accesses: count=" << workspace.bufferAccesses.size()
         << " scope=post-planning\n";
  for (auto [index, access] : llvm::enumerate(workspace.bufferAccesses)) {
    stream << "  buffer-access[" << index << "]: op=";
    if (access.operation)
      stream << access.operation->getName().getStringRef();
    else
      stream << "<unknown>";
    stream << " effect=" << getBufferAccessEffectName(access.effect)
           << " effect-index=" << access.effectIndex << " operand=";
    if (access.operandNumber >= 0)
      stream << access.operandNumber;
    else
      stream << "<unknown>";
    if (access.sourceOriginId)
      stream << " source-origin-id=" << *access.sourceOriginId;
    stream << " type=";
    if (access.accessedType)
      stream << access.accessedType;
    else
      stream << "<unknown>";
    stream << " memory-space=";
    if (!access.accessedType)
      stream << "<unknown>";
    else if (access.memorySpace)
      stream << access.memorySpace;
    else
      stream << "default";
    stream << " root=" << (access.storageRootKnown ? "known" : "unknown");
    if (access.storageRootKnown)
      stream << " root-id=" << access.storageRootId;
    stream << " views=[";
    llvm::interleaveComma(access.viewChain, stream, [&](Operation *view) {
      stream << view->getName().getStringRef();
    });
    stream << "] shape=";
    printPlanShape(stream, access.shape);
    stream << " logical-shape=";
    printPlanShape(stream, access.logicalShape);
    stream << " strides=";
    printPlanShape(stream, access.strides);
    stream << " valid-shape=";
    printPlanShape(stream, access.validShape);
    stream << " effective-byte-offset-bytes=";
    if (access.effectiveByteOffsetBytes)
      stream << *access.effectiveByteOffsetBytes;
    else
      stream << "<unknown>";
    stream << " valid-footprint-bytes=";
    if (access.logicalFootprintBytes)
      stream << *access.logicalFootprintBytes;
    else
      stream << "<unknown>";
    stream << " physical-span-bytes=";
    if (access.physicalSpanBytes)
      stream << *access.physicalSpanBytes;
    else
      stream << "<unknown>";
    stream << " task=";
    if (access.computeTaskId)
      stream << *access.computeTaskId;
    else
      stream << "<unannotated>";
    stream << " transfer=";
    if (access.transferIndex)
      stream << *access.transferIndex;
    else
      stream << "<none>";
    stream << " alias=unknown dma="
           << (access.transferIndex ? "planned-legal" : "unassessed");
    if (access.logicalValueId)
      stream << " logical-value=" << *access.logicalValueId;
    auto routeName = [](BufferAccessRoute route) -> StringRef {
      switch (route) {
      case BufferAccessRoute::Unassessed:
        return "unassessed";
      case BufferAccessRoute::DirectDdr:
        return "direct-ddr";
      case BufferAccessRoute::ExistingSpm:
        return "existing-spm";
      case BufferAccessRoute::Staging:
        return "staging";
      }
      llvm_unreachable("unknown access route");
    };
    stream << " route=" << routeName(access.route);
    if (!access.routeReason.empty())
      stream << " reason=" << access.routeReason;
    stream << " route-bytes=";
    if (access.routeBytes)
      stream << *access.routeBytes;
    else
      stream << "<unknown>";
    stream << " route-bank=" << access.routeBank
           << " route-offset=" << access.routeOffset
           << " route-capacity=" << access.routeCapacityBytes;
    stream << '\n';
  }
  stream << "  buffer-inventory: count=" << workspace.bufferInventory.size()
         << " scope=post-planning-observed-memref-values\n";
  for (auto [index, buffer] : llvm::enumerate(workspace.bufferInventory)) {
    stream << "  buffer-inventory[" << index << "]: function=@";
    if (buffer.functionName)
      stream << buffer.functionName.getValue();
    else
      stream << "<unknown>";
    stream << " value=";
    buffer.descriptor.print(stream);
    stream << " type=";
    if (buffer.descriptorType)
      stream << buffer.descriptorType;
    else
      stream << "<unknown>";
    stream << " memory-space=";
    if (buffer.memorySpace)
      stream << buffer.memorySpace;
    else
      stream << "default";
    stream << " root=" << (buffer.storageRootKnown ? "known" : "unknown");
    if (buffer.storageRootKnown)
      stream << " root-id=" << buffer.storageRootId;
    stream << " root-capacity-bytes=";
    if (buffer.rootCapacityBytes)
      stream << *buffer.rootCapacityBytes;
    else
      stream << "<unknown>";
    stream << " views=[";
    llvm::interleaveComma(buffer.viewChain, stream, [&](Operation *view) {
      stream << view->getName().getStringRef();
    });
    stream << "] shape=";
    printPlanShape(stream, buffer.shape);
    stream << " strides=";
    printPlanShape(stream, buffer.strides);
    stream << " effective-byte-offset-bytes=";
    if (buffer.effectiveByteOffsetBytes)
      stream << *buffer.effectiveByteOffsetBytes;
    else
      stream << "<unknown>";
    stream << " accesses=[";
    llvm::interleaveComma(buffer.accessIndices, stream,
                          [&](size_t accessIndex) { stream << accessIndex; });
    stream << "] logical-values=";
    printPlanShape(stream, buffer.logicalValueIds);
    stream << " tasks=";
    printPlanShape(stream, buffer.taskIds);
    stream << " transfers=[";
    llvm::interleaveComma(
        buffer.transferIndices, stream,
        [&](size_t transferIndex) { stream << transferIndex; });
    stream << "] physical-lifetime="
           << (buffer.physicalLifetimeKnown ? "known" : "unknown") << '\n';
  }
  stream << "  memory-dependencies: count="
         << workspace.memoryDependencies.size() << " scope=lexical status="
         << (workspace.memoryDependenciesBounded    ? "bounded"
             : workspace.memoryDependenciesComplete ? "complete"
                                                    : "incomplete")
         << '\n';
  for (auto [index, dependency] : llvm::enumerate(workspace.memoryDependencies))
    stream << "  memory-dependency[" << index << "]: function=@"
           << dependency.functionName.getValue() << " accesses=["
           << dependency.predecessorAccessIndex << ','
           << dependency.successorAccessIndex
           << "] kind=" << getMemoryDependencyKindName(dependency.kind) << '\n';
  stream << "  resources: transfers=" << workspace.resources.plannedTransfers
         << " dma-packets=" << workspace.resources.plannedDmaPackets
         << " read-bytes=" << workspace.resources.plannedReadBytes
         << " write-bytes=" << workspace.resources.plannedWriteBytes
         << " totals-exact="
         << (workspace.resources.transferTotalsExact ? "true" : "false")
         << " latency=";
  if (workspace.cost.latencyCalibrated && workspace.cost.latencyCycles)
    stream << *workspace.cost.latencyCycles << "-cycles";
  else
    stream << "uncalibrated";
  stream << '\n';
  stream << "  bank-allocation-search: scope=operation-local groups="
         << workspace.resources.bankAllocationGroups << " exact-assignments="
         << workspace.resources.bankAllocationCandidates
         << " exact-rejected-assignments="
         << workspace.resources.bankAllocationRejectedCapacity
         << " bounded-groups="
         << workspace.resources.bankAllocationBoundedGroups
         << " repair-groups=" << workspace.resources.bankAllocationRepairGroups
         << " repair-search-nodes="
         << workspace.resources.bankAllocationRepairSearchNodes
         << " repair-rejected-placements="
         << workspace.resources.bankAllocationRepairRejectedPlacements
         << " objective=minimize-bank-high-water status="
         << (workspace.resources.bankAllocationBoundedGroups == 0 ? "exact"
                                                                  : "bounded")
         << " latency=uncalibrated\n";

  uint64_t selectedArgumentResidencies = llvm::count_if(
      workspace.functionArgumentResidency,
      [](const FunctionArgumentResidencyDecision &decision) {
        return decision.reason == ResidencyDecisionReason::Selected;
      });
  uint64_t capacityRejectedArgumentResidencies = llvm::count_if(
      workspace.functionArgumentResidency,
      [](const FunctionArgumentResidencyDecision &decision) {
        return decision.reason == ResidencyDecisionReason::Capacity;
      });
  stream << "  function-argument-residency-search: candidates="
         << workspace.functionArgumentResidency.size()
         << " selected=" << selectedArgumentResidencies
         << " rejected-capacity=" << capacityRejectedArgumentResidencies
         << " objective=minimize-read-bytes-then-packets status="
         << (workspace.functionArgumentResidencySearchBounded ? "bounded"
                                                              : "exact")
         << " latency=uncalibrated\n";
  for (auto [index, decision] :
       llvm::enumerate(workspace.functionArgumentResidency)) {
    auto argument = dyn_cast<BlockArgument>(decision.argument);
    auto function =
        argument ? dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp())
                 : func::FuncOp();
    stream << "  argument-residency[" << index << "]: function=@";
    if (function)
      stream << function.getSymName();
    else
      stream << "<unknown>";
    stream << " arg#" << (argument ? argument.getArgNumber() : 0)
           << " consumers=" << decision.consumerCount
           << " capacity=" << decision.capacityBytes
           << " resident-read-bytes=" << decision.residentReadBytes
           << " resident-packets=" << decision.residentPackets
           << " reload-read-bytes=" << decision.reloadReadBytes
           << " reload-packets=" << decision.reloadPackets << " decision="
           << (decision.reason == ResidencyDecisionReason::Selected ? "resident"
                                                                    : "reload")
           << " reason=" << getResidencyDecisionReason(decision.reason) << '\n';
  }

  uint64_t selectedProducerResidencies =
      llvm::count_if(workspace.producerOutputResidency,
                     [](const ProducerOutputResidencyDecision &decision) {
                       return decision.selected;
                     });
  uint64_t eligibleArgumentResidencies = llvm::count_if(
      workspace.functionArgumentResidency,
      [](const FunctionArgumentResidencyDecision &decision) {
        return decision.reason != ResidencyDecisionReason::NoTrafficBenefit;
      });
  uint64_t jointResidencyCandidates =
      eligibleArgumentResidencies + workspace.producerOutputResidency.size();
  uint64_t jointResidencySelected =
      selectedArgumentResidencies + selectedProducerResidencies;
  uint64_t jointResidencyCapacityRejected =
      capacityRejectedArgumentResidencies +
      (workspace.producerOutputResidency.size() - selectedProducerResidencies);
  stream << "  cross-task-residency-search: candidates="
         << jointResidencyCandidates << " selected=" << jointResidencySelected
         << " rejected-capacity=" << jointResidencyCapacityRejected
         << " objective=minimize-total-read-bytes-then-packets status="
         << (workspace.functionArgumentResidencySearchBounded ||
                     workspace.producerOutputResidencySearchBounded
                 ? "bounded"
                 : "exact")
         << " latency=uncalibrated\n";
  stream << "  producer-output-residency-search: candidates="
         << workspace.producerOutputResidency.size()
         << " selected=" << selectedProducerResidencies << " rejected-capacity="
         << (workspace.producerOutputResidency.size() -
             selectedProducerResidencies)
         << " objective=maximize-read-bytes-then-packets-saved status="
         << (workspace.producerOutputResidencySearchBounded ? "bounded"
                                                            : "exact")
         << " latency=uncalibrated\n";
  for (auto [index, decision] :
       llvm::enumerate(workspace.producerOutputResidency)) {
    Operation *producer = decision.producer;
    auto function =
        producer ? producer->getParentOfType<func::FuncOp>() : func::FuncOp();
    stream << "  producer-output-residency[" << index << "]: function=@";
    if (function)
      stream << function.getSymName();
    else
      stream << "<unknown>";
    stream << " consumers=" << decision.consumerCount
           << " capacity=" << decision.capacityBytes
           << " saved-read-bytes=" << decision.savedReadBytes
           << " saved-packets=" << decision.savedPackets << " decision="
           << (decision.selected ? "resident" : "writeback-and-reload")
           << " reason="
           << (decision.selected ? "traffic-benefit-and-capacity"
                                 : "spm-capacity")
           << '\n';
  }

  int64_t bankUsage[2] = {workspace.usedBytes[0], workspace.usedBytes[1]};
  module.walk([&](SpmViewOp view) {
    int64_t bank = view.getBank();
    auto type = dyn_cast<MemRefType>(view.getView().getType());
    auto capacity = type ? spmdma::getContiguousZeroOffsetByteSize(type)
                         : std::optional<int64_t>();
    if (bank >= 0 && bank < 2 && capacity) {
      int64_t offset = static_cast<int64_t>(view.getOffset());
      bankUsage[bank] = std::max(bankUsage[bank], offset + *capacity);
    }
  });
  for (int64_t bank = 0; bank < 2; ++bank)
    stream << "  bank[" << bank << "]: high-water=" << bankUsage[bank] << '/'
           << workspace.bankCapacity << " bytes\n";

  unsigned markedMatmuls = 0;
  module.walk([&](linalg::MatmulOp matmul) {
    if (!matmul->hasAttr(::buddy::boscame::lowering::kLowerToAme))
      return;
    ++markedMatmuls;
    auto aType = dyn_cast<MemRefType>(matmul.getInputs()[0].getType());
    auto bType = dyn_cast<MemRefType>(matmul.getInputs()[1].getType());
    stream << "  marked-matmul: execution=AME m=" << aType.getDimSize(0)
           << " n=" << bType.getDimSize(1) << " k=" << aType.getDimSize(1)
           << " a_origin_id=";
    if (auto origin =
            matmul->getAttrOfType<IntegerAttr>("bosc_ame.a_origin_id"))
      stream << origin.getInt();
    else
      stream << "<unknown>";
    stream << " b_origin_id=";
    if (auto origin =
            matmul->getAttrOfType<IntegerAttr>("bosc_ame.b_origin_id"))
      stream << origin.getInt();
    else
      stream << "<unknown>";
    stream << " accumulator_origin_id=";
    if (auto origin = matmul->getAttrOfType<IntegerAttr>(
            "bosc_ame.accumulator_origin_id"))
      stream << origin.getInt();
    else
      stream << "<unknown>";
    stream << " direct-A="
           << getDirectDecision(
                  matmul->hasAttr("bosc_ame.direct_ddr_a"),
                  findPlannedTransfer(workspace, matmul.getInputs()[0]))
           << " direct-B="
           << getDirectDecision(
                  matmul->hasAttr("bosc_ame.direct_ddr_b"),
                  findPlannedTransfer(workspace, matmul.getInputs()[1]))
           << " direct-C="
           << getDirectDecision(
                  matmul->hasAttr("bosc_ame.direct_ddr_c"),
                  findPlannedTransfer(workspace, matmul.getOutputs()[0]))
           << '\n';
    if (const AmeMatmulTilePlan *tilePlan =
            findAmeMatmulTilePlan(workspace, matmul.getOperation())) {
      if (tilePlan->mTileCount > 0 && tilePlan->nTileCount > 0 &&
          tilePlan->kTileCount > 0) {
        stream << "  ame-tile-plan: m=" << tilePlan->m << " n=" << tilePlan->n
               << " k=" << tilePlan->k << " tile=[" << tilePlan->tileM << ','
               << tilePlan->tileN << ',' << tilePlan->tileK << "] tile-counts=["
               << tilePlan->mTileCount << ',' << tilePlan->nTileCount << ','
               << tilePlan->kTileCount << "] tail=[" << tilePlan->tailM << ','
               << tilePlan->tailN << ',' << tilePlan->tailK << "]\n";
        auto route = [&](bool direct, bool eligible, bool byCapacity,
                         bool bySpmDisabled, uint64_t directBytes,
                         uint64_t stagedBytes, Value input) {
          if (direct) {
            if (byCapacity)
              return StringRef("direct-capacity-fallback");
            if (bySpmDisabled)
              return StringRef("direct-spm-disabled");
            return directBytes == stagedBytes
                       ? StringRef("direct-byte-tie")
                       : StringRef("direct-lower-read-bytes");
          }
          if (findPlannedTransfer(workspace, input))
            return eligible ? StringRef("staged-lower-read-bytes")
                            : StringRef("staged-direct-unavailable");
          return StringRef("not-inferred");
        };
        stream << "  ame-input-routes: A="
               << route(matmul->hasAttr("bosc_ame.direct_ddr_a"),
                        tilePlan->directAEligible, tilePlan->directAByCapacity,
                        tilePlan->directABySpmDisabled,
                        tilePlan->directAReadBytes, tilePlan->stagedAReadBytes,
                        matmul.getInputs()[0])
               << " direct-bytes=" << tilePlan->directAReadBytes
               << " staged-bytes=" << tilePlan->stagedAReadBytes << " B="
               << route(matmul->hasAttr("bosc_ame.direct_ddr_b"),
                        tilePlan->directBEligible, tilePlan->directBByCapacity,
                        tilePlan->directBBySpmDisabled,
                        tilePlan->directBReadBytes, tilePlan->stagedBReadBytes,
                        matmul.getInputs()[1])
               << " direct-bytes=" << tilePlan->directBReadBytes
               << " staged-bytes=" << tilePlan->stagedBReadBytes << '\n';
      } else {
        stream << "  ame-tile-plan: unavailable\n";
      }
    }
  });
  if (!markedMatmuls)
    stream << "  marked-matmuls: none\n";

  bool reportedStoreDecision = false;
  module.walk([&](Operation *operation) {
    auto memory = operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
    auto kind = memory ? memory.getAs<StringAttr>("kind") : StringAttr();
    auto destination =
        memory ? memory.getAs<StringAttr>("destination") : StringAttr();
    if (!kind || kind.getValue() != "store" || !destination ||
        destination.getValue() != "external_ddr")
      return;

    StringRef decision;
    if (operation->hasAttr("bosc_ame.direct_ddr") ||
        operation->hasAttr("bosc_ame.direct_ddr_store") ||
        operation->hasAttr("bosc_ame.direct_ddr_c")) {
      decision = "direct-ddr (proven)";
    } else if (auto linalg = dyn_cast<linalg::LinalgOp>(operation)) {
      for (Value output : linalg.getDpsInits())
        if (findPlannedTransfer(workspace, output)) {
          decision = "spm-dma (planned)";
          break;
        }
    }
    if (decision.empty())
      return;
    if (!reportedStoreDecision)
      stream << "  provenance-marked stores/direct decisions:\n";
    reportedStoreDecision = true;
    stream << "    op=" << operation->getName().getStringRef()
           << " destination=external_ddr decision=" << decision << '\n';
  });
  if (!reportedStoreDecision)
    stream << "  provenance-marked stores/direct decisions: none proven\n";

  if (config.bufferPlacement && !workspace.bufferPlacements.empty())
    stream << "  limits: interval reuse, spill search, general control-flow "
              "memory ordering, "
              "materialization "
              "elimination, register budgeting, and cross-program reuse are "
              "not modeled.\n";
  else
    stream
        << "  limits: calibrated cost search, general cross-op residency, and "
           "cross-program reuse are not modeled.\n";
}

template <bool Independent>
class LowerLinalgToBOSCAMEPass
    : public PassWrapper<LowerLinalgToBOSCAMEPass<Independent>,
                         OperationPass<ModuleOp>> {
public:
  using Base = PassWrapper<LowerLinalgToBOSCAMEPass<Independent>,
                           OperationPass<ModuleOp>>;
  using Base::getContext;
  using Base::getOperation;
  using Base::signalPassFailure;
  template <typename T> using Option = typename Base::template Option<T>;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBOSCAMEPass)

  StringRef getArgument() const final {
    return Independent ? "linalg-to-boscame" : "lower-linalg-to-boscame";
  }
  StringRef getDescription() const final {
    return Independent
               ? "Lower buffer-form Linalg through a typed SPM/DMA planner."
               : "Lower linalg dialect operations to BOSCAME dialect "
                 "operations.";
  }

  LowerLinalgToBOSCAMEPass() = default;
  LowerLinalgToBOSCAMEPass(const LowerLinalgToBOSCAMEPass &) {}

  Option<bool> enableSpmDma{*this, "enable-spm-dma",
                            llvm::cl::desc("plan typed SPM/DMA transfers"),
                            llvm::cl::init(false)};
  Option<bool> enableAME{*this, "enable-ame",
                         llvm::cl::desc("lower marked i8 matmuls to AME"),
                         llvm::cl::init(false)};
  Option<std::string> dmaAbi{*this, "dma-abi",
                             llvm::cl::desc("DMA caller contract"),
                             llvm::cl::init("")};
  Option<std::string> target{*this, "target",
                             llvm::cl::desc("BOSCAME target profile"),
                             llvm::cl::init("")};
  Option<int64_t> bm{*this, "bm", llvm::cl::desc("M tile size"),
                     llvm::cl::init(64)};
  Option<int64_t> bn{*this, "bn", llvm::cl::desc("N tile size"),
                     llvm::cl::init(64)};
  Option<int64_t> bk{*this, "bk", llvm::cl::desc("K tile size"),
                     llvm::cl::init(128)};
  Option<bool> planReport{
      *this, "plan-report",
      llvm::cl::desc("print the planned BOSCAME workspace to stderr"),
      llvm::cl::init(false)};
  Option<bool> bufferPlacement{
      *this, "buffer-placement",
      llvm::cl::desc(
          "place connected static buffer roots in fixed SPM windows"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<math::MathDialect>();
  }

  void runOnOperation() override;
};
} // namespace

template <bool Independent>
void LowerLinalgToBOSCAMEPass<Independent>::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  if (failed(verifyKnownLaunchGrid(module))) {
    signalPassFailure();
    return;
  }

  if (planReport && !Independent) {
    module.emitError("plan-report is only supported by --linalg-to-boscame");
    signalPassFailure();
    return;
  }

  if (module->hasAttr("bosc_ame.store_policy")) {
    module.emitError("bosc_ame.store_policy is obsolete; remove it and "
                     "rerun the automatic BOSCAME planner");
    signalPassFailure();
    return;
  }

  // The planner is opt-in.  Keeping this boundary explicit preserves the
  // historical --lower-linalg-to-boscame output for existing users while
  // making the new buffer-form contract fail early and diagnostically.
  if (Independent || enableSpmDma || enableAME || bufferPlacement ||
      !target.empty() || module->hasAttr("bosc_ame.enable_spm_dma") ||
      module->hasAttr("bosc_ame.enable_ame") ||
      module->hasAttr("bosc_ame.target")) {
    using namespace ::buddy::boscame::spmdma;
    auto inheritBool = [&](StringRef name, auto &option) -> LogicalResult {
      if (Attribute attr = module->getAttr(name)) {
        auto value = dyn_cast<BoolAttr>(attr);
        if (!value ||
            (option.getNumOccurrences() && value.getValue() != option))
          return module.emitError()
                 << "invalid or conflicting BOSCAME module configuration: "
                 << name;
        option = value.getValue();
      }
      return success();
    };
    auto inheritTile = [&](StringRef name, auto &option) -> LogicalResult {
      if (Attribute attr = module->getAttr(name)) {
        auto value = dyn_cast<IntegerAttr>(attr);
        if (!value || value.getInt() <= 0 ||
            (option.getNumOccurrences() && value.getInt() != option))
          return module.emitError()
                 << "invalid or conflicting BOSCAME module configuration: "
                 << name;
        option = value.getInt();
      }
      return success();
    };
    if (failed(inheritBool("bosc_ame.enable_spm_dma", enableSpmDma)) ||
        failed(inheritBool("bosc_ame.enable_ame", enableAME)) ||
        failed(inheritTile("bosc_ame.bm", bm)) ||
        failed(inheritTile("bosc_ame.bn", bn)) ||
        failed(inheritTile("bosc_ame.bk", bk))) {
      signalPassFailure();
      return;
    }
    if (module->hasAttr(kDmaAbiAttrName) &&
        !module->getAttrOfType<StringAttr>(kDmaAbiAttrName)) {
      module.emitError(
          "invalid BOSCAME module DMA ABI contract: expected string");
      signalPassFailure();
      return;
    }
    StringRef existingAbi =
        module->getAttrOfType<StringAttr>(kDmaAbiAttrName)
            ? module->getAttrOfType<StringAttr>(kDmaAbiAttrName).getValue()
            : StringRef();
    if (module->hasAttr(kDmaAbiAttrName) && !dmaAbi.empty() &&
        existingAbi != dmaAbi) {
      module.emitError(
          "command-line DMA ABI conflicts with the module contract");
      signalPassFailure();
      return;
    }
    if (module->hasAttr(kDmaAbiAttrName) &&
        existingAbi != kDmaAbiAlignedNonAliasing) {
      module.emitError("invalid BOSCAME module DMA ABI contract");
      signalPassFailure();
      return;
    }
    if (!dmaAbi.empty() && dmaAbi != kDmaAbiAlignedNonAliasing) {
      module.emitError() << "invalid dma-abi: " << dmaAbi;
      signalPassFailure();
      return;
    }
    if (dmaAbi.empty() && !existingAbi.empty())
      dmaAbi = existingAbi.str();
    if (enableSpmDma && dmaAbi.empty()) {
      module.emitError(
          "enable-spm-dma requires dma-abi=args_aligned64_nonalias");
      signalPassFailure();
      return;
    }
    if (Attribute attr = module->getAttr("bosc_ame.target")) {
      auto contract = dyn_cast<StringAttr>(attr);
      if (!contract || contract.getValue() != "nr-fpga" ||
          (target.getNumOccurrences() && contract.getValue() != target)) {
        module.emitError(
            "invalid or conflicting BOSCAME module target contract");
        signalPassFailure();
        return;
      }
      target = contract.getValue().str();
    }
    ::buddy::boscame::LinalgToBOSCAMEConfig config;
    config.enableSpmDma = enableSpmDma;
    config.enableAME = enableAME;
    // Logical frontend buffers are compiler-owned snapshots. Place them
    // automatically; retain the existing option as a physical-placement
    // debugging override without changing tensor fusion.
    auto placement = module->getAttrOfType<StringAttr>("bosc_ame.placement");
    config.bufferPlacement =
        bufferPlacement.getNumOccurrences()
            ? bool(bufferPlacement)
            : enableSpmDma && placement && placement.getValue() == "logical";
    config.dmaAbi = dmaAbi;
    config.target = target;
    config.bm = bm;
    config.bn = bn;
    config.bk = bk;
    if (Attribute attr = module->getAttr("bosc_ame.expected_ame_matmuls")) {
      auto expected = dyn_cast<IntegerAttr>(attr);
      int64_t actual = 0;
      module.walk([&](Operation *op) {
        actual += op->hasAttr(::buddy::boscame::lowering::kLowerToAme);
      });
      if (!expected || expected.getInt() < 0 || expected.getInt() != actual) {
        module.emitError("BOSCAME AME marker count changed across explicit "
                         "bufferization");
        signalPassFailure();
        return;
      }
    }
    if (failed(verifyStandaloneSurface(module, false)) ||
        failed(::buddy::boscame::normalizeBOSCAMEMemoryCopies(module, true))) {
      signalPassFailure();
      return;
    }
    auto workspace = ::buddy::boscame::BOSCAMEPlanner(config).plan(module);
    if (failed(workspace)) {
      signalPassFailure();
      return;
    }
    if (planReport)
      printBOSCAMEPlanReport(module, config, *workspace);
    auto graphVersion =
        module->getAttrOfType<IntegerAttr>("bosc_ame.compute_graph_version");
    bool preserveGraph =
        (graphVersion && graphVersion.getInt() == 1) ||
        llvm::any_of(workspace->bufferPlacements,
                     [](const BufferPlacementDecision &decision) {
                       return decision.applied;
                     });
    if (config.bufferPlacement)
      module.walk([&](Operation *op) {
        if (op->hasAttr("bosc_ame.compute_id") &&
            llvm::any_of(op->getOperandTypes(), [](Type type) {
              return ::buddy::boscame::spmdma::getSpmBank(type).has_value();
            }))
          preserveGraph = true;
      });
    auto verifyGraph = [&](ModuleOp current) {
      return preserveGraph
                 ? ::buddy::boscame::verifyCompleteComputeGraphs(current)
                 : success();
    };
    if (failed(verifyStandaloneSurface(module)) ||
        failed(
            ::buddy::boscame::verifyBOSCAMEAmeConsumers(module, *workspace)) ||
        failed(::buddy::boscame::emitBOSCAMEDmaTransfers(module, *workspace)) ||
        failed(::buddy::boscame::lowerBOSCAMEAmeConsumers(
            module, config, *workspace, preserveGraph)) ||
        failed(::buddy::boscame::emitBOSCAMEDmaCopies(module, workspace->target,
                                                      preserveGraph)) ||
        failed(::buddy::boscame::lowering::lowerSpmComputations(
            module, preserveGraph ? llvm::function_ref<LogicalResult(ModuleOp)>(
                                        verifyGraph)
                                  : nullptr))) {
      signalPassFailure();
      return;
    }
    LogicalResult ordinaryStatus = success();
    if (preserveGraph) {
      SmallVector<linalg::LinalgOp> operations;
      module.walk([&](linalg::LinalgOp op) { operations.push_back(op); });
      PatternRewriter rewriter(context);
      for (linalg::LinalgOp op : operations) {
        rewriter.setInsertionPoint(op);
        if (failed(lowerOrdinaryLinalg(op, rewriter, true)) ||
            failed(::buddy::boscame::verifyCompleteComputeGraphs(module))) {
          ordinaryStatus = failure();
          break;
        }
      }
    } else {
      RewritePatternSet ordinary(context);
      ordinary.add<OrdinaryLinalgToLoops>(context);
      ordinaryStatus = applyPatternsGreedily(module, std::move(ordinary));
    }
    if (succeeded(ordinaryStatus)) {
      ::buddy::boscame::lowering::promotePrivateScalarBuffers(module);
      ::buddy::boscame::lowering::eraseDeadSpmComputationArtifacts(module);
      if (preserveGraph)
        ordinaryStatus = ::buddy::boscame::verifyCompleteComputeGraphs(module);
    }
    if (failed(ordinaryStatus) || failed(promoteStaticTemporaries(module)) ||
        failed(verifyStandaloneSurface(module)) ||
        failed(::buddy::boscame::lowering::verifyExit(module)) ||
        failed(::buddy::boscame::verifySpmDmaContract(module)))
      signalPassFailure();
    return;
  }

  RewritePatternSet patterns(context);
  patterns.add<
      MatmulToBOSCAMELowering, GenericMatmulToBOSCAMELowering,
      GenericElementwiseToBOSCAMELowering, GenericUnarySquareToBOSCAMELowering,
      GenericTransposeToBOSCAMELowering, GenericBroadcastToBOSCAMELowering>(
      context);

  ConversionTarget conversionTarget(*context);
  conversionTarget
      .addLegalDialect<::buddy::boscame::BOSCAMEDialect, arith::ArithDialect,
                       memref::MemRefDialect, scf::SCFDialect>();
  conversionTarget.addIllegalOp<linalg::MatmulOp>();
  conversionTarget.addDynamicallyLegalOp<linalg::GenericOp>(
      [](linalg::GenericOp op) { return !isLowerableGeneric(op); });

  if (failed(applyPartialConversion(module, conversionTarget,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
std::unique_ptr<Pass> createLowerLinalgToBOSCAMEPass() {
  return std::make_unique<LowerLinalgToBOSCAMEPass<false>>();
}

std::unique_ptr<Pass> createLinalgToBOSCAMEPass() {
  return std::make_unique<LowerLinalgToBOSCAMEPass<true>>();
}

void registerLowerLinalgToBOSCAMEPass() {
  PassRegistration<LowerLinalgToBOSCAMEPass<false>>();
  PassRegistration<LowerLinalgToBOSCAMEPass<true>>();
}
} // namespace buddy
} // namespace mlir
