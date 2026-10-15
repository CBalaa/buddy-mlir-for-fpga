//===- SPMComputations.cpp - One-dimensional computation on SPM
//------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {
namespace {

Value index(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

bool touchesSpm(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](Type type) { return getSpmBank(type).has_value(); });
}

/// A vector always follows the last physical axis. Outer dimensions are
/// scalar loop indices, never extra dimensions of the vector value: rank-two
/// vectors are reserved for AME by the common LLVM conversion.
LogicalResult checkVectorBuffer(Operation *op, Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 2 ||
      llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }))
    return op->emitError(
        "SPM computation needs static positive rank-1 or rank-2 buffers");
  unsigned bytes = getSpmElementByteWidth(type.getElementType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!bytes || failed(type.getStridesAndOffset(strides, offset)) ||
      strides.back() != 1 ||
      (type.getRank() == 2 && strides.front() < type.getShape().back()))
    return op->emitError(
        "SPM computation needs a contiguous final physical axis");
  if (type.getShape().back() % (kDmaAlignmentBytes / bytes))
    return op->emitError(
        "SPM computation requires complete 64 B vectors; scalar tails are "
        "unsupported");
  return success();
}

int64_t lanesFor(Value value) {
  auto type = cast<MemRefType>(value.getType());
  return kDmaAlignmentBytes / getSpmElementByteWidth(type.getElementType());
}

template <typename Body>
void vectorLoop(OpBuilder &builder, Location loc, MemRefType type,
                int64_t lanes, Body body) {
  OpBuilder::InsertionGuard guard(builder);
  Value zero = index(builder, loc, 0), one = index(builder, loc, 1);
  SmallVector<Value> indices;
  if (type.getRank() == 2) {
    auto rows = scf::ForOp::create(
        builder, loc, zero, index(builder, loc, type.getDimSize(0)), one);
    builder.setInsertionPointToStart(rows.getBody());
    indices.push_back(rows.getInductionVar());
  }
  auto columns = scf::ForOp::create(builder, loc, zero,
                                    index(builder, loc, type.getShape().back()),
                                    index(builder, loc, lanes));
  builder.setInsertionPointToStart(columns.getBody());
  indices.push_back(columns.getInductionVar());
  body(indices);
}

LogicalResult checkFill(linalg::FillOp op) {
  if (!op.hasPureBufferSemantics() || op.getInputs().size() != 1 ||
      op.getOutputs().size() != 1)
    return op.emitError("SPM fill requires one buffer output");
  return checkVectorBuffer(op, op.getOutputs()[0]);
}

void lowerFill(linalg::FillOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value output = op.getOutputs()[0];
  auto type = cast<MemRefType>(output.getType());
  int64_t lanes = lanesFor(output);
  auto vectorType = VectorType::get({lanes}, type.getElementType());
  Value fill =
      vector::BroadcastOp::create(builder, loc, vectorType, op.getInputs()[0]);
  vectorLoop(builder, loc, type, lanes, [&](ValueRange indices) {
    vector::StoreOp::create(builder, loc, fill, output, indices);
  });
  op.erase();
}

bool isVectorizableBodyOp(Operation &op) {
  // These lower to scalar/vector instructions, without runtime calls. In
  // particular exp/log/tanh/rsqrt cannot silently introduce a libm call while
  // an SPM vector is live. Scalar rsqrt outside this island remains legal.
  return isa<arith::ConstantOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
             arith::DivFOp, arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
             arith::MaxNumFOp, arith::MinNumFOp, arith::AddIOp, arith::SubIOp,
             arith::MulIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
             arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::CmpFOp,
             arith::CmpIOp, arith::SelectOp, arith::ExtFOp, arith::TruncFOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp,
             arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp, math::AbsFOp,
             math::SqrtOp>(op);
}

bool isVectorElement(Type type) {
  if (type.isF32())
    return true;
  if (auto integer = dyn_cast<IntegerType>(type))
    return llvm::is_contained({1u, 8u, 16u, 32u, 64u}, integer.getWidth());
  // Unsupported float widths and oversized integers may expand into runtime
  // calls even when the MLIR operation itself is side-effect free.
  return false;
}

LogicalResult checkGeneric(linalg::GenericOp op) {
  if (!op.hasPureBufferSemantics() || op.getOutputs().size() != 1 ||
      !llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::parallel;
      }))
    return op.emitError(
        "SPM generic requires a parallel single-output buffer operation");
  Value output = op.getOutputs()[0];
  if (failed(checkVectorBuffer(op, output)))
    return failure();
  auto outputType = cast<MemRefType>(output.getType());
  unsigned bytes = getSpmElementByteWidth(outputType.getElementType());
  auto maps = op.getIndexingMapsArray();
  for (auto [number, operand] : llvm::enumerate(op.getOperands())) {
    if (auto type = dyn_cast<MemRefType>(operand.getType())) {
      if (!type.getRank()) {
        if (getSpmBank(type))
          return op.emitError("SPM scalar broadcast buffers are unsupported");
        if (maps[number].getNumResults())
          return op.emitError("SPM scalar broadcast requires a rank-zero map");
        continue;
      }
      if (failed(checkVectorBuffer(op, operand)))
        return failure();
      if (!getSpmBank(type))
        return op.emitError(
            "SPM generic requires placement of every non-scalar operand; "
            "mixed DDR/SPM computation could carry vectors across cache "
            "policy calls");
      if (type.getShape() != outputType.getShape() ||
          !maps[number].isIdentity() ||
          getSpmElementByteWidth(type.getElementType()) != bytes)
        return op.emitError(
            "SPM generic requires matching shapes, element byte widths and "
            "identity indexing maps");
    } else if (!isVectorElement(operand.getType()) ||
               maps[number].getNumResults()) {
      return op.emitError("unsupported SPM generic scalar operand");
    }
  }
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (!isVectorizableBodyOp(inner) || inner.getNumRegions() ||
        !llvm::all_of(inner.getResultTypes(), isVectorElement))
      return op.emitError()
             << "unsupported operation in SPM vector computation: "
             << inner.getName();
    if (auto constant = dyn_cast<arith::ConstantOp>(inner))
      if (!isa<IntegerAttr, FloatAttr>(constant.getValue()))
        return op.emitError("SPM vector constants must be scalar numbers");
    for (Value operand : inner.getOperands())
      if (!isVectorElement(operand.getType()))
        return op.emitError(
            "SPM vector computation supports only f32 and native integer "
            "elements");
  }
  return success();
}

void lowerGeneric(linalg::GenericOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value output = op.getOutputs()[0];
  auto outputType = cast<MemRefType>(output.getType());
  int64_t lanes = lanesFor(output);
  Block &body = *op.getBody();
  // Read every ordinary scalar before producing any vectors. The late DDR
  // policy pass may introduce calls before these reads, outside the island.
  DenseMap<Value, Value> scalars;
  for (auto [argument, operand] :
       llvm::zip(body.getArguments(), op.getOperands()))
    if (!argument.use_empty()) {
      if (auto type = dyn_cast<MemRefType>(operand.getType())) {
        if (!type.getRank())
          scalars[argument] =
              memref::LoadOp::create(builder, loc, operand, ValueRange{});
      } else {
        scalars[argument] = operand;
      }
    }
  vectorLoop(builder, loc, outputType, lanes, [&](ValueRange indices) {
    IRMapping mapping;
    for (auto [argument, operand] :
         llvm::zip(body.getArguments(), op.getOperands())) {
      if (argument.use_empty())
        continue;
      auto vectorType = VectorType::get({lanes}, argument.getType());
      Value value;
      if (auto scalar = scalars.find(argument); scalar != scalars.end())
        value = vector::BroadcastOp::create(builder, loc, vectorType,
                                            scalar->second);
      else
        value =
            vector::LoadOp::create(builder, loc, vectorType, operand, indices);
      mapping.map(argument, value);
    }
    auto mapped = [&](Value value) -> Value {
      if (Value result = mapping.lookupOrNull(value))
        return result;
      Value result = vector::BroadcastOp::create(
          builder, loc, VectorType::get({lanes}, value.getType()), value);
      mapping.map(value, result);
      return result;
    };
    for (Operation &inner : body.without_terminator()) {
      if (auto constant = dyn_cast<arith::ConstantOp>(inner)) {
        auto vectorType = VectorType::get({lanes}, constant.getType());
        Value result = arith::ConstantOp::create(
            builder, inner.getLoc(), vectorType,
            DenseElementsAttr::get(vectorType,
                                   cast<TypedAttr>(constant.getValue())));
        mapping.map(constant.getResult(), result);
        continue;
      }
      OperationState state(inner.getLoc(), inner.getName());
      for (Value operand : inner.getOperands())
        state.addOperands(mapped(operand));
      for (Type type : inner.getResultTypes())
        state.addTypes(VectorType::get({lanes}, type));
      state.addAttributes(inner.getAttrs());
      Operation *created = builder.create(state);
      for (auto [oldResult, newResult] :
           llvm::zip(inner.getResults(), created->getResults()))
        mapping.map(oldResult, newResult);
    }
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    vector::StoreOp::create(builder, loc, mapped(yield.getValues()[0]), output,
                            indices);
  });
  op.erase();
}

LogicalResult checkReduce(linalg::ReduceOp op) {
  if (!op.hasPureBufferSemantics() || op.getInputs().size() != 1 ||
      op.getInits().size() != 1 || op.getDimensions().size() != 1 ||
      op.getDimensions()[0] != 0)
    return op.emitError("SPM reduction requires a rank-1 to rank-0 sum");
  auto input = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto output = dyn_cast<MemRefType>(op.getInits()[0].getType());
  if (!input || input.getRank() != 1 || !input.getElementType().isF32() ||
      !output || output.getRank() || !output.getElementType().isF32() ||
      getSpmBank(output))
    return op.emitError(
        "SPM reduction requires an f32 vector and an ordinary scalar result");
  if (failed(checkVectorBuffer(op, op.getInputs()[0])))
    return failure();
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 2)
    return op.emitError("SPM reduction supports only a single ordered addf");
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!add || !yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != add.getResult() ||
      !((add.getLhs() == body.getArgument(0) &&
         add.getRhs() == body.getArgument(1)) ||
        (add.getRhs() == body.getArgument(0) &&
         add.getLhs() == body.getArgument(1))) ||
      add.getFastmath() != arith::FastMathFlags::none)
    return op.emitError("SPM reduction requires addf without fast-math flags");
  return success();
}

void lowerReduce(linalg::ReduceOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value input = op.getInputs()[0], output = op.getInits()[0];
  auto type = cast<MemRefType>(input.getType());
  Value initial = memref::LoadOp::create(builder, loc, output, ValueRange{});
  auto loop = scf::ForOp::create(builder, loc, index(builder, loc, 0),
                                 index(builder, loc, type.getDimSize(0)),
                                 index(builder, loc, 1), ValueRange{initial});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value loaded = vector::LoadOp::create(
        builder, loc, VectorType::get({1}, type.getElementType()), input,
        ValueRange{loop.getInductionVar()});
    // The measured FPGA vfredosum sequence differs from the source's f32
    // rounding after every addition. Read one lane through the vector port,
    // then perform the source's scalar f32 addition. Extracting only lane zero
    // avoids unsupported vector slides and keeps the source reduction order.
    // Other SPM transfers retain their complete 64 B vector blocks.
    Block &body = op.getRegion().front();
    auto add = cast<arith::AddFOp>(body.front());
    Value element =
        vector::ExtractOp::create(builder, loc, loaded, ArrayRef<int64_t>{0});
    IRMapping mapping;
    mapping.map(body.getArgument(0), element);
    mapping.map(body.getArgument(1), loop.getRegionIterArgs()[0]);
    Value accumulator = builder.clone(*add, mapping)->getResult(0);
    scf::YieldOp::create(builder, loc, accumulator);
  }
  memref::StoreOp::create(builder, loc, loop.getResult(0), output,
                          ValueRange{});
  op.erase();
}

LogicalResult checkCopy(memref::CopyOp op) {
  if (failed(checkVectorBuffer(op, op.getSource())) ||
      failed(checkVectorBuffer(op, op.getTarget())))
    return failure();
  auto source = cast<MemRefType>(op.getSource().getType());
  auto target = cast<MemRefType>(op.getTarget().getType());
  if (source.getShape() != target.getShape() ||
      source.getElementType() != target.getElementType())
    return op.emitError("SPM copy needs matching static buffer shapes");
  return success();
}

void lowerCopy(memref::CopyOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  auto type = cast<MemRefType>(op.getSource().getType());
  int64_t lanes = lanesFor(op.getSource());
  auto vectorType = VectorType::get({lanes}, type.getElementType());
  vectorLoop(builder, loc, type, lanes, [&](ValueRange indices) {
    Value loaded = vector::LoadOp::create(builder, loc, vectorType,
                                          op.getSource(), indices);
    vector::StoreOp::create(builder, loc, loaded, op.getTarget(), indices);
  });
  op.erase();
}

} // namespace

LogicalResult lowerSpmComputations(ModuleOp module) {
  SmallVector<Operation *> computations;
  WalkResult result = module.walk([&](Operation *op) {
    if (!touchesSpm(op) ||
        (!isa<linalg::LinalgOp>(op) && !isa<memref::CopyOp>(op)))
      return WalkResult::advance();
    LogicalResult checked =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<linalg::FillOp>(checkFill)
            .Case<linalg::GenericOp>(checkGeneric)
            .Case<linalg::ReduceOp>(checkReduce)
            .Case<memref::CopyOp>(checkCopy)
            .Default([](Operation *op) {
              return op->emitError("unsupported Linalg computation on SPM");
            });
    if (failed(checked))
      return WalkResult::interrupt();
    computations.push_back(op);
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  for (Operation *op : computations)
    llvm::TypeSwitch<Operation *>(op)
        .Case<linalg::FillOp>(lowerFill)
        .Case<linalg::GenericOp>(lowerGeneric)
        .Case<linalg::ReduceOp>(lowerReduce)
        .Case<memref::CopyOp>(lowerCopy);
  return success();
}

} // namespace buddy::boscame::lowering
