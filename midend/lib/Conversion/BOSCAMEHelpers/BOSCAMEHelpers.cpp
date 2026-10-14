//===- BOSCAMEHelpers.cpp - Shared staging and workspace
//--------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace buddy::boscame;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {

FailureOr<Value> lowerRank2Load(OpBuilder &, Operation *, Value,
                                RankedTensorType, ArrayRef<OpFoldResult>, Value,
                                Workspace &);
LogicalResult lowerRank2Store(OpBuilder &, Operation *, Value, Value,
                              ArrayRef<OpFoldResult>, Workspace &);

constexpr StringLiteral kDdrReadBegin = "bosc_ame_ddr_read_begin";
constexpr StringLiteral kDdrWriteBegin = "bosc_ame_ddr_write_begin";

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

LogicalResult emitCpuDdrPolicies(ModuleOp module) {
  if (!module->hasAttr(kCpuDdrPolicy))
    return success();
  if (failed(declareDdrPolicies(module)))
    return failure();
  // Expand ordinary copies so each real read and write receives its policy.
  // An opaque memcpy call cannot express a policy switch between its accesses.
  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp op) { copies.push_back(op); });
  for (memref::CopyOp copy : copies) {
    OpBuilder builder(copy);
    Location loc = copy.getLoc();
    auto type = cast<MemRefType>(copy.getSource().getType());
    SmallVector<Value> indices;
    auto build = [&](auto &&self, int64_t dim) -> void {
      if (dim == type.getRank()) {
        Value value =
            memref::LoadOp::create(builder, loc, copy.getSource(), indices);
        memref::StoreOp::create(builder, loc, value, copy.getTarget(), indices);
        return;
      }
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      Value end =
          type.isDynamicDim(dim)
              ? memref::DimOp::create(builder, loc, copy.getSource(), dim)
                    .getResult()
              : arith::ConstantIndexOp::create(builder, loc,
                                               type.getDimSize(dim))
                    .getResult();
      Value step = arith::ConstantIndexOp::create(builder, loc, 1);
      auto loop = scf::ForOp::create(builder, loc, zero, end, step);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
      self(self, dim + 1);
      indices.pop_back();
    };
    build(build, 0);
    copy.erase();
  }
  module.walk([&](Operation *op) {
    Value memory;
    bool write = false;
    if (auto load = dyn_cast<memref::LoadOp>(op))
      memory = load.getMemref();
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      memory = store.getMemref();
      write = true;
    }
    if (auto load = dyn_cast<vector::LoadOp>(op))
      memory = load.getBase();
    if (auto store = dyn_cast<vector::StoreOp>(op)) {
      memory = store.getBase();
      write = true;
    }
    if (memory && !getSpmBank(memory.getType())) {
      OpBuilder builder(op);
      // A policy call between an SPM vector load and its DDR store would
      // force a live RVV value across the call. Select the write policy before
      // the adjacent SPM load, which does not itself use the DDR cache.
      if (auto store = dyn_cast<vector::StoreOp>(op))
        if (auto load = store.getValueToStore().getDefiningOp<vector::LoadOp>())
          if (getSpmBank(load.getBase().getType()) && load->getNextNode() == op)
            builder.setInsertionPoint(load);
      emitDdrPolicy(builder, op->getLoc(), write);
    }
  });
  return success();
}

FailureOr<Value> Workspace::allocate(OpBuilder &builder, Operation *anchor,
                                     MemRefType type, int64_t preferredBank) {
  auto function = anchor->getParentOfType<func::FuncOp>();
  auto size = getContiguousZeroOffsetByteSize(type);
  if (!function || !size || *size <= 0) {
    anchor->emitError(
        "BOSCAME workspace needs a static contiguous buffer in a function");
    return failure();
  }
  auto [entry, inserted] =
      used.try_emplace(function, std::array<int64_t, 2>{0, 0});
  if (inserted)
    function.walk([&](SpmViewOp view) {
      auto bytes =
          getContiguousZeroOffsetByteSize(cast<MemRefType>(view.getType()));
      if (bytes)
        entry->second[view.getBank()] = std::max(
            entry->second[view.getBank()], int64_t(view.getOffset()) + *bytes);
    });
  auto &budget = entry->second;
  int64_t bank = preferredBank;
  if (bank < 0) {
    bank = budget[0] <= budget[1] ? 0 : 1;
    if (int64_t(llvm::alignTo(budget[bank], kDmaAlignmentBytes)) >
        kSpmBankCapacityBytes - *size)
      bank = 1 - bank;
  }
  int64_t offset = llvm::alignTo(budget[bank], kDmaAlignmentBytes);
  if (*size > kSpmBankCapacityBytes || offset > kSpmBankCapacityBytes - *size) {
    anchor->emitError() << "BOSCAME SPM capacity exceeded: " << *size
                        << " bytes requested in bank " << bank << " at offset "
                        << offset;
    return failure();
  }
  budget[bank] = offset + *size;
  auto spmType =
      MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(),
                      SPMAttr::get(builder.getContext(), bank));
  return SpmViewOp::create(builder, anchor->getLoc(), spmType,
                           builder.getI32IntegerAttr(bank),
                           builder.getI64IntegerAttr(offset))
      .getView();
}

void emitDmaAndWait(OpBuilder &builder, Location loc, Value src, Value dst,
                    int64_t bytes) {
  Value length = arith::ConstantIndexOp::create(builder, loc, bytes);
  auto start = DmaStartOp::create(
      builder, loc, DMATokenType::get(builder.getContext(), 0), src, dst,
      length, builder.getI32IntegerAttr(0));
  DmaWaitOp::create(builder, loc, start.getToken());
}

LogicalResult copyVectors(OpBuilder &builder, Location loc, Value src,
                          Value dst, int64_t elementCount) {
  auto source = cast<MemRefType>(src.getType());
  auto destination = cast<MemRefType>(dst.getType());
  unsigned width = getSpmElementByteWidth(source.getElementType());
  if (!width || source.getRank() != 1 || destination.getRank() != 1 ||
      source.getElementType() != destination.getElementType() ||
      elementCount <= 0 || elementCount % (kDmaAlignmentBytes / width))
    return emitError(loc, "BOSCAME vector copy requires complete 64 B blocks");
  int64_t lanes = kDmaAlignmentBytes / width;
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value end = arith::ConstantIndexOp::create(builder, loc, elementCount);
  Value step = arith::ConstantIndexOp::create(builder, loc, lanes);
  auto loop = scf::ForOp::create(builder, loc, zero, end, step);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  auto vectorType = VectorType::get({lanes}, source.getElementType());
  Value vector = vector::LoadOp::create(builder, loc, vectorType, src,
                                        ValueRange{loop.getInductionVar()});
  vector::StoreOp::create(builder, loc, vector, dst,
                          ValueRange{loop.getInductionVar()});
  return success();
}

static FailureOr<int64_t> getRank1Count(Operation *anchor, Value ptr,
                                        RankedTensorType type,
                                        ArrayRef<OpFoldResult> mask) {
  auto memref = dyn_cast<MemRefType>(ptr.getType());
  if (!memref || type.getRank() != 1 || !type.hasStaticShape() ||
      memref.getShape() != type.getShape() ||
      memref.getElementType() != type.getElementType() ||
      !getDmaEndpoint(memref)) {
    anchor->emitError(
        "BOSCAME access requires a static rank-1 contiguous memref; "
        "gather/scatter and split pointers are unsupported");
    return failure();
  }
  auto count = mask.empty() ? std::optional<int64_t>(type.getDimSize(0))
                            : getConstantIntValue(mask[0]);
  unsigned width = getSpmElementByteWidth(type.getElementType());
  if (!count || *count <= 0 || *count > type.getDimSize(0) ||
      *count * width % kDmaAlignmentBytes != 0) {
    anchor->emitError(
        "BOSCAME access requires a positive 64 B multiple of authorized DDR "
        "bytes; short or dynamic DMA ranges are unsupported");
    return failure();
  }
  return *count;
}

static Value rank1Prefix(OpBuilder &builder, Location loc, Value ptr,
                         int64_t count) {
  SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(count)};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
  return memref::SubViewOp::create(builder, loc, ptr, offsets, sizes, strides);
}

FailureOr<Value> lowerLoad(OpBuilder &builder, Operation *anchor, Value ptr,
                           RankedTensorType type, ArrayRef<OpFoldResult> mask,
                           Value other, Workspace &workspace) {
  if (type.getRank() == 2)
    return lowerRank2Load(builder, anchor, ptr, type, mask, other, workspace);
  auto count = getRank1Count(anchor, ptr, type, mask);
  if (failed(count))
    return failure();
  Location loc = anchor->getLoc();
  Value ddr = rank1Prefix(builder, loc, ptr, *count);
  auto spm = workspace.allocate(
      builder, anchor, MemRefType::get({*count}, type.getElementType()));
  if (failed(spm))
    return failure();
  auto ordinaryType = MemRefType::get(type.getShape(), type.getElementType());
  Value ordinary = memref::AllocOp::create(builder, loc, ordinaryType,
                                           builder.getI64IntegerAttr(64));
  if (other)
    linalg::FillOp::create(builder, loc, ValueRange{other},
                           ValueRange{ordinary});
  emitDmaAndWait(builder, loc, ddr, *spm,
                 *count * getSpmElementByteWidth(type.getElementType()));
  // The CPU reads the newly arrived SPM data and writes the ordinary buffer.
  emitDdrPolicy(builder, loc, /*write=*/false);
  emitDdrPolicy(builder, loc, /*write=*/true);
  if (failed(copyVectors(builder, loc, *spm, ordinary, *count)))
    return failure();
  return bufferization::ToTensorOp::create(builder, loc, type, ordinary, true,
                                           true)
      .getResult();
}

LogicalResult lowerStore(OpBuilder &builder, Operation *anchor, Value ptr,
                         Value tensor, ArrayRef<OpFoldResult> mask,
                         Workspace &workspace) {
  auto type = dyn_cast<RankedTensorType>(tensor.getType());
  if (!type)
    return anchor->emitError("BOSCAME store expects a tensor result");
  if (type.getRank() == 2)
    return lowerRank2Store(builder, anchor, ptr, tensor, mask, workspace);
  auto count = getRank1Count(anchor, ptr, type, mask);
  if (failed(count))
    return failure();
  Location loc = anchor->getLoc();
  Value ddr = rank1Prefix(builder, loc, ptr, *count);
  if (auto source = tensor.getDefiningOp<bufferization::ToTensorOp>()) {
    auto sourceType = dyn_cast<MemRefType>(source.getBuffer().getType());
    if (sourceType && getSpmBank(sourceType)) {
      if (!getDmaEndpoint(sourceType))
        return anchor->emitError(
            "BOSCAME direct SPM store needs a static contiguous source");
      Value spm = rank1Prefix(builder, loc, source.getBuffer(), *count);
      emitDmaAndWait(builder, loc, spm, ddr,
                     *count * getSpmElementByteWidth(type.getElementType()));
      return success();
    }
  }
  auto spm = workspace.allocate(
      builder, anchor, MemRefType::get({*count}, type.getElementType()));
  if (failed(spm))
    return failure();
  auto ordinaryType = MemRefType::get(type.getShape(), type.getElementType());
  Value ordinary =
      bufferization::ToBufferOp::create(builder, loc, ordinaryType, tensor);
  emitDdrPolicy(builder, loc, /*write=*/false);
  if (failed(copyVectors(builder, loc, ordinary, *spm, *count)))
    return failure();
  emitDmaAndWait(builder, loc, *spm, ddr,
                 *count * getSpmElementByteWidth(type.getElementType()));
  return success();
}

LogicalResult verifyExit(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    StringRef dialect = op->getName().getDialectNamespace();
    bool tensor =
        llvm::any_of(op->getOperandTypes(),
                     [](Type type) { return isa<TensorType>(type); }) ||
        llvm::any_of(op->getResultTypes(),
                     [](Type type) { return isa<TensorType>(type); });
    for (Region &region : op->getRegions())
      for (Block &block : region)
        tensor |= llvm::any_of(block.getArgumentTypes(),
                               [](Type type) { return isa<TensorType>(type); });
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
      auto input = dyn_cast<ShapedType>(matmul.getInputs()[0].getType());
      auto output = dyn_cast<ShapedType>(matmul.getOutputs()[0].getType());
      if (input && output && input.getElementType().isInteger(8) &&
          output.getElementType().isInteger(32)) {
        op->emitError("incomplete BOSCAME lowering: integer target matmul was "
                      "not lowered to AME");
        return WalkResult::interrupt();
      }
    }
    if (dialect == "tt" || dialect == "tts" || tensor ||
        dialect == "bufferization" || isa<UnrealizedConversionCastOp>(op) ||
        op->hasAttr(kLowerToAme)) {
      op->emitError("incomplete BOSCAME lowering: residual "
                    "Triton/tensor/bridge or target matmul");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

} // namespace buddy::boscame::lowering
