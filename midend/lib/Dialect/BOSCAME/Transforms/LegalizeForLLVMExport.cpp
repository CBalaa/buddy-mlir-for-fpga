//===- LegalizeForLLVMExport.cpp - Unified BOSCAME/SPM/DMA LLVM lowering -===//
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
// `-lower-bosc-ame` is the single BOSCAME lowering entry point.  It owns the
// whole AME / SPM / DMA surface and finishes the module with the standard
// MemRef/Func/Arith/CF/SCF/Vector/Math lowering so that its output is ready
// for `buddy-translate --buddy-to-llvmir`.
//
// The pass runs in four stages:
//
//   1. Checks.  The cross-operation SPM/DMA contract (token lifetime, channel
//      occupancy, aliasing) and the SPM-typed-value rules are verified while
//      the `#bosc_ame.spm<bank>` type annotation is still present.  A module
//      that cannot be lowered is rejected before any type information is lost.
//
//   2. Preparation.  Converter-agnostic rewrites that the standard lowering
//      expects to have run first: linalg to loops, strided-metadata
//      expansion, memref expansion and vector-to-SCF.  They only rewrite
//      operations this pass sees through, never BOSCAME operations.
//
//   3. SCF structure.  Loop-carried BOSCAME matrix values (`vector<4x4xT>`)
//      are converted to the scalable register type with the same type
//      converter used by the final conversion.  Without this an `scf.for`
//      accumulator chain would survive with a fixed vector type and a
//      permanent `unrealized_conversion_cast` bridge.
//
//   4. Conversion.  A single LLVM conversion with one `LLVMTypeConverter`
//      lowers, in one go:
//
//        * `bosc_ame.spm.view` to an LLVM memref descriptor whose base is the
//          fixed bank base address plus the view offset.  The bank is
//          compiled into the address and never has to be recovered, so no
//          workspace root argument is needed and the kernel ABI stays clean.
//          `#bosc_ame.spm<bank>` maps to LLVM address space 0: the two banks
//          are two windows in one address space, not two address spaces.
//        * `bosc_ame.dma.start` / `bosc_ame.dma.wait` to calls into the
//          runtime ABI.  The DMA token becomes the `i64` returned by the
//          start call (the effective source address), which the matching
//          wait hands back to the runtime as its completion barrier.  The
//          wait call returns void and the status check lives in the runtime,
//          so the operation is simply erased.
//        * AME matrix load/store to `bosc_ame.intr.*` intrinsics whose base
//          pointer is decoded from the converted memref descriptor
//          (`alignedPtr + offset * elementBytes`), so nested subviews and
//          strided layouts are folded in exactly once.
//        * the remaining BOSCAME operations to their `bosc_ame.intr.*`
//          counterparts, keeping the existing scalable-vector mapping.
//        * MemRef, Func, Arith, CF, SCF, Vector, Math, Index and UB to the
//          LLVM dialect.
//
//      Leftover `unrealized_conversion_cast` operations are reconciled at the
//      end, mirroring the `-reconcile-unrealized-casts` stage of the shipped
//      pipeline.
//
//===----------------------------------------------------------------------===//

#include "Dialect/BOSCAME/BOSCAMEAttrs.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/BOSCAMETypes.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "Dialect/BOSCAME/Transform.h"
#include "Dialect/BOSCAME/Transforms/BOSCAMETileLowering.h"

#include "BOSCAMEConversionUtils.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <string>

using namespace mlir;
using namespace buddy::boscame;
using namespace buddy::boscame::spmdma;

namespace {

//===----------------------------------------------------------------------===//
// BOSCAME matrix register types
//===----------------------------------------------------------------------===//

// The scalable register type a matrix value is lowered to, and the descriptor
// decoding helpers, are shared with the tile lowering; see
// `BOSCAMEConversionUtils.h`.
using buddy::boscame::detail::createBOSCAMEIntrinsicOp;
using buddy::boscame::detail::extractPointerFromDescriptor;
using buddy::boscame::detail::getBOSCAMERegisterType;

//===----------------------------------------------------------------------===//
// SPM storage space
//===----------------------------------------------------------------------===//

/// True when `type` is a ranked memref in an SPM memory space.
bool hasSpmSpace(Type type) {
  auto memref = dyn_cast_or_null<MemRefType>(type);
  return memref && isa_and_nonnull<SPMAttr>(memref.getMemorySpace());
}

/// The fixed hardware base address of `bank`, as a byte address.
uint64_t getBankBaseAddress(int64_t bank) {
  assert(bank >= 0 && bank < kSpmBankCount && "bank out of range");
  return kSpmBankBaseAddress[bank];
}

/// Operations whose result may legitimately keep an SPM memory space: they
/// only re-describe a buffer `bosc_ame.spm.view` already anchored in a bank
/// and never change the base address.  Anything else would need an SPM
/// allocation the hardware does not hand out and is rejected.
bool preservesSpmSpace(Operation *op) {
  return isa<SpmViewOp, memref::ViewOp, memref::SubViewOp, memref::CastOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp,
             memref::TransposeOp, memref::ReinterpretCastOp,
             memref::MemorySpaceCastOp, memref::AssumeAlignmentOp,
             memref::ExtractStridedMetadataOp, arith::SelectOp, scf::IfOp,
             scf::ForOp, scf::WhileOp, scf::IndexSwitchOp>(op);
}

/// Reject SPM-typed values this pass cannot give an address to: an SPM-typed
/// function argument has no statically known position inside its bank, and an
/// SPM-typed value produced by an operation other than a view of an existing
/// SPM buffer has no address at all.  SPM memory-space casts that change the
/// bank are rejected too.
LogicalResult validateSpmSurface(ModuleOp module) {
  bool ok = true;
  module.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (!hasSpmSpace(result.getType()) || preservesSpmSpace(op))
        continue;
      op->emitOpError()
          << "produces an SPM-typed value, but only `bosc_ame.spm.view` and "
             "memref view-like operations may carry `"
          << SPMAttr::getMnemonic()
          << "`; this value has no SPM address and cannot be lowered";
      ok = false;
    }
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(op)) {
      std::optional<int64_t> sourceBank =
          getSpmBank(cast.getSource().getType());
      std::optional<int64_t> destBank = getSpmBank(cast.getDest().getType());
      if (sourceBank && destBank && *sourceBank != *destBank) {
        cast.emitOpError() << "cannot reinterpret an SPM buffer in bank "
                           << *sourceBank << " as bank " << *destBank;
        ok = false;
      }
    }
    // SPM is a fixed hardware workspace: it is addressed through
    // `bosc_ame.spm.view`, which compiles the bank base into the address.  A
    // kernel that receives an SPM memref argument would need a runtime base
    // address the ABI no longer carries.
    if (auto function = dyn_cast<FunctionOpInterface>(op)) {
      for (BlockArgument argument : function.getArguments()) {
        if (!hasSpmSpace(argument.getType()))
          continue;
        function->emitOpError()
            << "takes an SPM memref argument " << argument.getType()
            << "; SPM is addressed through `bosc_ame.spm.view` at the fixed "
               "bank base address and is not an ABI parameter";
        ok = false;
      }
    }
  });
  return success(ok);
}

//===----------------------------------------------------------------------===//
// DMA runtime declarations
//===----------------------------------------------------------------------===//

/// Declare the two runtime entry points the DMA lowering calls, but only when
/// the module actually contains a request.  A module without DMA must not gain
/// two external declarations.
void declareDmaRuntimeHelpers(ModuleOp module) {
  bool needed = false;
  module.walk([&](Operation *op) { needed |= isa<DmaStartOp, DmaWaitOp>(op); });
  if (!needed)
    return;

  MLIRContext *context = module.getContext();
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  Type voidType = LLVM::LLVMVoidType::get(context);
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(kDmaStartHelperName))
    LLVM::LLVMFuncOp::create(builder, module.getLoc(), kDmaStartHelperName,
                             LLVM::LLVMFunctionType::get(
                                 i64Type, {i32Type, i64Type, i64Type, i64Type}),
                             LLVM::Linkage::External, /*dsoLocal=*/false,
                             LLVM::CConv::C);
  // The wait helper checks the hardware status itself: a normal return means
  // "completed and succeeded", a failure takes the runtime error path.
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(kDmaWaitHelperName))
    LLVM::LLVMFuncOp::create(
        builder, module.getLoc(), kDmaWaitHelperName,
        LLVM::LLVMFunctionType::get(voidType, {i32Type, i64Type}),
        LLVM::Linkage::External, /*dsoLocal=*/false, LLVM::CConv::C);
}

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

static void addBOSCAMETypeConversions(LLVMTypeConverter &converter) {
  // LLVMTypeConverter normally lowers a fixed 2-D vector to nested LLVM
  // arrays.  BOSCAME matrix registers are represented by scalable vectors in
  // the backend, so keep those values in the register form throughout this
  // conversion instead of materializing unrealized array casts.
  converter.addConversion([](VectorType type) -> Type {
    if (type.getRank() != 2)
      return Type();
    return getBOSCAMERegisterType(type);
  });

  // A DMA token is the effective source address of the request, i.e. an i64.
  converter.addConversion([](DMATokenType type) -> Type {
    return IntegerType::get(type.getContext(), 64);
  });

  // Both SPM banks are windows into the same address space, so the memory
  // space annotation lowers to address space 0.  Mapping bank 0/1 to LLVM
  // address spaces 0/1 would split one physical space into two.
  converter.addTypeAttributeConversion(
      [](MemRefType type,
         SPMAttr attr) -> TypeConverter::AttributeConversionResult {
        return IntegerAttr::get(IntegerType::get(attr.getContext(), 64), 0);
      });
}

//===----------------------------------------------------------------------===//
// Conversion helpers
//===----------------------------------------------------------------------===//

static bool isMemRefValue(Value value) {
  return isa<MemRefType, UnrankedMemRefType>(value.getType());
}

/// Materialize the effective byte address of a converted memref descriptor as
/// an `i64`, which is what the DMA runtime ABI takes.
static Value extractAddressFromDescriptor(OpBuilder &builder, Location loc,
                                          Value descriptor,
                                          Type originalMemRefType,
                                          const LLVMTypeConverter &converter) {
  Value pointer = extractPointerFromDescriptor(builder, loc, descriptor,
                                               originalMemRefType, converter);
  if (!pointer)
    return Value();
  return LLVM::PtrToIntOp::create(
      builder, loc, IntegerType::get(builder.getContext(), 64), pointer);
}

// Immediate BOSCAME operations carry an I64Attr. LLVM_IntrOpBase needs the
// same immediate represented as an SSA i64 constant on the bridge op.
static Value materializeBOSCAMEImmediate(Operation *op,
                                         ConversionPatternRewriter &rewriter) {
  static const char *const attributeNames[] = {"timm", "tilem", "tilen",
                                               "tilek", "imm"};
  for (const char *name : attributeNames) {
    IntegerAttr attribute = op->getAttrOfType<IntegerAttr>(name);
    if (!attribute)
      continue;
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    return LLVM::ConstantOp::create(
        rewriter, op->getLoc(), i64Type,
        rewriter.getI64IntegerAttr(attribute.getInt()));
  }
  return Value();
}

//===----------------------------------------------------------------------===//
// BOSCAME -> bosc_ame.intr.*
//===----------------------------------------------------------------------===//

// The old pass had one typed template and one patterns.add<> call per
// instruction. Their bodies were identical except for operation name and
// register-number attributes. The SSA ODS no longer has those attributes, so
// this generic conversion is both smaller and semantically correct.
class BOSCAMEToIntrinsicLowering : public ConversionPattern {
public:
  BOSCAMEToIntrinsicLowering(const TypeConverter &converter,
                             MLIRContext *context)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(),
                          PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();
    if (!name.starts_with("bosc_ame.") || name.starts_with("bosc_ame.intr."))
      return failure();

    // The SPM view and the DMA requests have dedicated conversion patterns
    // with a higher benefit; never fabricate an intrinsic for them.
    if (isa<SpmViewOp, DmaStartOp, DmaWaitOp>(op))
      return failure();

    // The tile operations are handled by `BOSCAMETileLowering.cpp`, which
    // derives the hardware configuration and the physical stride from the
    // operand types.  The generic pattern must not turn them into a
    // `bosc_ame.intr.tile.*` operation: those do not exist, and even if they
    // did, the naming convention of this pattern cannot express the slot, the
    // stride or the configuration sequence.
    if (isa<TileLoadAOp, TileLoadBOp, TileLoadAccOp, TileMmaOp, TileStoreAccOp>(
            op))
      return failure();

    const auto &converter =
        *static_cast<const LLVMTypeConverter *>(typeConverter);
    Location loc = op->getLoc();

    // Matrix load: memref, i64 stride -> scalable LLVM vector.
    if (op->getNumResults() == 1 && op->getNumOperands() == 2 &&
        isMemRefValue(op->getOperand(0))) {
      Type resultType = getBOSCAMERegisterType(op->getResult(0).getType());
      if (!resultType)
        return failure();
      Value base = extractPointerFromDescriptor(
          rewriter, loc, operands[0], op->getOperand(0).getType(), converter);
      if (!base)
        return failure();
      SmallVector<Value> loadOperands{base, operands[1]};
      Operation *intrinsic = createBOSCAMEIntrinsicOp(
          rewriter, loc, name, TypeRange{resultType}, loadOperands);
      rewriter.replaceOp(op, intrinsic->getResults());
      return success();
    }

    // Matrix store: scalable LLVM vector, memref, i64 stride -> no result.
    if (op->getNumResults() == 0 && op->getNumOperands() == 3 &&
        isMemRefValue(op->getOperand(1))) {
      Value base = extractPointerFromDescriptor(
          rewriter, loc, operands[1], op->getOperand(1).getType(), converter);
      if (!base)
        return failure();
      createBOSCAMEIntrinsicOp(rewriter, loc, name, TypeRange(),
                               ValueRange{operands[0], base, operands[2]});
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Value, 4> intrinsicOperands(operands.begin(), operands.end());
    if (Value immediate = materializeBOSCAMEImmediate(op, rewriter))
      intrinsicOperands.push_back(immediate);

    SmallVector<Type, 1> resultTypes;
    for (Value result : op->getResults()) {
      Type resultType = result.getType();
      if (isa<VectorType>(resultType))
        resultType = getBOSCAMERegisterType(resultType);
      if (!resultType)
        return failure();
      resultTypes.push_back(resultType);
    }

    Operation *intrinsic = createBOSCAMEIntrinsicOp(
        rewriter, loc, name, TypeRange(resultTypes), intrinsicOperands);
    rewriter.replaceOp(op, intrinsic->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// bosc_ame.spm.view -> LLVM memref descriptor at the fixed bank base
//===----------------------------------------------------------------------===//

/// `bosc_ame.spm.view {bank, offset}` becomes an LLVM memref descriptor whose
/// base is `bankBase[bank] + offset`.  The bank is a compile-time hardware
/// constant, so no root argument, no `memref.view` and no runtime-provided
/// base pointer is involved.  The descriptor offset is zero (the verifier
/// guarantees a contiguous, zero-offset view), so later address computations
/// add the descriptor offset exactly once.
class SpmViewOpConversion : public ConvertToLLVMPattern {
public:
  SpmViewOpConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(SpmViewOp::getOperationName(), context, converter,
                             PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto view = cast<SpmViewOp>(op);
    auto memrefType = dyn_cast<MemRefType>(view.getView().getType());
    if (!memrefType)
      return failure();
    int64_t bank = static_cast<int64_t>(view.getBank());
    if (bank < 0 || bank >= kSpmBankCount)
      return failure();

    Location loc = view.getLoc();
    uint64_t address = getBankBaseAddress(bank) + view.getOffset();
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    Value addressValue = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(address)));
    Type pointerType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), /*addressSpace=*/0);
    Value base =
        LLVM::IntToPtrOp::create(rewriter, loc, pointerType, addressValue);
    Value descriptor = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), memrefType, base);
    rewriter.replaceOp(op, descriptor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// bosc_ame.dma.start / bosc_ame.dma.wait -> runtime calls
//===----------------------------------------------------------------------===//

/// `bosc_ame.dma.start` becomes a call to `bosc_ame_dma_start(channel, src,
/// dst, len)`.  The runtime returns the effective source address, which *is*
/// the request token, so the call result replaces the token value directly.
class DmaStartOpConversion : public ConvertToLLVMPattern {
public:
  DmaStartOpConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(DmaStartOp::getOperationName(), context, converter,
                             PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto start = cast<DmaStartOp>(op);
    if (!getDmaEndpoint(dyn_cast<MemRefType>(start.getSrc().getType())) ||
        !getDmaEndpoint(dyn_cast<MemRefType>(start.getDst().getType())))
      return start.emitOpError()
             << "DMA endpoints must be 1-D i8 memrefs with a static length";

    Location loc = start.getLoc();
    const auto &converter =
        *static_cast<const LLVMTypeConverter *>(getTypeConverter());
    Value channel = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(start.getChannel())));
    Value src = extractAddressFromDescriptor(
        rewriter, loc, operands[0], start.getSrc().getType(), converter);
    Value dst = extractAddressFromDescriptor(
        rewriter, loc, operands[1], start.getDst().getType(), converter);
    if (!src || !dst)
      return failure();
    // `bytes` is an index operand; the converter has already turned it into the
    // LLVM index type (i64), which is the runtime's length type as well.
    Value length = operands[2];

    FlatSymbolRefAttr callee =
        FlatSymbolRefAttr::get(rewriter.getContext(), kDmaStartHelperName);
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Type}, callee,
                                     ValueRange{channel, src, dst, length});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// `bosc_ame.dma.wait` becomes a call to `bosc_ame_dma_wait(channel, token)`.
/// The channel comes from the *original* token type (the token has already
/// been converted to the start call's `i64` result), and the converted token
/// is the effective source address the runtime uses as the completion
/// barrier.  The helper returns void and owns the status check, so the wait
/// operation itself is erased.  No start/wait bookkeeping map is needed.
class DmaWaitOpConversion : public ConvertToLLVMPattern {
public:
  DmaWaitOpConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(DmaWaitOp::getOperationName(), context, converter,
                             PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto wait = cast<DmaWaitOp>(op);
    auto tokenType = dyn_cast<DMATokenType>(wait.getToken().getType());
    if (!tokenType)
      return wait.emitOpError() << "operand must be a `"
                                << DMATokenType::getMnemonic() << "` value";

    Location loc = wait.getLoc();
    Value channel = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(
            static_cast<int32_t>(tokenType.getChannel())));
    FlatSymbolRefAttr callee =
        FlatSymbolRefAttr::get(rewriter.getContext(), kDmaWaitHelperName);
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, callee,
                         ValueRange{channel, operands[0]});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Preparation stages
//===----------------------------------------------------------------------===//

/// Converter-agnostic preparation the standard lowering expects to have run:
/// linalg to loops (there is no linalg-to-LLVM lowering), strided-metadata
/// expansion, memref expansion and vector-to-SCF.  None of them looks at
/// BOSCAME operations.
LogicalResult runPreparationPipeline(ModuleOp module, MLIRContext &context) {
  PassManager pm(&context, PassManager::getAnyOpAnchorName(),
                 PassManager::Nesting::Implicit);
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createConvertMathToLibmPass());
  return pm.run(module);
}

/// Rewrite vector operations into the forms the LLVM conversion understands.
/// This mirrors the `-convert-vector-to-llvm` pipeline (default options) that
/// the shipped Makefile runs after the BOSCAME lowering.
void prepareVectorOps(ModuleOp module, MLIRContext &context) {
  RewritePatternSet patterns(&context);
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorBitCastLoweringPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vector::VectorContractLowering::Dot);
  vector::populateVectorMaskOpLoweringPatterns(patterns);
  vector::populateVectorShapeCastLoweringPatterns(patterns);
  vector::populateVectorInterleaveLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransposeLowering::EltWise);
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);
  vector::populateVectorMaskMaterializationPatterns(
      patterns, /*force32BitVectorIndices=*/false);
  vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
  vector::populateVectorRankReducingFMAPattern(patterns);
  vector::populateVectorGatherLoweringPatterns(patterns);
  vector::populateVectorFromElementsUnrollPatterns(patterns);
  vector::populateVectorToElementsUnrollPatterns(patterns);
  (void)applyPatternsGreedily(module, std::move(patterns));
}

/// The final conversion: BOSCAME / SPM / DMA / tile plus every standard dialect
/// that still stands between the module and LLVM.
LogicalResult convertToLLVM(ModuleOp module, MLIRContext &context,
                            const TileLoweringOptions &tileOptions) {
  DataLayoutAnalysis dataLayoutAnalysis(module);
  LowerToLLVMOptions options(&context, dataLayoutAnalysis.getAtOrAbove(module));
  LLVMTypeConverter converter(&context, options, &dataLayoutAnalysis);
  addBOSCAMETypeConversions(converter);

  LLVMConversionTarget target(context);
  target.addDynamicallyLegalDialect<BOSCAMEDialect>([](Operation *op) {
    return op->getName().getStringRef().starts_with("bosc_ame.intr.");
  });
  target.addIllegalDialect<
      arith::ArithDialect, affine::AffineDialect, cf::ControlFlowDialect,
      func::FuncDialect, index::IndexDialect, linalg::LinalgDialect,
      math::MathDialect, memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
      vector::VectorDialect>();

  SymbolTableCollection symbolTables;
  RewritePatternSet patterns(&context);
  patterns.add<BOSCAMEToIntrinsicLowering>(converter, &context);
  patterns.add<SpmViewOpConversion>(converter, &context);
  patterns.add<DmaStartOpConversion>(converter, &context);
  patterns.add<DmaWaitOpConversion>(converter, &context);
  // The tile patterns share this converter, this target and this pattern set;
  // they do not run a conversion of their own.
  populateBOSCAMETileLoweringPatterns(converter, patterns, tileOptions);

  // A loop that carries an AME matrix value changes the type of its iter
  // arguments.  The structural SCF conversion has to run inside this same
  // conversion - and before SCF is lowered to a CFG - so that the loop itself,
  // the value it carries and the tile operations that produce and consume that
  // value are all rewritten against one converter; a separate type-conversion
  // pass would have to bridge the two conversions with casts that could not be
  // reconciled afterwards.
  scf::populateSCFStructuralTypeConversions(converter, patterns,
                                            PatternBenefit(10));
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  index::populateIndexToLLVMConversionPatterns(converter, patterns);
  ub::populateUBToLLVMConversionPatterns(converter, patterns);
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns, &symbolTables);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns,
                                                 &symbolTables);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  vector::populateVectorTransferLoweringPatterns(patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);

  return applyPartialConversion(module, target, std::move(patterns));
}

/// Remove the unrealized conversion casts the conversion may have left behind
/// (the equivalent of the final `-reconcile-unrealized-casts` stage).
void reconcileCasts(ModuleOp module) {
  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  reconcileUnrealizedCasts(casts);
}

//===----------------------------------------------------------------------===//
// Pass body
//===----------------------------------------------------------------------===//

/// Options of the shared lowering body.
struct BOSCAMELoweringOptions {
  /// `-lower-bosc-ame-tile`: the caller states that this pipeline is the FPGA
  /// tile path, so the module has to select an FPGA AME contract even when the
  /// tile operations are not present yet (they may be introduced earlier in the
  /// same pipeline).
  bool requireFpgaContract = false;
};

/// The single lowering body behind both `-lower-bosc-ame` and
/// `-lower-bosc-ame-tile`; the two differ only in which contract they demand
/// from the module, never in how they convert.
LogicalResult runBOSCAMELowering(ModuleOp module, MLIRContext &context,
                                 const BOSCAMELoweringOptions &options) {
  // Stage 1: contract checks, while the SPM annotation is still present.
  if (failed(verifySpmDmaContract(module)))
    return failure();
  if (failed(validateSpmSurface(module)))
    return failure();

  // The tile path is opt-in: it is only taken when the module both uses tile
  // operations and selects a hardware contract this build implements.
  TileLoweringOptions tileOptions;
  if (hasTileOps(module) || options.requireFpgaContract) {
    TileTargetProfile profile;
    if (failed(resolveTileTargetProfile(module, profile)))
      return failure();
    tileOptions.profile = profile;
    if (failed(verifyBOSCAMETileValues(module)))
      return failure();
  }
  declareDmaRuntimeHelpers(module);

  // Stage 2: preparation.
  if (failed(runPreparationPipeline(module, context)))
    return failure();
  prepareVectorOps(module, context);

  // Stage 3: the loop structure follows automatically from the type conversion
  // of stage 4 (see `convertToLLVM`), which converts the matrix value type and
  // the operations that carry it - the loop included - in one conversion, so
  // that no fixed-vector bridge can survive between them.

  // Stage 4: the unified BOSCAME / SPM / DMA / standard conversion.
  if (failed(convertToLLVM(module, context, tileOptions)))
    return failure();
  reconcileCasts(module);
  return success();
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// `-lower-bosc-ame`: the unified BOSCAME / SPM / DMA / tile entry point.
struct LegalizeBOSCAMEForLLVMExport
    : public PassWrapper<LegalizeBOSCAMEForLLVMExport,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeBOSCAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-bosc-ame"; }
  StringRef getDescription() const final {
    return "Lower the BOSCAME / SPM / DMA surface and the surrounding standard "
           "dialects to the LLVM dialect.";
  }

  LegalizeBOSCAMEForLLVMExport() = default;
  LegalizeBOSCAMEForLLVMExport(const LegalizeBOSCAMEForLLVMExport &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BOSCAMEDialect, LLVM::LLVMDialect, affine::AffineDialect,
                arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                index::IndexDialect, linalg::LinalgDialect, math::MathDialect,
                memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    if (failed(runBOSCAMELowering(getOperation(), getContext(), {})))
      signalPassFailure();
  }
};

/// `-lower-bosc-ame-tile`: the FPGA tile path of the same lowering.
///
/// It produces exactly the same boundary as `-lower-bosc-ame` (LLVM dialect,
/// ready for `buddy-translate`) and shares its implementation: the separate
/// entry point only exists so a pipeline can state that it is lowering the
/// FPGA AME tile path and therefore require `bosc_ame.target` to name a
/// contract this build implements, even before the tile operations appear.
struct LowerBOSCAMETileOps
    : public PassWrapper<LowerBOSCAMETileOps, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBOSCAMETileOps)

  StringRef getArgument() const final { return "lower-bosc-ame-tile"; }
  StringRef getDescription() const final {
    return "Lower the BOSCAME tile operations (and the SPM/DMA and standard "
           "dialect surface they need) to the LLVM dialect on the FPGA target.";
  }

  LowerBOSCAMETileOps() = default;
  LowerBOSCAMETileOps(const LowerBOSCAMETileOps &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BOSCAMEDialect, LLVM::LLVMDialect, affine::AffineDialect,
                arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                index::IndexDialect, linalg::LinalgDialect, math::MathDialect,
                memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    BOSCAMELoweringOptions options;
    options.requireFpgaContract = true;
    if (failed(runBOSCAMELowering(getOperation(), getContext(), options)))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateBOSCAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  addBOSCAMETypeConversions(converter);
  patterns.add<BOSCAMEToIntrinsicLowering>(converter, patterns.getContext());
  // NOTE: callers of this helper that lower the SPM/DMA surface or
  // loop-carried matrix values must add the SPM view / DMA patterns and
  // `scf::populateSCFStructuralTypeConversionsAndLegality` themselves; see the
  // unified pass above.
}

void mlir::configureBOSCAMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addDynamicallyLegalDialect<BOSCAMEDialect>([](Operation *op) {
    return op->getName().getStringRef().starts_with("bosc_ame.intr.");
  });
}

std::unique_ptr<Pass> buddy::boscame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeBOSCAMEForLLVMExport>();
}

std::unique_ptr<Pass> buddy::boscame::createLowerBOSCAMETilePass() {
  return std::make_unique<LowerBOSCAMETileOps>();
}

namespace mlir {
namespace buddy {
void registerLowerBOSCAMEPass() {
  PassRegistration<LegalizeBOSCAMEForLLVMExport>();
}

void registerLowerBOSCAMETilePass() { PassRegistration<LowerBOSCAMETileOps>(); }
} // namespace buddy
} // namespace mlir
