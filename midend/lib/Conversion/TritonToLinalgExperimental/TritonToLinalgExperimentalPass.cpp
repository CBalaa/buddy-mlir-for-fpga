//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
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
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/Transform.h"
#include "Dialect/TPtr/IR/TPtrDialect.h"
#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DECL_TRITONTOLINALGEXPERIMENTAL
#include "Conversion/TritonToLinalgExperimental/Passes.h.inc"
#define GEN_PASS_DEF_TRITONTOLINALGEXPERIMENTAL
#include "Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

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
    if (boscame)
      registry.insert<buddy::boscame::BOSCAMEDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    TritonLoweringConfig config{boscame, dmaAbi};
    if (config.enableBOSCAME) {
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
      moduleOp->setAttr(buddy::boscame::lowering::kCpuDdrPolicy,
                        UnitAttr::get(&getContext()));
    }
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
    // Tensor-first fusion consumes loads/stores before DMA staging can see
    // them.
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
    if (!config.enableBOSCAME)
      return;
    WalkResult integerTargets = moduleOp.walk([&](linalg::MatmulOp matmul) {
      auto input = cast<ShapedType>(matmul.getInputs()[0].getType());
      auto output = cast<ShapedType>(matmul.getOutputs()[0].getType());
      if (input.getElementType().isInteger(8) &&
          output.getElementType().isInteger(32)) {
        matmul.emitError(
            "AME matmul lowering is not installed at this pipeline stage");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (integerTargets.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgExperimentalPass() {
  return std::make_unique<TritonToLinalgExperimentalPass>();
}
