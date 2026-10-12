# TPtr (`tptr`) -> memref lowering for `--triton-to-linalg-experimental`.
#
# Migrated from thirdparty/triton-riscv/lib/Conversion/TritonPtrToMemref
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonPtrToMemref
  TritonPtrToMemrefPass.cpp

  DEPENDS
  BuddyTritonPtrToMemrefPassIncGen

  LINK_LIBS PUBLIC
  BuddyTritonAnalysis
  BuddyTritonAnalysisStructured
  BuddyTritonIR
  BuddyTritonStructuredIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
)
