# The `--triton-to-linalg-experimental` driver plus its private helper passes
# (collapse-shape, reconcile-ptr-casts, triton-to-ptr).
#
# Migrated from
# thirdparty/triton-riscv/lib/Conversion/TritonToLinalgExperimental
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  ReconcilePtrCastsPass.cpp
  TritonToPtrPass.cpp
  CollapseShape.cpp

  DEPENDS
  BuddyTritonToLinalgExperimentalPassIncGen

  LINK_LIBS PUBLIC
  BuddyTPtrIR
  BuddyTritonAnalysis
  BuddyTritonAnalysisStructured
  BuddyTritonArithToLinalg
  BuddyTritonIR
  BuddyTritonPtrToMemref
  BuddyTritonStructuredIR
  BuddyTritonStructuredToMemref
  BuddyTritonToStructured
  BuddyTritonToUnstructured
  BuddyTritonUnstructuredToMemref
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrDialect
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorDialect
)
