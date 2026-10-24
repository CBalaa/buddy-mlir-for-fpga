# The `--triton-to-linalg-experimental` driver plus its private helper passes
# (collapse-shape, reconcile-ptr-casts, triton-to-ptr).
#
# Migrated from
# thirdparty/triton-riscv/lib/Conversion/TritonToLinalgExperimental
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  ContractTileSpecialization.cpp
  ReconcilePtrCastsPass.cpp
  TritonToPtrPass.cpp
  CollapseShape.cpp

  DEPENDS
  BuddyTritonToLinalgExperimentalPassIncGen

  LINK_LIBS PUBLIC
  BuddyKernelContract
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
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrDialect
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorTransforms
)
