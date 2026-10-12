# Splits leftovers of the structured conversion that cannot be represented as
# `tts` ops; second stage of `--triton-to-linalg-experimental`.
#
# Migrated from thirdparty/triton-riscv/lib/Conversion/TritonToUnstructured
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonToUnstructured
  TritonToUnstructuredPass.cpp

  DEPENDS
  BuddyTritonStructuredTableGen
  BuddyTritonToUnstructuredPassIncGen

  LINK_LIBS PUBLIC
  BuddyTritonAnalysis
  BuddyTritonAnalysisStructured
  BuddyTritonIR
  BuddyTritonStructuredIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)
