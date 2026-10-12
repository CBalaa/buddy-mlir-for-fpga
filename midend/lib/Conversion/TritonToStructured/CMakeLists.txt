# TTIR -> TritonStructured (`tts`) conversion, first stage of the experimental
# pipeline driven by `--triton-to-linalg-experimental`.
#
# Migrated from thirdparty/triton-riscv/lib/Conversion/TritonToStructured
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonToStructured
  TritonToStructuredPass.cpp

  DEPENDS
  BuddyTritonStructuredTableGen
  BuddyTritonToStructuredPassIncGen

  LINK_LIBS PUBLIC
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
