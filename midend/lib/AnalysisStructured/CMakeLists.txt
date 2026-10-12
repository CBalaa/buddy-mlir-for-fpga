# Pointer analysis of the structured/experimental pipeline.  It is distinct
# from the ordinary TTIR -> Linalg Analysis/PtrAnalysis: `tts` structured state
# is tracked here.
#
# Migrated from thirdparty/triton-riscv/lib/AnalysisStructured (triton-riscv
# commit fe554ff).

add_mlir_library(BuddyTritonAnalysisStructured
  PtrAnalysis.cpp

  DEPENDS
  BuddyTritonTableGen
  BuddyTritonStructuredTableGen

  LINK_LIBS PUBLIC
  BuddyTritonAnalysis
  BuddyTritonIR
  BuddyTritonStructuredIR
  MLIRAnalysis
  MLIRArithDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSupport
  MLIRTransformUtils
)
