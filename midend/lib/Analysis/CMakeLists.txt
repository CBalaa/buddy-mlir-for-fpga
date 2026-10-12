# Analyses and helpers used by the TTIR -> Linalg conversion (conversion
# patterns and the experimental pipeline stages).
#
# Migrated from
# thirdparty/triton-riscv/{include/triton-shared/Analysis,lib/Analysis} plus
# triton-shared/Utils/Utils.{h,cpp} (triton-riscv commit fe554ff), which provide
# pointer/mask/use analysis, OpFoldResult utilities and `isPtrTypeLike`.

add_mlir_library(BuddyTritonAnalysis
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrAnalysis.cpp
  TritonUtils.cpp
  UseAnalysis.cpp

  DEPENDS
  BuddyTritonTableGen
  BuddyTritonStructuredTableGen

  LINK_LIBS PUBLIC
  BuddyTritonIR
  BuddyTritonStructuredIR
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
)
