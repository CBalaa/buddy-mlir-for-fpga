add_mlir_library(BuddyBOSCAMEHelpers
  BOSCAMEHelpers.cpp
  StructuredMemory.cpp
  MarkedMatmul.cpp
  SPMPlacement.cpp
  SPMComputations.cpp

  LINK_LIBS PUBLIC
  BuddyBOSCAME
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
