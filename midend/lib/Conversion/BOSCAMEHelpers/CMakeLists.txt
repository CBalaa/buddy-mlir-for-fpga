add_mlir_library(BuddyBOSCAMEHelpers
  BOSCAMEHelpers.cpp
  StructuredMemory.cpp

  LINK_LIBS PUBLIC
  BuddyBOSCAME
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
