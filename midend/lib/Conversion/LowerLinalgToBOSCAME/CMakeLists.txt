add_mlir_library(LowerLinalgToBOSCAMEPass
  LowerLinalgToBOSCAME.cpp
  BOSCAMEAmeEmitter.cpp
  BOSCAMEBoundaryTransfers.cpp
  BOSCAMEDmaEmitter.cpp
  BOSCAMEFootprint.cpp
  BOSCAMEPlacement.cpp
  BOSCAMEPlanner.cpp

  LINK_LIBS PUBLIC
  BuddyBOSCAMEAnalysis
  BuddyBOSCAME
  BuddyBOSCAMEHelpers
  BuddyBOSCAMETransforms
  MLIRArithDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)
