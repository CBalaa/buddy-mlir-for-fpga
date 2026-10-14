# The TTIR -> BOSCAME conversion (`--triton-to-boscame`): the FPGA AME matmul
# path of a Triton kernel, lowered onto `bosc_ame.spm.view` /
# `bosc_ame.dma.start` / `bosc_ame.dma.wait` / `bosc_ame.tile.*`.

add_mlir_library(BuddyTritonToBOSCAME
  TritonToBOSCAME.cpp

  DEPENDS
  BuddyTritonToBOSCAMEPassIncGen

  LINK_LIBS PUBLIC
  BuddyBOSCAMEHelpers
  BuddyBOSCAME
  BuddyTritonIR
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
)
