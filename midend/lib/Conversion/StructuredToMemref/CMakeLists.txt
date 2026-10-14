# TritonStructured (`tts`) -> memref/linalg lowering, third stage of
# `--triton-to-linalg-experimental`.
#
# Migrated from thirdparty/triton-riscv/lib/Conversion/StructuredToMemref
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonStructuredToMemref
  StructuredToMemref.cpp
  StructuredToMemrefPass.cpp

  DEPENDS
  BuddyTritonStructuredToMemrefPassIncGen

  LINK_LIBS PUBLIC
  BuddyBOSCAMEHelpers
  BuddyTPtrIR
  BuddyTritonIR
  BuddyTritonStructuredIR
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
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorDialect
)
