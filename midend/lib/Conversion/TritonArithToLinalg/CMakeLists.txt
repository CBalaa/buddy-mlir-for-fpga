# Arith/`tt.`-operator to Linalg conversion used inside the experimental
# pipeline.  The conversion patterns themselves live in the header-only
# Conversion/TritonArithToLinalg/ConversionPatterns.hpp and are compiled into
# this library; the pass wrapper added here is what
# `--triton-to-linalg-experimental` drives.
#
# Migrated from thirdparty/triton-riscv/lib/Conversion/TritonArithToLinalg
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonArithToLinalg
  TritonArithToLinalg.cpp
  TritonArithToLinalgPass.cpp

  DEPENDS
  BuddyTritonArithToLinalgPassIncGen

  LINK_LIBS PUBLIC
  BuddyTritonAnalysis
  BuddyTritonIR
  BuddyTritonStructuredIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
)
