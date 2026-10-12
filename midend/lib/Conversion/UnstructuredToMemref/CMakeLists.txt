# Unstructured (gather/scatter) lowering for `--triton-to-linalg-experimental`.
#
# Migrated from thirdparty/triton-riscv/lib/Conversion/UnstructuredToMemref
# (triton-riscv commit fe554ff).

add_mlir_library(BuddyTritonUnstructuredToMemref
  UnstructuredToMemrefPass.cpp

  DEPENDS
  BuddyTritonUnstructuredToMemrefPassIncGen

  LINK_LIBS PUBLIC
  BuddyTritonAnalysis
  BuddyTritonIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
)
