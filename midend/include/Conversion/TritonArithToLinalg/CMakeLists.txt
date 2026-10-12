# TableGen definitions for arith/tensor-to-linalg helpers of the experimental pipeline.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Conversion/TritonArithToLinalg
# (triton-riscv commit fe554ff).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonArithToLinalg)
add_public_tablegen_target(BuddyTritonArithToLinalgPassIncGen)
