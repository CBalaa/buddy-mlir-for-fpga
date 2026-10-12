# TableGen definitions for the unstructured fallback of the experimental pipeline.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Conversion/TritonToUnstructured
# (triton-riscv commit fe554ff).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToUnstructured)
add_public_tablegen_target(BuddyTritonToUnstructuredPassIncGen)
