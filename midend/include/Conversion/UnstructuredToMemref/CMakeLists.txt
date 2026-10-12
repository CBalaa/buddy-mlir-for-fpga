# TableGen definitions for unstructured (gather/scatter) lowering.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Conversion/UnstructuredToMemref
# (triton-riscv commit fe554ff).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name UnstructuredToMemref)
add_public_tablegen_target(BuddyTritonUnstructuredToMemrefPassIncGen)
