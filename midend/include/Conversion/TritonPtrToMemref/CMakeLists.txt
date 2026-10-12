# TableGen definitions for tptr -> memref lowering.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Conversion/TritonPtrToMemref
# (triton-riscv commit fe554ff).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonPtrToMemref)
add_public_tablegen_target(BuddyTritonPtrToMemrefPassIncGen)
