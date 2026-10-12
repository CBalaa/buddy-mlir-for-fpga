# TableGen definitions for tts -> memref/linalg lowering.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Conversion/StructuredToMemref
# (triton-riscv commit fe554ff).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name StructuredToMemref)
add_public_tablegen_target(BuddyTritonStructuredToMemrefPassIncGen)
