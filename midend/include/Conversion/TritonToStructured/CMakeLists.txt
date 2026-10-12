# TableGen definitions for the structured TTIR -> tts conversion.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Conversion/TritonToStructured
# (triton-riscv commit fe554ff).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToStructured)
add_public_tablegen_target(BuddyTritonToStructuredPassIncGen)
