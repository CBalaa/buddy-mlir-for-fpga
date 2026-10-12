# TableGen definitions for the --triton-to-linalg-experimental driver.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Conversion/TritonToLinalgExperimental
# (triton-riscv commit fe554ff).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToLinalgExperimental)
add_public_tablegen_target(BuddyTritonToLinalgExperimentalPassIncGen)
