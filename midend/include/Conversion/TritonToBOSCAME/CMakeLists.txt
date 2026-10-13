# TableGen definitions for the TTIR -> BOSCAME pass (`--triton-to-boscame`).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToBOSCAME)
add_public_tablegen_target(BuddyTritonToBOSCAMEPassIncGen)
