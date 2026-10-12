# TableGen definitions for the TritonStructured (`tts`) helper dialect.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Dialect/TritonStructured/IR
# (triton-riscv commit fe554ff).  Only the dialect and its ops are migrated: the
# TTIR -> Linalg analysis reads `tts.get_structured_state`.

set(LLVM_TARGET_DEFINITIONS TritonStructuredDialect.td)
mlir_tablegen(TritonStructuredDialect.h.inc -gen-dialect-decls -dialect=tts)
mlir_tablegen(TritonStructuredDialect.cpp.inc -gen-dialect-defs -dialect=tts)
mlir_tablegen(TritonStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(TritonStructuredOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BuddyTritonStructuredTableGen)
