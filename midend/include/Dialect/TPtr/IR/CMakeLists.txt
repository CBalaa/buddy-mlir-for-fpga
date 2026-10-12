# TableGen definitions for the TPtr (`tptr`) dialect.
#
# Migrated from
# thirdparty/triton-riscv/include/triton-shared/Dialect/TPtr/IR
# (triton-riscv commit fe554ff).  TPtr is the pointer dialect used by the
# experimental pipeline (`--triton-to-linalg-experimental`) and builds on
# MLIR's built-in `ptr` dialect.

set(LLVM_TARGET_DEFINITIONS TPtrDialect.td)
mlir_tablegen(TPtrAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=tptr)
mlir_tablegen(TPtrAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tptr)
mlir_tablegen(TPtrDialect.h.inc -gen-dialect-decls -dialect=tptr)
mlir_tablegen(TPtrDialect.cpp.inc -gen-dialect-defs -dialect=tptr)
mlir_tablegen(TPtrOps.h.inc -gen-op-decls)
mlir_tablegen(TPtrOps.cpp.inc -gen-op-defs)
mlir_tablegen(TPtrTypes.h.inc -gen-typedef-decls -typedefs-dialect=tptr)
mlir_tablegen(TPtrTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tptr)
add_public_tablegen_target(BuddyTPtrTableGen)
