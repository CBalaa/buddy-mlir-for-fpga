# TPtr (`tptr`) dialect implementation.
#
# Migrated from thirdparty/triton-riscv/lib/Dialect/TPtr/IR (triton-riscv
# commit fe554ff).

add_mlir_dialect_library(BuddyTPtrIR
  TPtrOps.cpp
  TPtrDialect.cpp

  DEPENDS
  BuddyTPtrTableGen

  LINK_LIBS PUBLIC
  BuddyTritonIR
  MLIRIR
  MLIRPtrDialect
)
