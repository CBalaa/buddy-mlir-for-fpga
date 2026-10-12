# TritonStructured (`tts`) helper dialect implementation.
#
# Migrated from
# thirdparty/triton-riscv/lib/Dialect/TritonStructured/IR (triton-riscv commit
# fe554ff).

add_mlir_dialect_library(BuddyTritonStructuredIR
  TritonStructuredOps.cpp
  TritonStructuredDialect.cpp

  DEPENDS
  BuddyTritonStructuredTableGen

  LINK_LIBS PUBLIC
  BuddyTritonIR
  MLIRIR
)
