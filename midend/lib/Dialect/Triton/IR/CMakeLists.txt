# TTIR (Triton IR) dialect implementation.
#
# Migrated from thirdparty/triton-riscv/triton/lib/Dialect/Triton/IR
# (Triton commit 58895270e6230491acc15fe7ba6d4c849e838db1).  The dialect keeps
# the upstream spelling: dialect name `tt`, C++ namespace `mlir::triton`.

add_mlir_dialect_library(BuddyTritonIR
  Dialect.cpp
  DiscardableAttributes.cpp
  Ops.cpp
  OpInterfaces.cpp
  Traits.cpp
  Types.cpp
  Utility.cpp

  DEPENDS
  BuddyTritonTableGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRSCFDialect
)
