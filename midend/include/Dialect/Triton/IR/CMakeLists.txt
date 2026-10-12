# TTIR (Triton IR, dialect name `tt`) TableGen definitions.
#
# Migrated from thirdparty/triton-riscv/triton/include/triton/Dialect/Triton/IR
# (Triton commit 58895270e6230491acc15fe7ba6d4c849e838db1).  All generated
# files are produced here by Buddy's own mlir-tblgen and land in the build
# tree under midend/include/Dialect/Triton/IR/.

set(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS TritonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS TritonTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TritonTypeEnums.td)
mlir_tablegen(TypesEnums.h.inc -gen-enum-decls)
mlir_tablegen(TypesEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS TritonInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS TritonOpInterfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS TritonTypeInterfaces.td)
mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)

set(LLVM_TARGET_DEFINITIONS Canonicalize.td)
mlir_tablegen(TritonCanonicalize.inc -gen-rewriters)

add_public_tablegen_target(BuddyTritonTableGen)
