// RUN: split-file %s %t
// RUN: not buddy-opt %t/bn15.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga bn=15' 2>&1 | FileCheck %s --check-prefix=BN
// RUN: not buddy-opt %t/dynamic.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=DYNAMIC

// Legacy tile-size overrides are rejected: geometry belongs to the logical
// buffer operands. The separate dynamic-offset case exercises address proofs
// without an obsolete override.
//--- bn15.mlir
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// An arbitrary dynamic descriptor offset can move a later C tile beyond the
// argument storage. Exact RVV staging handles alignment but cannot replace
// the missing bounds proof, so this endpoint remains rejected.
//--- dynamic.mlir
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %raw: memref<8192xi32>, %offset: index) {
    %c = memref.reinterpret_cast %raw to offset: [%offset], sizes: [64, 64],
      strides: [64, 1] : memref<8192xi32> to
      memref<64x64xi32, strided<[64, 1], offset: ?>>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32, strided<[64, 1], offset: ?>>)
    return
  }
}

// BN: legacy bm/bn/bk overrides are obsolete; geometry is taken from buffer-form Linalg operands
// DYNAMIC: DDR view exceeds its statically known argument storage
