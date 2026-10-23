// RUN: split-file %s %t
// RUN: not buddy-opt %t/vector.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=VECTOR
// RUN: not buddy-opt %t/ame.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=AME

// A direct vector store advances by the physical row stride.  65 f32 values
// per row put the second row at byte offset 260, which is not a cache line.
//--- vector.mlir
module {
  func.func @copy(%src: memref<64x64xf32>,
                  %dst: memref<64x64xf32, strided<[65, 1]>>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64, 64>,
                         valid_shape = array<i64: 64, 64>, masked = false}
    } : memref<64x64xf32> to memref<64x64xf32, strided<[65, 1]>>
    return
  }
}

// VECTOR: BOSCAME DMA physical row address is not 64 B aligned

// A direct AME accumulator uses the same physical row-pitch contract for
// msce32.m.  The byte stride here is 65 * sizeof(i32) = 260 bytes.
//--- ame.mlir
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32, strided<[65, 1]>>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32, strided<[65, 1]>>)
    return
  }
}

// AME: BOSCAME DMA physical row address is not 64 B aligned
