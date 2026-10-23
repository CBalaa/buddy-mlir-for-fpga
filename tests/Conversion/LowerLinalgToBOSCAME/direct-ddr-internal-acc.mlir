// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s

// An AME C operand backed by a compiler allocation is an internal temporary;
// direct-ddr applies only to external function arguments.  It therefore keeps
// the SPM accumulator and its write-back DMA.
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>) {
    %zero = arith.constant 0 : i32
    %c = memref.alloc() alignment = 64 : memref<64x64xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<64x64xi32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32>)
    memref.dealloc %c : memref<64x64xi32>
    return
  }
}

// CHECK: bosc_ame.tile.load_acc {{.*}} : memref<64x64xi32, strided<[64, 1]>, #bosc_ame.spm<
// CHECK: bosc_ame.tile.store_acc {{.*}} : vector<64x64xi32>, memref<64x64xi32, strided<[64, 1]>, #bosc_ame.spm<
// CHECK: bosc_ame.dma.start
