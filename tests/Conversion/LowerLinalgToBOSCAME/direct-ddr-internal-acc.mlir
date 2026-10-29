// RUN: buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s --implicit-check-not=bosc_ame.dma.start --implicit-check-not=bosc_ame.tile.load_acc --implicit-check-not=vector.store
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// A private accumulator remains in SPM while external A/B use direct DDR.
// Its lifetime has no external output boundary, so no write-back DMA is needed.
// Exact private zero initialization clears the accumulator register directly.
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

// CHECK: bosc_ame.spm.view {{.*}} : memref<64x64xi32, #bosc_ame.spm<0>>
// CHECK: bosc_ame.tile.zero_acc : vector<64x64xi32>
// CHECK: bosc_ame.tile.load_a {{.*}} {bosc_ame.direct_ddr}
// CHECK: bosc_ame.tile.load_b {{.*}} {bosc_ame.direct_ddr}
// CHECK: bosc_ame.tile.store_acc {{.*}} : vector<64x64xi32>, memref<64x64xi32, strided<[64, 1]>, #bosc_ame.spm<0>>
