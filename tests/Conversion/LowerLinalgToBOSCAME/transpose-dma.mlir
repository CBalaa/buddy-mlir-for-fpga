// RUN: buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s --implicit-check-not=bosc_ame.dma.start --implicit-check-not=bosc_ame.spm.view
// RUN: buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' --lower-bosc-ame -o /dev/null
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// The transposed external descriptor is directly addressable; retain its parent
// pitch instead of packing it through DMA.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @marked_transposed_b(%a: memref<64x64xi8>,
                                 %b: memref<64x64xi8, strided<[128, 1]>>,
                                 %c: memref<64x64xi32>) {
    %bt = memref.transpose %b (i, j) -> (j, i)
      : memref<64x64xi8, strided<[128, 1]>> to
        memref<64x64xi8, strided<[1, 128]>>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>,
                   bosc_ame.lower_to_ame}
      ins(%a, %bt : memref<64x64xi8>,
                    memref<64x64xi8, strided<[1, 128]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// CHECK-LABEL: func.func @marked_transposed_b
// CHECK: %[[BT:.*]] = memref.transpose
// CHECK: %[[B:.*]] = memref.subview %[[BT]][0, 0] [64, 64] [1, 1]
// CHECK-SAME: to memref<64x64xi8, strided<[1, 128]>>
// CHECK: bosc_ame.tile.load_b %[[B]] {bosc_ame.direct_ddr} : memref<64x64xi8, strided<[1, 128]>>
// CHECK: bosc_ame.tile.mma
// CHECK: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr,
