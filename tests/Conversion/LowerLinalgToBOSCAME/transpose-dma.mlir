// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' --lower-bosc-ame -o /dev/null

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
// CHECK: memref.transpose
// CHECK: bosc_ame.dma.start {{.*}}{channel = 0 : i32}
// CHECK: bosc_ame.dma.wait
// CHECK-COUNT-64: bosc_ame.dma.start {{.*}}{channel = 1 : i32}
// CHECK: bosc_ame.dma.wait
// CHECK: bosc_ame.tile.load_b
// CHECK: bosc_ame.tile.mma
