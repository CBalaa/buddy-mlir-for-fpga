// RUN: buddy-opt %s --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s

// The opt-in backend consumes buffer-form Linalg directly.  Ordinary matmul
// remains a memory/arithmetic loop, while the explicitly marked signed i8
// operation gets typed SPM windows, paired DMA requests and AME tile ops.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @ordinary(%a: memref<2x3xf32>, %b: memref<3x4xf32>,
                      %c: memref<2x4xf32>) {
    linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>)
      outs(%c : memref<2x4xf32>)
    return
  }

  func.func @marked(%a: memref<64x64xi8>,
                    %b: memref<64x64xi8, strided<[1, 64]>>,
                    %c: memref<64x64xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>,
                   bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// CHECK: func.func @ordinary
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.mulf
// CHECK-NOT: linalg.matmul
// CHECK: func.func @marked
// CHECK: bosc_ame.spm.view
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK: bosc_ame.tile.load_a
// CHECK: bosc_ame.tile.load_b
// CHECK: bosc_ame.tile.mma
// CHECK: bosc_ame.tile.store_acc
