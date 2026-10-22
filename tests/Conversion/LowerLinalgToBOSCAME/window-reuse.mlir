// RUN: buddy-opt %s --lower-linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s --implicit-check-not=vector.load --implicit-check-not=vector.store

// Each copy needs both complete 128 KiB banks. All transfers and consumers
// finish before the next operation, so sequential operations and independent
// functions can use the same windows without accumulating their capacities.
module {
  func.func @sequential(%input: memref<32768xf32>,
                        %temporary: memref<32768xf32>,
                        %out: memref<32768xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32768xf32>) outs(%temporary : memref<32768xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%temporary : memref<32768xf32>) outs(%out : memref<32768xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }

  func.func @independent(%input: memref<32768xf32>, %out: memref<32768xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32768xf32>) outs(%out : memref<32768xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @sequential
// CHECK: bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<32768xf32, #bosc_ame.spm<0>>
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<32768xf32, #bosc_ame.spm<1>>
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<32768xf32, #bosc_ame.spm<0>>
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<32768xf32, #bosc_ame.spm<1>>
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return

// CHECK-LABEL: func.func @independent
// CHECK: bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<32768xf32, #bosc_ame.spm<0>>
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<32768xf32, #bosc_ame.spm<1>>
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
