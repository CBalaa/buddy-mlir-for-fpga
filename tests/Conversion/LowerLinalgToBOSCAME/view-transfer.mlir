// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s

// memref.view shifts the base pointer in bytes.  The DMA endpoint must keep
// the view as its source so lowering observes that shift; it is not a
// descriptor-element offset that may be folded away.
module {
  func.func @view_shift(%input: memref<1024xi8>, %out: memref<16xf32>) {
    %c64 = arith.constant 64 : index
    %view = memref.view %input[%c64][] : memref<1024xi8> to memref<16xf32>
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%view : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @view_shift
// CHECK: %[[VIEW:.*]] = memref.view %{{.*}}[%{{.*}}][] : memref<1024xi8> to memref<16xf32>
// CHECK: memref.reinterpret_cast %[[VIEW]] to offset:
// CHECK: bosc_ame.dma.start %{{.*}},
