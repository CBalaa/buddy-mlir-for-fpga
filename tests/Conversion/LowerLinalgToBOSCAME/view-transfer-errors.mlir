// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s

module {
  func.func @unaligned_view(%input: memref<1024xi8>, %out: memref<16xf32>) {
    %c4 = arith.constant 4 : index
    %view = memref.view %input[%c4][] : memref<1024xi8> to memref<16xf32>
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

// CHECK: shifts the base pointer by 4 bytes
