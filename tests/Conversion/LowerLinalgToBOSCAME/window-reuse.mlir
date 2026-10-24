// RUN: not buddy-opt %s '--lower-linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s

// The original module reads an external region after writing it. V1 treats
// that write as terminal and rejects the later read before residency search.
// Private producer/consumer lifetimes remain covered by PlanRoutes.py.
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

// CHECK: kernel contract V23/V24: v1 does not support reading an external region after its terminal write
