// RUN: not buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// Preserve the unaligned byte-shifted view. V1 rejects its DDR-to-DDR copy
// boundary before testing endpoint alignment. PlanRoutes.py separately checks
// an unaligned compute operand with no legal direct or staging route.
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

// CHECK: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
