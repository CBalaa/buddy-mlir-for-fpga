// RUN: not buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// Preserve the byte-shifted memref.view input. Its identity generic becomes
// a DDR-to-DDR copy, rejected before endpoint address lowering.
// Typed endpoint offset proofs are covered by PlanEmitValidation.py.
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

// CHECK: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
