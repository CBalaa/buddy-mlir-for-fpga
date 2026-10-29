// RUN: not buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// Preserve the f16 identity-generic input. It normalizes to a DDR-to-DDR
// boundary, which V1 rejects before selecting a vector or staging recipe.
// Legal automatic staging fallback is covered by PlanRoutes.mlir/py.
module {
  func.func @f16(%input: memref<32xf16>, %out: memref<32xf16>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32xf16>) outs(%out : memref<32xf16>) {
    ^bb0(%x: f16, %unused: f16):
      linalg.yield %x : f16
    }
    return
  }

}

// CHECK: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
