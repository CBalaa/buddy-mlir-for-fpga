// RUN: not buddy-opt %s '--linalg-to-boscame=plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// The retained identity generic normalizes to an independent DDR-to-DDR
// copy, which V1 rejects before scheduling or physical placement.
// Supported direct/staged routes and capacity proofs are covered by
// PlanRoutes.mlir/py and PlanTerminalStore.py.
module {
  func.func @two_large_transfers(%src: memref<24576xf32>,
                                 %dst: memref<24576xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<24576xf32>) outs(%dst : memref<24576xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// CHECK: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
