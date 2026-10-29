// RUN: not buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// Adjacent static views pass the contract alias proof. The identity generic
// becomes DDR-to-DDR copy, which V1 does not plan independently.
// Supported direct compute and terminal stores are covered by PlanRoutes.py
// and PlanTerminalStore.py.
module {
  func.func @disjoint_views(%root: memref<32xf32>) {
    %input = memref.subview %root[0] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1]>>
    %output = memref.subview %root[16] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: 16>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"],
                    bosc_ame.memory = {kind = "store",
                                       destination = "external_ddr",
                                       logical_shape = array<i64: 16>,
                                       valid_shape = array<i64: 16>,
                                       masked = false}}
      ins(%input : memref<16xf32, strided<[1]>>)
      outs(%output : memref<16xf32, strided<[1], offset: 16>>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// CHECK: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
