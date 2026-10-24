// RUN: not buddy-opt %s --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o /dev/null 2>&1 | FileCheck %s --check-prefix=V1

// Retain the historical source case as a precise v1 compatibility boundary.
// The independent DDR copy has no execution recipe. New contract task/version,
// alias and source-lineage checks live in KernelContract and PlanRoutes tests;
// the removed legacy inventory/report must not imply those proofs succeeded.
// V1: v1 has no independent DDR-to-DDR copy recipe

module {
  func.func @consistent_origin(%source: memref<16xf32>,
                               %destination: memref<16xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"],
      bosc_ame.memory = {kind = "store",
                         logical_shape = array<i64: 16>, masked = false,
                         origin_id = 77 : i64, valid_shape = array<i64: 16>}
    } ins(%source : memref<16xf32>) outs(%destination : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @conflicting_origin(%source: memref<16xf32>,
                                %destination: memref<16xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"],
      bosc_ame.memory = {kind = "store",
                         logical_shape = array<i64: 16>, masked = false,
                         origin_id = 77 : i64, valid_shape = array<i64: 16>},
      bosc_ame.origin_id = 78 : i64
    } ins(%source : memref<16xf32>) outs(%destination : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}
