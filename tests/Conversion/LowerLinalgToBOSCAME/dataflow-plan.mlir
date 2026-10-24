// RUN: not buddy-opt %s --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o /dev/null 2>&1 | FileCheck %s --check-prefix=V1

// Retain the historical source case as a precise v1 compatibility boundary.
// The intermediate external write is followed by another read, so it cannot
// be treated as a terminal boundary. New task/version and alias checks live
// in KernelContract and PlanRoutes tests;
// the removed legacy inventory/report must not imply those proofs succeeded.
// V1: v1 does not support reading an external region after its terminal write

module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @chain(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                            bosc_ame.logical_value_users = array<i64: 0>},
      %backing: memref<128xi8>, %dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    %c64 = arith.constant 64 : index
    %tmp = memref.view %backing[%c64][] : memref<128xi8> to memref<16xf32>
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 11>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<16xf32>) outs(%tmp : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 11>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp : memref<16xf32>) outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}
