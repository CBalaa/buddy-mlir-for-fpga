// RUN: not buddy-opt %s '--linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s

// The original module reads an external region after writing it. V1 treats
// that write as terminal and rejects the later read before residency search.
// Private producer/consumer lifetimes remain covered by PlanRoutes.py.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func private @opaque()
  func.func @unknown_effect(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0>},
      %tmp: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64,
                             bosc_ame.logical_value_users = array<i64: 1>},
      %dst: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
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
    func.call @opaque() : () -> ()
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

// CHECK: kernel contract V23/V24: v1 does not support reading an external region after its terminal write
