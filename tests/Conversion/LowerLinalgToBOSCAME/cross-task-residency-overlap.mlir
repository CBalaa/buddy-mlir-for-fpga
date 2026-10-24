// RUN: not buddy-opt %s '--linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s

// Projected legacy memory metadata cannot be translated to a concrete
// identity buffer copy. The adapter rejects it before residency planning.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @overlap_at_consumer(
      %a: memref<32768xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
      %b: memref<32768xf32> {bosc_ame.logical_value_id = 20 : i64,
                             bosc_ame.logical_value_users = array<i64: 1, 2>},
      %dst0: memref<32768xf32>, %dst1: memref<32768xf32>,
      %dst2: memref<32768xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 3 : i64} {
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
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 32768>, masked = false,
                         valid_shape = array<i64: 32768>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a : memref<32768xf32>) outs(%dst0 : memref<32768xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10, 20>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 32768>, masked = false,
                         valid_shape = array<i64: 32768>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : memref<32768xf32>, memref<32768xf32>)
      outs(%dst1 : memref<32768xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 20>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 13>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 32768>, masked = false,
                         valid_shape = array<i64: 32768>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%b : memref<32768xf32>) outs(%dst2 : memref<32768xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// CHECK: legacy memory facts require a concrete buffer copy
