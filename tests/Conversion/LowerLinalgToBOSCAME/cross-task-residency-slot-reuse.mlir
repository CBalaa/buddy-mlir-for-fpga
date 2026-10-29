// RUN: not buddy-opt %s '--linalg-to-boscame=plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// The retained identity generic normalizes to an independent DDR-to-DDR
// copy, which V1 rejects before scheduling or physical placement.
// Supported direct/staged routes and capacity proofs are covered by
// PlanRoutes.mlir/py and PlanTerminalStore.py.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @sequential_slot_reuse(
      %a: memref<32768xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
      %b: memref<32768xf32> {bosc_ame.logical_value_id = 20 : i64,
                             bosc_ame.logical_value_users = array<i64: 2, 3>},
      %c: memref<32768xf32> {bosc_ame.logical_value_id = 30 : i64,
                             bosc_ame.logical_value_users = array<i64: 4, 5>},
      %dst0: memref<32768xf32>, %dst1: memref<32768xf32>,
      %dst2: memref<32768xf32>, %dst3: memref<32768xf32>,
      %dst4: memref<32768xf32>, %dst5: memref<32768xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 6 : i64} {
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
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 32768>, masked = false,
                         valid_shape = array<i64: 32768>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a : memref<32768xf32>) outs(%dst1 : memref<32768xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
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
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 3 : i64,
      bosc_ame.compute_input_values = array<i64: 20>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 14>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 32768>, masked = false,
                         valid_shape = array<i64: 32768>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%b : memref<32768xf32>) outs(%dst3 : memref<32768xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 4 : i64,
      bosc_ame.compute_input_values = array<i64: 30>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 15>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 32768>, masked = false,
                         valid_shape = array<i64: 32768>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%c : memref<32768xf32>) outs(%dst4 : memref<32768xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 5 : i64,
      bosc_ame.compute_input_values = array<i64: 30>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 16>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 32768>, masked = false,
                         valid_shape = array<i64: 32768>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%c : memref<32768xf32>) outs(%dst5 : memref<32768xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// CHECK: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
