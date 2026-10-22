// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga store-policy=direct-ddr plan-report=true' -o %t.reported 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga store-policy=direct-ddr' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: cmp %t.reported %t.default

// Three sequential argument residencies each need a full bank. Their windows
// end before the next one starts, so all three can share the same SPM range.
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

// PLAN-COUNT-3: residency=cross-task{{.*}}bank=0 offset=0 capacity=131072
// PLAN: resources: transfers=3 dma-packets=3 read-bytes=393216 write-bytes=0 totals-exact=true latency=uncalibrated
// PLAN: function-argument-residency-search: candidates=3 selected=3 rejected-capacity=0 objective=minimize-read-bytes-then-packets status=exact latency=uncalibrated
// PLAN: bank[0]: high-water=131072/131072 bytes
// PLAN: bank[1]: high-water=0/131072 bytes
