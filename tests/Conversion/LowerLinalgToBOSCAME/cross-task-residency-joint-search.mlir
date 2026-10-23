// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.reported 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga ' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: cmp %t.reported %t.default

// The producer output saves one reload. The function argument saves two.
// Both windows compete with the later 64 KiB transfer, so joint search must
// choose the argument residency even though the producer appears first.
module attributes {bosc_ame.compute_graph_version = 1 : i32,
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @producer_candidate(
      %seed: memref<f32> {bosc_ame.logical_value_id = 100 : i64,
                           bosc_ame.logical_value_users = array<i64: 0>},
      %tmp: memref<28672xf32> {bosc_ame.logical_value_id = 101 : i64,
                               bosc_ame.logical_value_users = array<i64: 1>},
      %out: memref<28672xf32> {bosc_ame.logical_value_id = 102 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64: 100>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 101>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed : memref<f32>) outs(%tmp : memref<28672xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 101>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 102>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                          logical_shape = array<i64: 28672>, masked = false,
                          valid_shape = array<i64: 28672>},
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp : memref<28672xf32>) outs(%out : memref<28672xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @argument_candidate(
      %arg: memref<28672xf32> {bosc_ame.logical_value_id = 200 : i64,
                               bosc_ame.logical_value_users = array<i64: 2, 3, 4>},
      %out0: memref<28672xf32> {bosc_ame.logical_value_id = 201 : i64},
      %out1: memref<28672xf32> {bosc_ame.logical_value_id = 202 : i64},
      %out2: memref<28672xf32> {bosc_ame.logical_value_id = 203 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 3 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 200>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 201>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                          logical_shape = array<i64: 28672>, masked = false,
                          valid_shape = array<i64: 28672>},
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%arg : memref<28672xf32>) outs(%out0 : memref<28672xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 3 : i64,
      bosc_ame.compute_input_values = array<i64: 200>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 202>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                          logical_shape = array<i64: 28672>, masked = false,
                          valid_shape = array<i64: 28672>},
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%arg : memref<28672xf32>) outs(%out1 : memref<28672xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 4 : i64,
      bosc_ame.compute_input_values = array<i64: 200>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 203>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                          logical_shape = array<i64: 28672>, masked = false,
                          valid_shape = array<i64: 28672>},
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%arg : memref<28672xf32>) outs(%out2 : memref<28672xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @capacity_pressure(
      %src: memref<16384xf32> {bosc_ame.logical_value_id = 300 : i64,
                              bosc_ame.logical_value_users = array<i64: 5>},
      %dst: memref<16384xf32> {bosc_ame.logical_value_id = 301 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 1 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 5 : i64,
      bosc_ame.compute_input_values = array<i64: 300>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 301>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                          logical_shape = array<i64: 16384>, masked = false,
                          valid_shape = array<i64: 16384>},
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<16384xf32>) outs(%dst : memref<16384xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// PLAN: function-argument-residency-search: candidates=1 selected=1 rejected-capacity=0 objective=minimize-read-bytes-then-packets status=exact latency=uncalibrated
// PLAN-NEXT: argument-residency[0]: function=@argument_candidate arg#0 consumers=3 capacity=114688 resident-read-bytes=114688 resident-packets=1 reload-read-bytes=344064 reload-packets=3 decision=resident reason=traffic-benefit-and-capacity
// PLAN: cross-task-residency-search: candidates=2 selected=1 rejected-capacity=1 objective=minimize-total-read-bytes-then-packets status=exact latency=uncalibrated
// PLAN: producer-output-residency-search: candidates=1 selected=0 rejected-capacity=1 objective=maximize-read-bytes-then-packets-saved status=exact latency=uncalibrated
// PLAN-NEXT: producer-output-residency[0]: function=@producer_candidate consumers=1 capacity=114688 saved-read-bytes=114688 saved-packets=1 decision=writeback-and-reload reason=spm-capacity
