// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.reported 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: cmp %t.reported %t.default

// Retaining %shared in one entry function would leave room for only one of
// the other function's two 96 KiB windows. Reloading it at its consumers lets
// that function use both banks.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @capacity_fallback(
      %shared: memref<24576xf32> {bosc_ame.logical_value_id = 10 : i64,
                                  bosc_ame.logical_value_users = array<i64: 0, 1>},
      %dst0: memref<24576xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst1: memref<24576xf32> {bosc_ame.logical_value_id = 12 : i64})
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
      ins(%shared : memref<24576xf32>) outs(%dst0 : memref<24576xf32>) {
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
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%shared : memref<24576xf32>) outs(%dst1 : memref<24576xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @heavy(
      %src: memref<24576xf32> {bosc_ame.logical_value_id = 13 : i64,
                               bosc_ame.logical_value_users = array<i64: 2>},
      %dst2: memref<24576xf32> {bosc_ame.logical_value_id = 14 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 1 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 13>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 14>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<24576xf32>) outs(%dst2 : memref<24576xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// PLAN: transfer[{{[0-9]+}}]: consumer=linalg.generic residency=operation-local consumers=[linalg.generic] role=input-a
// PLAN: resources: transfers=6 dma-packets=6 read-bytes=294912 write-bytes=294912 totals-exact=true latency=uncalibrated
// PLAN: function-argument-residency-search: candidates=1 selected=0 rejected-capacity=1 objective=minimize-read-bytes-then-packets status=exact latency=uncalibrated
// PLAN-NEXT: argument-residency[0]: function=@capacity_fallback arg#0 consumers=2 capacity=98304 resident-read-bytes=98304 resident-packets=1 reload-read-bytes=196608 reload-packets=2 decision=reload reason=spm-capacity
