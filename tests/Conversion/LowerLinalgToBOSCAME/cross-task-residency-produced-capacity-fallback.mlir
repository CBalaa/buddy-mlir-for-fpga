// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.reported 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: cmp %t.reported %t.default
// RUN: FileCheck %s --check-prefix=IR < %t.reported
// RUN: FileCheck %s --check-prefix=IR-ORDER < %t.reported

// Keeping %tmp resident would reserve one 96 KiB bank across the function
// boundary and prevent @pressure from staging both of its 96 KiB buffers.
// Falling back writes %tmp after its producer and reloads it at the consumer.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @produced(
      %src: memref<24576xf32> {bosc_ame.logical_value_id = 10 : i64,
                               bosc_ame.logical_value_users = array<i64: 0>},
      %tmp: memref<24576xf32> {bosc_ame.logical_value_id = 11 : i64,
                               bosc_ame.logical_value_users = array<i64: 1>},
      %dst: memref<24576xf32> {bosc_ame.logical_value_id = 12 : i64})
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
      ins(%src : memref<24576xf32>) outs(%tmp : memref<24576xf32>) {
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
      ins(%tmp : memref<24576xf32>) outs(%dst : memref<24576xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @pressure(
      %src: memref<24576xf32> {bosc_ame.logical_value_id = 13 : i64,
                               bosc_ame.logical_value_users = array<i64: 2>},
      %dst: memref<24576xf32> {bosc_ame.logical_value_id = 14 : i64})
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
      ins(%src : memref<24576xf32>) outs(%dst : memref<24576xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// PLAN-COUNT-6: residency=operation-local
// PLAN: resources: transfers=6 dma-packets=6 read-bytes=294912 write-bytes=294912 totals-exact=true

// IR-COUNT-9: bosc_ame.dma.start
// IR-NOT: bosc_ame.dma.start

// The producer writes %tmp to DDR before the consumer reloads it.
// IR-ORDER: bosc_ame.dma.start {{.*}}#bosc_ame.spm<1>>, memref<24576xf32, strided<[1], offset: ?>>, index)
// IR-ORDER: bosc_ame.dma.wait
// IR-ORDER: %[[TMP_RELOAD:.*]] = memref.reinterpret_cast %arg1 to
// IR-ORDER: bosc_ame.dma.start %[[TMP_RELOAD]], {{.*}}#bosc_ame.spm<0>>
