// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.reported 2>%t.plan
// RUN: cmp %t.default %t.reported
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan

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

// PLAN: BOSCAME plan report:
// PLAN: compute-graph: status=complete tasks=2 dependencies=1 logical-values=3
// PLAN: task[0]: op=linalg.generic kind=elementwise stage=0 predecessors=[] inputs=[10] outputs=[11] varying-axes=[] barrier-after=false rematerializable=true
// PLAN: task[1]: op=linalg.generic kind=elementwise stage=0 predecessors=[0] inputs=[11] outputs=[12] varying-axes=[] barrier-after=false rematerializable=true
// PLAN: dependency: 0 -> 1 resolved=true
// PLAN: logical-value[10]: producer=function-argument consumers=[0] lifetime=known
// PLAN: logical-value[11]: producer=0 consumers=[1] lifetime=known
// PLAN: logical-value[12]: producer=1 consumers=[] lifetime=known
// PLAN: buffer[0]: root=known root-id=0 views=[] shape=[16] strides=[1] effective-byte-offset-bytes=0 capacity=64 spm-lifetime=operation-local task=0..0 physical-users=[0] physical-lifetime=known tasks=0..0
// PLAN: buffer[1]: root=known root-id=1 views=[memref.view] shape=[16] strides=[1] effective-byte-offset-bytes=64 capacity=64 spm-lifetime=cross-task task=0..1 physical-users=[0,1] physical-lifetime=known tasks=0..1
// PLAN: buffer[2]: root=known root-id=2 views=[] shape=[16] strides=[1] effective-byte-offset-bytes=0 capacity=64 spm-lifetime=operation-local task=1..1 physical-users=[1] physical-lifetime=known tasks=1..1
// PLAN: buffer-accesses: count=4 scope=post-planning
// PLAN: buffer-access[0]: op=linalg.generic effect=read
// PLAN-SAME: root=known root-id=0
// PLAN-SAME: task=0 transfer=0 alias=unknown dma=planned-legal
// PLAN-SAME: logical-value=10
// PLAN: buffer-access[1]: op=linalg.generic effect=write
// PLAN-SAME: task=0 transfer=1 alias=unknown dma=planned-legal logical-value=11
// PLAN: buffer-access[2]: op=linalg.generic effect=read
// PLAN-SAME: root=known root-id=1
// PLAN-SAME: task=1 transfer=1 alias=unknown dma=planned-legal
// PLAN-SAME: logical-value=11
// PLAN: buffer-access[3]: op=linalg.generic effect=write
// PLAN-SAME: task=1 transfer=2 alias=unknown dma=planned-legal logical-value=12
// PLAN: resources: transfers=3 dma-packets=3 read-bytes=64 write-bytes=128 totals-exact=true latency=uncalibrated
