// RUN: buddy-opt %s --linalg-to-boscame='buffer-placement=true enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: FileCheck %s --check-prefix=GRAPH < %t
// RUN: buddy-opt %t --bosc-ame-spm-dma-check -o /dev/null

// The adapter recomputes task IDs from actual operations. A resident fill and
// direct DDR matmul each retain one task at their emitted completion point.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @resident_and_ame_graph(
      %a: memref<16x64xi8> {bosc_ame.logical_value_id = 10 : i64,
        bosc_ame.logical_value_users = array<i64: 1>},
      %b: memref<64x16xi8, strided<[1, 64]>> {
        bosc_ame.logical_value_id = 11 : i64,
        bosc_ame.logical_value_users = array<i64: 1>},
      %c: memref<16x16xi32> {bosc_ame.logical_value_id = 12 : i64,
        bosc_ame.logical_value_users = array<i64: 1>})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    %zero = arith.constant 0.0 : f32
    %resident = memref.alloc() alignment = 64 : memref<16xf32>
    linalg.fill {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64>,
      bosc_ame.compute_kind = "fill",
      bosc_ame.compute_output_values = array<i64: 13>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.logical_value_ids = array<i64: 13>,
      bosc_ame.logical_value_users = [array<i64>],
      bosc_ame.rematerializable = true}
      ins(%zero : f32) outs(%resident : memref<16xf32>)
    linalg.matmul {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10, 11, 12>,
      bosc_ame.compute_kind = "matmul",
      bosc_ame.compute_output_values = array<i64: 14>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.logical_value_ids = array<i64: 14>,
      bosc_ame.logical_value_users = [array<i64>],
      bosc_ame.lower_to_ame,
      bosc_ame.rematerializable = false}
      ins(%a, %b : memref<16x64xi8>, memref<64x16xi8, strided<[1, 64]>>)
      outs(%c : memref<16x16xi32>)
    return
  }
}

// PLAN: memref.alloc plan.buffer={bank = 0 : i64, capacity = 64 : i64, level = "spm", offset = 0 : i64
// PLAN: linalg.matmul plan.route={a = {bytes = 1024 : i64, level = "ddr"
// PLAN-SAME: accumulator = {bytes = 1024 : i64, level = "ddr"
// PLAN-SAME: b = {bytes = 1024 : i64, level = "ddr"
// PLAN: high_water = array<i64: 64, 0>
// GRAPH: bosc_ame.source_task_count = 2 : i64
// GRAPH: contract.compute_graph = {complete = true
// GRAPH: contract.task = {barrier_after = false, id = 0 : i64
// GRAPH-NOT: contract.task
// GRAPH: bosc_ame.tile.store_acc
// GRAPH-SAME: contract.task = {barrier_after = true, id = 1 : i64
// GRAPH-SAME: input_values = array<i64: 0, 1, 2>
// GRAPH-NOT: contract.task
