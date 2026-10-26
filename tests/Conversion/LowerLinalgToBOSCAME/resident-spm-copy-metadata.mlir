// RUN: buddy-opt %s '--linalg-to-boscame=buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: FileCheck %s --check-prefix=EMIT < %t
// RUN: buddy-opt %t --bosc-ame-spm-dma-check -o /dev/null

// The pitched internal copy uses a synchronous raw ACC32 transfer between
// independent fixed windows. Its completion retains the predecessor on the
// external DMA load and the original load provenance.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @pitched_copy_graph(
      %input: memref<64xf32> {bosc_ame.logical_value_id = 10 : i64,
        bosc_ame.logical_value_users = array<i64: 0>})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xf32, #bosc_ame.spm<0>>
    %b = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<32xf32, #bosc_ame.spm<1>>
    memref.copy %input, %a {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "load",
      bosc_ame.compute_output_values = array<i64: 11>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.logical_value_ids = array<i64: 11>,
      bosc_ame.logical_value_users = [array<i64: 1>],
      bosc_ame.memory = {kind = "load", logical_shape = array<i64: 64>,
        masked = false, origin_id = 0 : i64, valid_shape = array<i64: 64>},
      bosc_ame.rematerializable = false}
      : memref<64xf32> to memref<64xf32, #bosc_ame.spm<0>>
    %pitched = memref.reinterpret_cast %a to offset: [0], sizes: [2, 16], strides: [32, 1]
      : memref<64xf32, #bosc_ame.spm<0>>
      to memref<2x16xf32, strided<[32, 1]>, #bosc_ame.spm<0>>
    %packed = memref.reinterpret_cast %b to offset: [0], sizes: [2, 16], strides: [16, 1]
      : memref<32xf32, #bosc_ame.spm<1>>
      to memref<2x16xf32, strided<[16, 1]>, #bosc_ame.spm<1>>
    memref.copy %pitched, %packed {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 11>,
      bosc_ame.compute_kind = "copy",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.logical_value_ids = array<i64: 12>,
      bosc_ame.logical_value_users = [array<i64>],
      bosc_ame.rematerializable = false}
      : memref<2x16xf32, strided<[32, 1]>, #bosc_ame.spm<0>>
      to memref<2x16xf32, strided<[16, 1]>, #bosc_ame.spm<1>>
    return
  }
}

// PLAN: memref.copy plan.transfer={chunks = {bytes = 64 : i64, count = 2 : i64, source_stride = 128 : i64, target_stride = 64 : i64}
// PLAN-SAME: recipe = "spm-acc32-spm", role = "internal"
// PLAN-SAME: temp_bytes = 0 : i64, temporary_window = "none"
// PLAN-SAME: source_pitch = 128 : i64
// PLAN-SAME: target_pitch = 64 : i64
// PLAN-SAME: total_bytes = 128 : i64

// EMIT: %[[DMA:.*]] = bosc_ame.dma.start
// EMIT: bosc_ame.dma.wait %[[DMA]]
// EMIT-SAME: contract.task = {barrier_after = true, id = 0 : i64
// EMIT-SAME: output_values = array<i64: 1>, predecessors = array<i64>
// EMIT: bosc_ame.matrix.copy
// EMIT-SAME: contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}
// EMIT-SAME: contract.task = {barrier_after = false, id = 1 : i64, input_values = array<i64: 1>, kind = "copy", output_values = array<i64: 2>, predecessors = array<i64: 0>
// EMIT-SAME: source_pitch = 128 : i64
// EMIT-SAME: target_pitch = 64 : i64
