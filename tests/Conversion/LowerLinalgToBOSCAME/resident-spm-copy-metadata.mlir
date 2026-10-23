// RUN: buddy-opt %s --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t
// RUN: buddy-opt %t --bosc-ame-spm-dma-check -o /dev/null

// A pitched SPM copy needs two packets. Its final wait represents completion
// of the whole event and retains exactly one original compute task, including
// predecessor references and reverse logical users. No packet duplicates IDs.
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

// CHECK: bosc_ame.compute_graph_node_count = 2 : i64
// CHECK: bosc_ame.compute_id = 0 : i64
// CHECK: bosc_ame.dma.start
// CHECK-NOT: bosc_ame.compute_id
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.compute_id
// CHECK: bosc_ame.dma.start
// CHECK-NOT: bosc_ame.compute_id
// CHECK: bosc_ame.dma.wait
// CHECK-SAME: bosc_ame.compute_id = 1 : i64
// CHECK-SAME: bosc_ame.compute_input_values = array<i64: 11>
// CHECK-SAME: bosc_ame.compute_output_values = array<i64: 12>
// CHECK-SAME: bosc_ame.compute_predecessors = array<i64: 0>
// CHECK-SAME: bosc_ame.logical_value_users = [array<i64>]
// CHECK-NOT: bosc_ame.compute_id
