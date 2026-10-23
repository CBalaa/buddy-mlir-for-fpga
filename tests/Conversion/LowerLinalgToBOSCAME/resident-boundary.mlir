// RUN: buddy-opt %s --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: FileCheck %s --check-prefix=IR < %t
// RUN: FileCheck %s --check-prefix=OFFSETS < %t
// RUN: buddy-opt %t --bosc-ame-spm-dma-check -o /dev/null

// A masked event's static subview is its valid range. The resident window is
// larger than that range, and its subview contributes 64 B exactly once. Both
// external rows retain their dynamic aligned descriptor offset. The two DMA
// starts retain the identity copies' complete compute graph and logical users.
module attributes {bosc_ame.compute_graph_version = 1 : i32,
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @masked_boundary(
      %src: memref<151296xf32> {bosc_ame.logical_value_id = 10 : i64,
        bosc_ame.logical_value_users = array<i64: 0>},
      %dst: memref<151296xf32>, %row: index)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    %zero = arith.constant 0.0 : f32
    %c768 = arith.constant 768 : index
    %offset = arith.muli %row, %c768 : index
    %resident = bosc_ame.spm.view {bank = 1 : i32, offset = 128 : i64}
      : memref<1024xf32, #bosc_ame.spm<1>>
    linalg.fill ins(%zero : f32)
      outs(%resident : memref<1024xf32, #bosc_ame.spm<1>>)
    %valid = memref.subview %resident[16] [768] [1]
      : memref<1024xf32, #bosc_ame.spm<1>>
      to memref<768xf32, strided<[1], offset: 16>, #bosc_ame.spm<1>>
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [768], strides: [1]
      : memref<151296xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %input, %valid {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "load",
      bosc_ame.compute_output_values = array<i64: 11>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.logical_value_ids = array<i64: 11>,
      bosc_ame.logical_value_users = [array<i64: 1>],
      bosc_ame.memory = {kind = "load", logical_shape = array<i64: 1024>,
        masked = true, origin_id = 0 : i64, other = 0.0 : f32,
        valid_shape = array<i64: 768>},
      bosc_ame.rematerializable = false,
      bosc_ame.varying_program_axes = array<i64>}
      : memref<768xf32, strided<[1], offset: ?>>
      to memref<768xf32, strided<[1], offset: 16>, #bosc_ame.spm<1>>
    %output = memref.reinterpret_cast %dst to offset: [%offset], sizes: [768], strides: [1]
      : memref<151296xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %valid, %output {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 11>,
      bosc_ame.compute_kind = "store",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.logical_value_ids = array<i64: 12>,
      bosc_ame.logical_value_users = [array<i64>],
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
        logical_shape = array<i64: 1024>, masked = true,
        origin_id = 1 : i64, valid_shape = array<i64: 768>},
      bosc_ame.rematerializable = false,
      bosc_ame.varying_program_axes = array<i64>}
      : memref<768xf32, strided<[1], offset: 16>, #bosc_ame.spm<1>>
      to memref<768xf32, strided<[1], offset: ?>>
    return
  }
}

// PLAN: transfer[0]:
// PLAN-SAME: bank=1 offset=128 capacity=4096
// PLAN-SAME: read=true write=false
// PLAN: transfer[1]:
// PLAN-SAME: bank=1 offset=128 capacity=4096
// PLAN-SAME: read=false write=true
// PLAN: resources: transfers=2 dma-packets=2 read-bytes=3072 write-bytes=3072

// IR: bosc_ame.compute_graph_node_count = 2 : i64
// IR: bosc_ame.dma.start
// IR-SAME: bosc_ame.compute_id = 0 : i64
// IR-SAME: bosc_ame.compute_input_values = array<i64: 10>
// IR-SAME: bosc_ame.compute_output_values = array<i64: 11>
// IR-SAME: bosc_ame.logical_value_users = [array<i64: 1>]
// IR: bosc_ame.dma.wait
// IR: bosc_ame.dma.start
// IR-SAME: bosc_ame.compute_id = 1 : i64
// IR-SAME: bosc_ame.compute_input_values = array<i64: 11>
// IR-SAME: bosc_ame.compute_predecessors = array<i64: 0>
// IR: bosc_ame.dma.wait
// IR-NOT: bosc_ame.dma.start
// IR-NOT: memref.copy
// OFFSETS: %[[VALID:.*]] = memref.subview {{.*}}[16] [768] [1]
// OFFSETS: %[[LOAD_OFFSET:.*]] = arith.constant 16 : index
// OFFSETS: memref.reinterpret_cast %[[VALID]] to offset: [%[[LOAD_OFFSET]]], sizes: [768], strides: [1]
// OFFSETS: %[[STORE_OFFSET:.*]] = arith.constant 16 : index
// OFFSETS: memref.reinterpret_cast %[[VALID]] to offset: [%[[STORE_OFFSET]]], sizes: [768], strides: [1]
