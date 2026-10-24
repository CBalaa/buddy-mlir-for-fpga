// RUN: not buddy-opt %s '--linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s

// The legacy dynamic-prefix memory facts do not specify a positive static
// valid shape. V1 rejects those access facts before capacity planning.
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @dynamic_prefix_capacity(
      %shared: memref<64x256xf32, strided<[320, 1]>>
          {bosc_ame.logical_value_id = 10 : i64,
           bosc_ame.logical_value_users = array<i64: 0, 1>},
      %dst0: memref<64x256xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst1: memref<64x256xf32> {bosc_ame.logical_value_id = 12 : i64},
      %external: memref<20992xf32>, %limit: i32)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    %window0 = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<8192xi32, #bosc_ame.spm<0>>
    %window1 = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<12288xi32, #bosc_ame.spm<1>>

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
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%shared : memref<64x256xf32, strided<[320, 1]>>)
      outs(%dst0 : memref<64x256xf32>) {
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
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%shared : memref<64x256xf32, strided<[320, 1]>>)
      outs(%dst1 : memref<64x256xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }

    %packet_mask = arith.constant -16 : i32
    %zero = arith.constant 0 : index
    %maximum = arith.constant 20992 : index
    %aligned = arith.andi %limit, %packet_mask : i32
    %as_index = arith.index_cast %aligned : i32 to index
    %upper = arith.minsi %as_index, %maximum : index
    %extent = arith.maxsi %upper, %zero : index
    %source = memref.subview %external[0] [%extent] [1]
      : memref<20992xf32> to memref<?xf32, strided<[1]>>
    %local = memref.alloc() alignment = 64 : memref<20992xf32>
    %target = memref.subview %local[0] [%extent] [1]
      : memref<20992xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source, %target {bosc_ame.memory = {
      kind = "load", logical_shape = array<i64: 20992>, masked = true,
      valid_shape = array<i64: -9223372036854775808>}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}

// CHECK: kernel contract V4: static valid shape must match the access rank and be positive
