// RUN: split-file %s %t
// RUN: not buddy-opt %t/entry.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ENTRY
// RUN: not buddy-opt %t/helper.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=HELPER

// The original entry copy has no independent V1 DDR-to-DDR recipe. Called
// helper formals have no entry disjointness guarantee; the contract rejects
// their potentially overlapping copy before physical route selection.
// ENTRY: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// HELPER: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer

//--- entry.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @disjoint_entry(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
      %other: memref<16xf32>,
      %dst0: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst1: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
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
      ins(%src : memref<16xf32>) outs(%dst0 : memref<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    memref.store %zero, %other[%c0] : memref<16xf32>
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
      ins(%src : memref<16xf32>) outs(%dst1 : memref<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }
}

//--- helper.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func private @alias_helper(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
      %other: memref<16xf32>,
      %dst0: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst1: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
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
      ins(%src : memref<16xf32>) outs(%dst0 : memref<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    memref.store %zero, %other[%c0] : memref<16xf32>
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
      ins(%src : memref<16xf32>) outs(%dst1 : memref<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @alias_entry(%src: memref<16xf32>, %dst: memref<16xf32>) {
    func.call @alias_helper(%src, %src, %dst, %dst)
      : (memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}
