// RUN: split-file %s %t
// RUN: not buddy-opt %t/reader.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=READER
// RUN: not buddy-opt %t/produced.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=PRODUCED
// RUN: not buddy-opt %t/entry.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ENTRY
// RUN: not buddy-opt %t/fresh.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=FRESH
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// Retain all original helper/caller alias relationships. The reader has a
// potentially overlapping helper operands without an exact-address proof;
// the produced value is read after an
// external terminal write; the entry has a DDR-to-DDR boundary; the fresh
// helper has no entry-ABI alignment proof for its required DDR write.
// Entry ABI isolation for called helpers is tested directly by
// tests/KernelContract/entry-abi.py and PlanWorkspaceCalls.py.
// READER: kernel contract V25: overlapping views must have identical element addresses
// PRODUCED: kernel contract V23/V24: v1 does not support reading an external region after its terminal write
// ENTRY: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// FRESH: no legal boundary route: natural endpoint alignment, element type, or SPM placement is unproven

//--- reader.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @reader_helper(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
      %alias: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
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
      ins(%src : memref<16xf32>) outs(%alias : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      %one = arith.constant 1.0 : f32
      %incremented = arith.addf %value, %one : f32
      linalg.yield %incremented : f32
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
      ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @reader_entry(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 0 : i64} {
    func.call @reader_helper(%src, %src, %dst)
      : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}

//--- produced.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func private @produced_helper(
      %tmp: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 1>},
      %other: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64,
                               bosc_ame.logical_value_users = array<i64: 1>},
      %dst: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 10>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      outs(%tmp : memref<16xf32>) {
    ^bb0(%old: f32):
      %two = arith.constant 2.0 : f32
      linalg.yield %two : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10, 11>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp, %other : memref<16xf32>, memref<16xf32>)
      outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %alias: f32, %old: f32):
      %sum = arith.addf %value, %alias : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func @produced_entry(%tmp: memref<16xf32>, %dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 0 : i64} {
    func.call @produced_helper(%tmp, %tmp, %dst)
      : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}

//--- entry.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @public_entry(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
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
      ins(%src : memref<16xf32>) outs(%dst1 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

//--- fresh.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func private @fresh_helper(
      %dst0: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst1: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 3 : i64} {
    %tmp = memref.alloc() alignment = 64 : memref<16xf32>
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 10>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      outs(%tmp : memref<16xf32>) {
    ^bb0(%old: f32):
      %two = arith.constant 2.0 : f32
      linalg.yield %two : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 11>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp : memref<16xf32>) outs(%dst0 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp : memref<16xf32>) outs(%dst1 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @fresh_entry(%dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 0 : i64} {
    func.call @fresh_helper(%dst, %dst)
      : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}
