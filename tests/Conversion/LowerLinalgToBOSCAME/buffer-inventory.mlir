// RUN: not buddy-opt %s --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o /dev/null 2>&1 | FileCheck %s --check-prefix=V1

// Retain the historical source case as a precise v1 compatibility boundary.
// A single private scalar store cannot initialize a larger root. New
// task/version, alias and source-lineage checks live in KernelContract and
// PlanRoutes tests;
// the removed legacy inventory/report must not imply those proofs succeeded.
// V1: v1 private scalar stores require rank-zero storage

module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  memref.global "private" @inventory_global : memref<8xf32> = uninitialized

  func.func @inventory(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                            bosc_ame.logical_value_users = array<i64: 0>},
      %dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 1 : i64} {
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
      ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %scratch = memref.alloc() : memref<4xf32>
    memref.store %one, %scratch[%c0] : memref<4xf32>
    %loaded = memref.load %scratch[%c0] : memref<4xf32>
    memref.store %loaded, %scratch[%c0] : memref<4xf32>
    memref.dealloc %scratch : memref<4xf32>
    %scratch_view = memref.subview %scratch[0] [2] [1]
        : memref<4xf32> to memref<2xf32, strided<[1]>>
    %view_loaded = memref.load %scratch_view[%c0]
        : memref<2xf32, strided<[1]>>
    %cond = arith.constant true
    %selected = arith.select %cond, %src, %dst : memref<16xf32>
    %selected_value = memref.load %selected[%c0] : memref<16xf32>
    memref.store %selected_value, %selected[%c0] : memref<16xf32>
    %global = memref.get_global @inventory_global : memref<8xf32>
    %global_value = memref.load %global[%c0] : memref<8xf32>
    %stack = memref.alloca() : memref<2xf32>
    %stack_value = memref.load %stack[%c0] : memref<2xf32>
    return
  }
}

// Transfer requirements stay in their own table; this inventory deduplicates
// descriptors observed by memory effects and makes no lifetime claim.
