// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  memref.global "private" @side_effect : memref<16xf32> = uninitialized

  func.func @hidden_memory_effect(%src: memref<16xf32>,
                                 %dst: memref<16xf32>) {
    %zero = arith.constant 0 : index
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      %global = memref.get_global @side_effect : memref<16xf32>
      memref.store %value, %global[%zero] : memref<16xf32>
      linalg.yield %value : f32
    }
    return
  }
}

// CHECK: func.func @hidden_memory_effect
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
