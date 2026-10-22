// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

// The module entry ABI does not promise that a helper's distinct formals are
// disjoint. Preserve known same-root edges without inventing alias lineage.
module {
  func.func private @private_helper(%src: memref<4xf32>,
      %left: memref<4xf32>, %right: memref<4xf32>, %dst: memref<4xf32>) {
    memref.copy %src, %left {
      bosc_ame.memory = {kind = "load", origin_id = 10 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %right, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 20 : i64}
    } : memref<4xf32> to memref<4xf32>
    // This read has a definite same-root writer regardless of aliases.
    memref.copy %left, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 30 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @public_helper(%src: memref<4xf32>,
      %left: memref<4xf32>, %right: memref<4xf32>, %dst: memref<4xf32>) {
    memref.copy %src, %left {
      bosc_ame.memory = {kind = "load", origin_id = 40 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %right, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 50 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func private @unranked_helper(%src: memref<4xf32>,
      %left: memref<*xf32>, %right: memref<*xf32>, %dst: memref<4xf32>) {
    %left_view = memref.cast %left : memref<*xf32> to memref<4xf32>
    %right_view = memref.cast %right : memref<*xf32> to memref<4xf32>
    memref.copy %src, %left_view {
      bosc_ame.memory = {kind = "load", origin_id = 60 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %right_view, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 70 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @caller(%src: memref<4xf32>, %shared: memref<4xf32>,
      %dst: memref<4xf32>) {
    func.call @private_helper(%src, %shared, %shared, %dst)
      : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    func.call @public_helper(%src, %shared, %shared, %dst)
      : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    %unranked = memref.cast %shared : memref<4xf32> to memref<*xf32>
    func.call @unranked_helper(%src, %unranked, %unranked, %dst)
      : (memref<4xf32>, memref<*xf32>, memref<*xf32>, memref<4xf32>) -> ()
    return
  }

  // No module-internal uses: the explicit entry ABI proves these parameters
  // disjoint, so existing complete entry graphs remain complete.
  func.func @entry(%src: memref<4xf32>, %dst: memref<4xf32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 80 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func private @fresh_local_helper(%src: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %src, %buffer {
      bosc_ame.memory = {kind = "load", origin_id = 90 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %buffer, %src {
      bosc_ame.memory = {kind = "store", origin_id = 100 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func private @tensor_helper(%src: tensor<4xf32>,
      %dst: tensor<4xf32>) -> tensor<4xf32> {
    %one = arith.constant 1.0 : f32
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%src : tensor<4xf32>) outs(%dst : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // A tensor read through to_tensor still reads its memref formal root.
  func.func private @tensor_bridge_helper(%src: memref<4xf32>,
      %dst: memref<4xf32>) -> tensor<4xf32> {
    %one = arith.constant 1.0 : f32
    linalg.fill ins(%one : f32) outs(%dst : memref<4xf32>)
    %snapshot = bufferization.to_tensor %src restrict
      : memref<4xf32> to tensor<4xf32>
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // Two potential aliases only matter to this gate when an access writes.
  func.func private @readonly_helper(%left: memref<4xf32>,
      %right: memref<4xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %lhs = memref.load %left[%c0] {
      bosc_ame.memory = {kind = "load", origin_id = 110 : i64}
    } : memref<4xf32>
    %rhs = memref.load %right[%c0] {
      bosc_ame.memory = {kind = "load", origin_id = 120 : i64}
    } : memref<4xf32>
    %sum = arith.addf %lhs, %rhs : f32
    return %sum : f32
  }

  // The entry ABI only provides a nonalias promise in the default space.
  func.func @custom_space_entry(%src: memref<4xf32, 3>,
      %dst: memref<4xf32, 3>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 130 : i64}
    } : memref<4xf32, 3> to memref<4xf32, 3>
    return
  }
}

// CHECK-LABEL: func.func private @private_helper
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 20 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64>
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 30 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 10>
// CHECK-LABEL: func.func @public_helper
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 50 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64>
// CHECK-LABEL: func.func private @unranked_helper
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 70 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64>
// CHECK-LABEL: func.func @caller
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK-LABEL: func.func @entry
// CHECK-SAME: bosc_ame.compute_graph_complete = true
// CHECK-SAME: bosc_ame.source_lineage_complete = true
// CHECK-LABEL: func.func private @fresh_local_helper
// CHECK-SAME: bosc_ame.compute_graph_complete = true
// CHECK-SAME: bosc_ame.source_lineage_complete = true
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 100 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 90>
// CHECK-LABEL: func.func private @tensor_helper
// CHECK-SAME: bosc_ame.compute_graph_complete = true
// CHECK-SAME: bosc_ame.source_lineage_complete = true
// CHECK-LABEL: func.func private @tensor_bridge_helper
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK-LABEL: func.func private @readonly_helper
// CHECK-SAME: bosc_ame.compute_graph_complete = true
// CHECK-SAME: bosc_ame.source_lineage_complete = true
// CHECK-LABEL: func.func @custom_space_entry
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
