// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  // The read precedes the full-buffer update in the loop body. On a later
  // iteration it can observe the update from the previous iteration, while
  // the post-loop read must also retain the zero-trip initial contents.
  func.func @memref_recurrence(%condition: i1, %upper: index,
      %initial: tensor<4xf32>,
      %replacement: tensor<4xf32>, %observed_dst: memref<4xf32>,
      %dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<4xf32>
    %initial_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 110 : i64}}
        ins(%initial : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %buffer = bufferization.to_buffer %initial_values
      : tensor<4xf32> to memref<4xf32>
    %replacement_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 130 : i64}}
        ins(%replacement : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %replacement_buffer = bufferization.to_buffer %replacement_values
      : tensor<4xf32> to memref<4xf32>

    scf.for %iv = %c0 to %upper step %c1 {
      %snapshot = bufferization.to_tensor %buffer restrict
        : memref<4xf32> to tensor<4xf32>
      %observed = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 120 : i64}
          ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %observed_buffer = bufferization.to_buffer %observed
        : tensor<4xf32> to memref<4xf32>
      memref.copy %observed_buffer, %observed_dst {
        bosc_ame.memory = {kind = "store", origin_id = 121 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.for %inner = %c0 to %upper step %c1 {
        scf.if %condition {
          memref.copy %replacement_buffer, %buffer {
            bosc_ame.memory = {kind = "store", origin_id = 131 : i64}
          } : memref<4xf32> to memref<4xf32>
        }
      }
    }
    memref.copy %buffer, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 140 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  // Slots 0 and 1 exchange values. Slot 2 carries an unrelated source and
  // must remain separate from both exchanged slots.
  func.func @tensor_slot_swap(%upper: index, %lhs: tensor<4xf32>,
      %rhs: tensor<4xf32>, %guard: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<4xf32>
    %lhs_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 210 : i64}}
        ins(%lhs : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %rhs_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 220 : i64}}
        ins(%rhs : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %guard_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 230 : i64}}
        ins(%guard : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>

    %result:3 = scf.for %iv = %c0 to %upper step %c1
        iter_args(%slot0 = %lhs_values, %slot1 = %rhs_values,
                  %slot2 = %guard_values)
        -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
      %next0 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 310 : i64}
          ins(%slot1 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %next1 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 320 : i64}
          ins(%slot0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %next2 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 330 : i64}
          ins(%slot2 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.yield %next0, %next1, %next2 :
        tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
    }

      %consume0 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 401 : i64}
        ins(%result#0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %consume1 = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 402 : i64}
        ins(%result#1 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %consume2 = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 403 : i64}
        ins(%result#2 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %consume0, %consume1, %consume2 :
      tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @memref_recurrence
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK: linalg.generic {{.*}}bosc_ame.compute_id = [[INITIAL_ID:[0-9]+]] : i64{{.*}}bosc_ame.compute_origin_id = 110 : i64
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 130 : i64
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 120 : i64
// CHECK-SAME: bosc_ame.compute_predecessors = array<i64: [[INITIAL_ID]]>
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 110, 130>
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 140 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 110, 130>

// CHECK-LABEL: func.func @tensor_slot_swap
// CHECK: linalg.generic {{.*}}bosc_ame.compute_id = [[LHS_ID:[0-9]+]] : i64{{.*}}bosc_ame.compute_origin_id = 210 : i64
// CHECK: linalg.generic {{.*}}bosc_ame.compute_id = [[RHS_ID:[0-9]+]] : i64{{.*}}bosc_ame.compute_origin_id = 220 : i64
// CHECK: linalg.generic {{.*}}bosc_ame.compute_id = [[GUARD_ID:[0-9]+]] : i64{{.*}}bosc_ame.compute_origin_id = 230 : i64
// CHECK: linalg.generic {{.*}}bosc_ame.compute_predecessors = array<i64: [[RHS_ID]]>{{.*}}bosc_ame.origin_id = 310 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 210, 220>
// CHECK: linalg.generic {{.*}}bosc_ame.compute_predecessors = array<i64: [[LHS_ID]]>{{.*}}bosc_ame.origin_id = 320 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 210, 220>
// CHECK: linalg.generic {{.*}}bosc_ame.compute_predecessors = array<i64: [[GUARD_ID]]>{{.*}}bosc_ame.origin_id = 330 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 230>
// CHECK: bosc_ame.loop_carried_slots = [{
// CHECK-SAME: slot = 0 : i64
// CHECK-SAME: slot = 1 : i64
// CHECK-SAME: slot = 2 : i64
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 401 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 210, 220>
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 402 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 210, 220>
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 403 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 230>
