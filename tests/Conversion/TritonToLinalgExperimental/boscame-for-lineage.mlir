// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  tt.func public @for_lineage(
      %initial_ptr: !tt.ptr<f32>, %body_ptr: !tt.ptr<f32>,
      %dst_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %initial_splat = tt.splat %initial_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %body_splat = tt.splat %body_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %initial_ptrs = tt.addptr %initial_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %body_ptrs = tt.addptr %body_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %initial = tt.load %initial_ptrs : tensor<4x!tt.ptr<f32>>
    %result = scf.for %iv = %c0 to %c2 step %c1
        iter_args(%acc = %initial) -> (tensor<4xf32>) {
      %body = tt.load %body_ptrs : tensor<4x!tt.ptr<f32>>
      %next = arith.addf %acc, %body : tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    tt.store %dst_ptrs, %result : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: bosc_ame.compute_graph_complete = false
// CHECK: bosc_ame.source_lineage_complete = false
// CHECK: bosc_ame.compute_predecessors = array<i64: 0, 1>
// CHECK: bosc_ame.logical_value_ids = array<i64: 7>, bosc_ame.loop_carried_slots = [{
// CHECK-SAME: backedge_producer_compute_ids = array<i64: 2>
// CHECK-SAME: exit_consumer_compute_ids = array<i64: 3>
// CHECK-SAME: init_producer_compute_ids = array<i64: 0>
// CHECK-SAME: init_value_id = 2 : i64
// CHECK-SAME: iter_arg_consumer_compute_ids = array<i64: 2>
// CHECK-SAME: iter_arg_value_id = 10 : i64
// CHECK-SAME: result_value_id = 7 : i64
// CHECK-SAME: slot = 0 : i64
// CHECK-SAME: yielded_value_id = 6 : i64
// CHECK: bosc_ame.memory = {destination = "external_ddr", event_kind = "store", kind = "store"
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 0, 1>
