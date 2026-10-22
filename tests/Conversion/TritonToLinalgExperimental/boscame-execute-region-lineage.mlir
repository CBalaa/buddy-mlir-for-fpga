// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  func.func @execute_region_tensor(%input: tensor<4xf32>) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %forwarded = scf.execute_region -> tensor<4xf32> no_inline {
      %loaded = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"],
          bosc_ame.memory = {kind = "load", origin_id = 10 : i64}}
          ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.yield %loaded : tensor<4xf32>
    }
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 20 : i64}
        ins(%forwarded : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @execute_region_memref(%source: memref<4xf32>,
      %destination: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    %forwarded = scf.execute_region -> memref<4xf32> no_inline {
      memref.copy %source, %buffer {
        bosc_ame.memory = {kind = "load", origin_id = 30 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.yield %buffer : memref<4xf32>
    }
    memref.copy %forwarded, %destination {
      bosc_ame.memory = {kind = "store", origin_id = 40 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @execute_region_loop_consumer(%input: tensor<4xf32>, %upper: index)
      -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 50 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop_result = scf.for %iv = %c0 to %upper step %c1
        iter_args(%iter = %seed) -> (tensor<4xf32>) {
      %forwarded = scf.execute_region -> tensor<4xf32> no_inline {
        %inside = linalg.generic {
            indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
            iterator_types = ["parallel"], bosc_ame.origin_id = 55 : i64}
            ins(%iter : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %negated = arith.negf %value : f32
            linalg.yield %negated : f32
        } -> tensor<4xf32>
        scf.yield %inside : tensor<4xf32>
      }
      scf.yield %forwarded : tensor<4xf32>
    }
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 60 : i64}
        ins(%loop_result : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @execute_region_program_axis(
      %program: index {bosc_ame.program_role = "program_id",
                       bosc_ame.program_axis = 2 : i64}) {
    %one = arith.constant 1 : index
    %size = scf.execute_region -> index no_inline {
      %extent = arith.addi %program, %one : index
      scf.yield %extent : index
    }
    %source = memref.alloc(%size) : memref<?xf32>
    %destination = memref.alloc(%size) : memref<?xf32>
    memref.copy %source, %destination {
      bosc_ame.memory = {kind = "load", origin_id = 70 : i64}
    } : memref<?xf32> to memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @execute_region_tensor
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 20 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 10>
// CHECK-LABEL: func.func @execute_region_memref
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 40 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 30>
// CHECK-LABEL: func.func @execute_region_loop_consumer
// CHECK: bosc_ame.loop_carried_slots = [{{.*}}exit_consumer_compute_ids = array<i64: 6>{{.*}}]
// CHECK-LABEL: func.func @execute_region_program_axis
// CHECK: memref.copy {{.*}}bosc_ame.varying_program_axes = array<i64: 2>
