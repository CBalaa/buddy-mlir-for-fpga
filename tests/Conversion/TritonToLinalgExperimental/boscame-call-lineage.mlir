// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.plain
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.reported 2> %t.facts
// RUN: diff %t.plain %t.reported
// RUN: FileCheck %s < %t.plain

module {
  func.func private @select_second(%first: tensor<4xf32>,
      %second: tensor<4xf32>) -> tensor<4xf32> {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %selected = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%first, %second : tensor<4xf32>, tensor<4xf32>)
        outs(%empty : tensor<4xf32>) {
      ^bb0(%unused_value: f32, %second_value: f32, %unused_output: f32):
        %selected_value = arith.addf %second_value, %one : f32
        linalg.yield %selected_value : f32
    } -> tensor<4xf32>
    return %selected : tensor<4xf32>
  }

  func.func private @select_second_with_two_generics(%first: tensor<4xf32>,
      %second: tensor<4xf32>) -> tensor<4xf32> {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %first_result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%first, %second : tensor<4xf32>, tensor<4xf32>)
        outs(%empty : tensor<4xf32>) {
      ^bb0(%unused_first: f32, %second_value: f32, %unused: f32):
        %updated = arith.addf %second_value, %one : f32
        linalg.yield %updated : f32
    } -> tensor<4xf32>
    %second_result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%first_result : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %updated = arith.addf %value, %one : f32
        linalg.yield %updated : f32
    } -> tensor<4xf32>
    return %second_result : tensor<4xf32>
  }

  func.func @call_lineage(%upper: index, %first: tensor<4xf32>,
      %second: tensor<4xf32>, %dst: memref<4xf32>,
      %alias_dst: memref<4xf32>, %conservative_dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %first_load = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 10 : i64}}
        ins(%first : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %loaded_value = arith.addf %value, %one : f32
        linalg.yield %loaded_value : f32
    } -> tensor<4xf32>
    %second_load = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 20 : i64}}
        ins(%second : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %loaded_value = arith.addf %value, %one : f32
        linalg.yield %loaded_value : f32
    } -> tensor<4xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop_value = scf.for %iv = %c0 to %upper step %c1
        iter_args(%slot = %first_load) -> tensor<4xf32> {
      %loop_result = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>,
                           affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}
          ins(%slot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %updated = arith.addf %value, %one : f32
          linalg.yield %updated : f32
      } -> tensor<4xf32>
      scf.yield %loop_result : tensor<4xf32>
    }
    %selected = func.call @select_second(%second_load, %loop_value)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %output = bufferization.to_buffer %selected
        : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      bosc_ame.memory = {kind = "store", origin_id = 30 : i64}
    } : memref<4xf32> to memref<4xf32>
    %aliased = func.call @select_second(%loop_value, %loop_value)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %alias_output = bufferization.to_buffer %aliased
        : tensor<4xf32> to memref<4xf32>
    memref.copy %alias_output, %alias_dst {
      bosc_ame.memory = {kind = "store", origin_id = 40 : i64}
    } : memref<4xf32> to memref<4xf32>
    %conservative = func.call @select_second_with_two_generics(
        %first_load, %second_load)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %conservative_output = bufferization.to_buffer %conservative
        : tensor<4xf32> to memref<4xf32>
    memref.copy %conservative_output, %conservative_dst {
      bosc_ame.memory = {kind = "store", origin_id = 50 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }
}

// CHECK-LABEL: func.func @call_lineage
// CHECK-SAME: bosc_ame.compute_graph_complete = false
// CHECK-SAME: bosc_ame.source_lineage_complete = false
// CHECK: bosc_ame.loop_carried_slots = [{
// CHECK-SAME: exit_consumer_compute_ids = array<i64: [[STORE_ID:[0-9]+]], [[ALIAS_STORE_ID:[0-9]+]]>
// CHECK: memref.copy {{.*}}bosc_ame.compute_id = [[STORE_ID]] : i64{{.*}}bosc_ame.compute_origin_id = 30 : i64{{.*}}bosc_ame.source_load_origin_ids = array<i64: 10>
// CHECK: memref.copy {{.*}}bosc_ame.compute_id = [[ALIAS_STORE_ID]] : i64{{.*}}bosc_ame.compute_origin_id = 40 : i64{{.*}}bosc_ame.source_load_origin_ids = array<i64: 10>
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 50 : i64{{.*}}bosc_ame.source_load_origin_ids = array<i64: 20>
