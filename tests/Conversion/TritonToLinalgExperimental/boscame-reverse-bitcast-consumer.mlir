// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  func.func @reverse_bitcast_consumer(%input: tensor<4xi32>, %upper: index)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty_i32 = tensor.empty() : tensor<4xi32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 700 : i64}}
        ins(%input : tensor<4xi32>) outs(%empty_i32 : tensor<4xi32>) {
      ^bb0(%value: i32, %unused: i32):
        %next = arith.addi %value, %value : i32
        linalg.yield %next : i32
    } -> tensor<4xi32>
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %seed) -> (tensor<4xi32>) {
      %next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 701 : i64}
          ins(%acc : tensor<4xi32>) outs(%empty_i32 : tensor<4xi32>) {
        ^bb0(%value: i32, %unused: i32):
          %doubled = arith.addi %value, %value : i32
          linalg.yield %doubled : i32
      } -> tensor<4xi32>
      scf.yield %next : tensor<4xi32>
    }
    %bits = tensor.bitcast %loop : tensor<4xi32> to tensor<4xf32>
    %empty_f32 = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 702 : i64}
        ins(%bits : tensor<4xf32>) outs(%empty_f32 : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @reverse_bitcast_consumer
// CHECK: scf.for
// CHECK: exit_consumer_compute_ids = array<i64: 2>
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 702 : i64
