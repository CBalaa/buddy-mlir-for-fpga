// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  func.func @reverse_pad_consumer(%input: tensor<4xf32>, %upper: index)
      -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 710 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %next = arith.addf %value, %value : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %seed) -> (tensor<4xf32>) {
      %next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 711 : i64}
          ins(%acc : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %doubled = arith.addf %value, %value : f32
          linalg.yield %doubled : f32
      } -> tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    %zero = arith.constant 0.0 : f32
    %padded = tensor.pad %loop low[0] high[4] {
    ^bb0(%index: index):
      tensor.yield %zero : f32
    } : tensor<4xf32> to tensor<8xf32>
    %empty_padded = tensor.empty() : tensor<8xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 712 : i64}
        ins(%padded : tensor<8xf32>) outs(%empty_padded : tensor<8xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// CHECK-LABEL: func.func @reverse_pad_consumer
// CHECK: scf.for
// CHECK: exit_consumer_compute_ids = array<i64: 2>
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 712 : i64
