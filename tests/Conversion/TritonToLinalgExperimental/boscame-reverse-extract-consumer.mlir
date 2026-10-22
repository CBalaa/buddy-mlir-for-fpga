// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  func.func @reverse_extract_consumer(%input: tensor<4xf32>, %upper: index)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %idx = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %input) -> (tensor<4xf32>) {
      %next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 761 : i64}
          ins(%acc : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %doubled = arith.addf %value, %value : f32
          linalg.yield %doubled : f32
      } -> tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    %scalar = tensor.extract %loop[%idx] : tensor<4xf32>
    %direct = linalg.fill ins(%scalar : f32) outs(%empty : tensor<4xf32>)
        -> tensor<4xf32>
    %factor = arith.constant 2.0 : f32
    %scaled = arith.mulf %scalar, %factor : f32
    %filled = linalg.fill ins(%scaled : f32) outs(%empty : tensor<4xf32>)
        -> tensor<4xf32>
    return %direct, %filled : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @reverse_extract_consumer
// CHECK: scf.for
// CHECK: exit_consumer_compute_ids = array<i64: 1, 2>
// CHECK: linalg.fill
