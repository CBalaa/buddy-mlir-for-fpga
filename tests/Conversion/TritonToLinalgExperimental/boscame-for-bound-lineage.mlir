// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  func.func @for_bound_lineage(%input: tensor<4xf32>, %bound: memref<1xindex>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = memref.load %bound[%c0] {
      bosc_ame.memory = {kind = "load", origin_id = 730 : i64}
    } : memref<1xindex>
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 731 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %next = arith.addf %value, %value : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %seed) -> (tensor<4xf32>) {
      %next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], bosc_ame.origin_id = 732 : i64}
          ins(%acc : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %doubled = arith.addf %value, %value : f32
          linalg.yield %doubled : f32
      } -> tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 733 : i64}
        ins(%loop : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @for_bound_lineage
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 733 : i64
// CHECK-SAME: bosc_ame.compute_predecessors = array<i64: 0, 1, 2>
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 730, 731>
