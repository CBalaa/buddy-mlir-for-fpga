// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  func.func @switch_selector_is_a_data_dependency(
      %selector_buffer: memref<1xindex>, %then_input: tensor<4xf32>,
      %else_input: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %selector = memref.load %selector_buffer[%c0] {
      bosc_ame.memory = {kind = "load", origin_id = 90 : i64}
    } : memref<1xindex>
    %then_value = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 91 : i64}}
        ins(%then_input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %else_value = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        bosc_ame.memory = {kind = "load", origin_id = 92 : i64}}
        ins(%else_input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %selected = scf.index_switch %selector -> tensor<4xf32>
      case 0 {
        scf.yield %then_value : tensor<4xf32>
      }
      default {
        scf.yield %else_value : tensor<4xf32>
      }
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], bosc_ame.origin_id = 93 : i64}
        ins(%selected : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @switch_selector_is_a_data_dependency
// CHECK: linalg.generic {{.*}}bosc_ame.compute_origin_id = 93 : i64
// CHECK-SAME: bosc_ame.compute_predecessors = array<i64: 0, 1, 2>
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 90, 91, 92>
