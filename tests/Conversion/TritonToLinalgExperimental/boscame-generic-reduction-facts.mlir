// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.facts.mlir 2> %t.facts
// RUN: FileCheck %s --check-prefix=GRAPH < %t.facts.mlir
// RUN: FileCheck %s --check-prefix=FACTS < %t.facts
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.plain.mlir
// RUN: diff %t.facts.mlir %t.plain.mlir

module {
  func.func private @unknown(f32) -> f32

  // A reduction written as generic has the same stage boundary as a named
  // reduction. Its tensor semantics alone do not permit rematerialization.
  func.func @generic_reduction(%input: tensor<4x8xf32>, %init: tensor<4xf32>)
      -> tensor<4xf32> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.addf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%sum : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @multi_operation_reduction(%input: tensor<4x8xf32>,
                                       %init: tensor<4xf32>)
      -> tensor<4xf32> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %square = arith.mulf %value, %value : f32
        %next = arith.addf %square, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @zero_filled_sum(%input: tensor<4x8xf32>) -> tensor<4xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.addf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @dynamic_seed_sum(%input: tensor<4x8xf32>, %seed: f32)
      -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%seed : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.addf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @multi_output_extrema(%maximum_input: tensor<4x8xf32>,
                                  %minimum_input: tensor<4x8xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %negative_infinity = arith.constant 0xFF800000 : f32
    %positive_infinity = arith.constant 0x7F800000 : f32
    %empty_max = tensor.empty() : tensor<4xf32>
    %empty_min = tensor.empty() : tensor<4xf32>
    %init_max = linalg.fill ins(%negative_infinity : f32)
        outs(%empty_max : tensor<4xf32>) -> tensor<4xf32>
    %init_min = linalg.fill ins(%positive_infinity : f32)
        outs(%empty_min : tensor<4xf32>) -> tensor<4xf32>
    %maximum, %minimum = linalg.reduce
        ins(%maximum_input, %minimum_input : tensor<4x8xf32>, tensor<4x8xf32>)
        outs(%init_max, %init_min : tensor<4xf32>, tensor<4xf32>)
        dimensions = [1]
        (%max_value: f32, %min_value: f32,
         %max_accumulator: f32, %min_accumulator: f32) {
      %next_max = arith.maximumf %max_value, %max_accumulator : f32
      %next_min = arith.minimumf %min_value, %min_accumulator : f32
      linalg.yield %next_max, %next_min : f32, f32
    }
    return %maximum, %minimum : tensor<4xf32>, tensor<4xf32>
  }

  func.func @unmatched_multi_operation_combiner(%input: tensor<4x8xf32>,
                                                %init: tensor<4xf32>)
      -> tensor<4xf32> {
    %two = arith.constant 2.0 : f32
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %sum = arith.addf %value, %accumulator : f32
        %next = arith.mulf %sum, %two : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @nan_neutral_with_nnan(%input: tensor<4x8xf32>)
      -> tensor<4xf32> {
    %nan = arith.constant 0xFFC00000 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%nan : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %maximum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.maxnumf %value, %accumulator fastmath<nnan> : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %maximum : tensor<4xf32>
  }

  func.func @nan_neutral_without_nnan(%input: tensor<4x8xf32>)
      -> tensor<4xf32> {
    %nan = arith.constant 0xFFC00000 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%nan : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %maximum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.maxnumf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %maximum : tensor<4xf32>
  }

  func.func @vector_element_maximum(
      %input: tensor<4x8xvector<2xf32>>,
      %init: tensor<4xvector<2xf32>>) -> tensor<4xvector<2xf32>> {
    %maximum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xvector<2xf32>>)
        outs(%init : tensor<4xvector<2xf32>>) {
      ^bb0(%value: vector<2xf32>, %accumulator: vector<2xf32>):
        %next = arith.maximumf %value, %accumulator : vector<2xf32>
        linalg.yield %next : vector<2xf32>
    } -> tensor<4xvector<2xf32>>
    return %maximum : tensor<4xvector<2xf32>>
  }

  // Pure fill and all-parallel payloads retain their existing metadata.
  func.func @parallel_and_fill(%input: tensor<4xf32>, %initial: f32)
      -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %filled = linalg.fill ins(%initial : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%filled : tensor<4xf32>) {
      ^bb0(%value: f32, %initial_value: f32):
        %added = arith.addf %value, %initial_value : f32
        linalg.yield %added : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // A captured write is observable even when the enclosing Linalg operation
  // has pure tensor operands and results.
  func.func @hidden_write(%input: tensor<4xf32>, %side: memref<1xf32>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        memref.store %value, %side[%c0] : memref<1xf32>
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // Captured reads also prevent rematerialization: a later write could change
  // the value seen when replaying the payload.
  func.func @hidden_read(%input: tensor<4xf32>, %side: memref<1xf32>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %captured = memref.load %side[%c0] : memref<1xf32>
        %added = arith.addf %value, %captured : f32
        linalg.yield %added : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @unknown_call(%input: tensor<4xf32>) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %called = func.call @unknown(%value) : (f32) -> f32
        linalg.yield %called : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// GRAPH-LABEL: func.func @generic_reduction
// GRAPH-SAME: bosc_ame.compute_graph_complete = true
// GRAPH: linalg.generic {{.*}}bosc_ame.compute_barrier_after = true
// GRAPH-SAME: bosc_ame.compute_id = [[REDUCE:[0-9]+]] : i64
// GRAPH-SAME: bosc_ame.compute_kind = "reduction"
// GRAPH-SAME: bosc_ame.compute_stage = 0 : i64
// GRAPH-SAME: bosc_ame.rematerializable = false
// GRAPH: linalg.generic {{.*}}bosc_ame.compute_barrier_after = false
// GRAPH-SAME: bosc_ame.compute_kind = "elementwise"
// GRAPH-SAME: bosc_ame.compute_predecessors = array<i64: [[REDUCE]]>
// GRAPH-SAME: bosc_ame.compute_stage = 1 : i64
// GRAPH-SAME: bosc_ame.rematerializable = true

// FACTS: op=linalg.generic {{.*}} kind=reduction {{.*}} reduction-dimensions=[1] reduction-structure={complete=true input-args=[{{[0-9]+}}] init-args=[{{[0-9]+}}] init-values=[{{[0-9]+}}] yields=[{{[0-9]+}}] reducer-body-ops=[{{[0-9]+}}] output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=0.000000e+00 : f32 seed-neutral=unknown}]}
// FACTS: op=linalg.generic {{.*}} kind=reduction {{.*}} reduction-dimensions=[1] reduction-structure={complete=true input-args=[{{[0-9]+}}] init-args=[{{[0-9]+}}] init-values=[{{[0-9]+}}] yields=[{{[0-9]+}}] reducer-body-ops=[{{[0-9]+}}, {{[0-9]+}}] output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=0.000000e+00 : f32 seed-neutral=unknown}]}
// FACTS: {{.*}}output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=0.000000e+00 : f32 seed-neutral=matches}]
// FACTS: {{.*}}output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=0.000000e+00 : f32 seed-neutral=unknown}]
// FACTS: {{.*}}output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=0xFF800000 : f32 seed-neutral=matches}, {output-index=1 combiner-op={{[0-9]+}} neutral=0x7F800000 : f32 seed-neutral=matches}]
// FACTS: {{.*}}output-facts=[{output-index=0 combiner-op=<unknown> neutral=<unknown> seed-neutral=unknown}]
// FACTS: {{.*}}output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=<unknown> seed-neutral=unknown}]
// FACTS: {{.*}}output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=0xFFC00000 : f32 seed-neutral=matches}]
// FACTS: {{.*}}output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=<unknown> seed-neutral=unknown}]

// GRAPH-LABEL: func.func @parallel_and_fill
// GRAPH-SAME: bosc_ame.compute_graph_complete = true
// GRAPH: linalg.fill {{.*}}bosc_ame.compute_barrier_after = false
// GRAPH-SAME: bosc_ame.compute_kind = "fill"
// GRAPH-SAME: bosc_ame.compute_stage = 0 : i64
// GRAPH-SAME: bosc_ame.rematerializable = true
// GRAPH: linalg.generic {{.*}}bosc_ame.compute_barrier_after = false
// GRAPH-SAME: bosc_ame.compute_kind = "elementwise"
// GRAPH-SAME: bosc_ame.compute_stage = 0 : i64
// GRAPH-SAME: bosc_ame.rematerializable = true

// GRAPH-LABEL: func.func @hidden_write
// GRAPH-SAME: bosc_ame.compute_graph_complete = false
// GRAPH: linalg.generic {{.*}}bosc_ame.compute_kind = "elementwise"
// GRAPH-SAME: bosc_ame.rematerializable = false
// GRAPH: memref.store

// GRAPH-LABEL: func.func @hidden_read
// GRAPH-SAME: bosc_ame.compute_graph_complete = false
// GRAPH: linalg.generic {{.*}}bosc_ame.compute_kind = "elementwise"
// GRAPH-SAME: bosc_ame.rematerializable = false
// GRAPH: memref.load

// GRAPH-LABEL: func.func @unknown_call
// GRAPH-SAME: bosc_ame.compute_graph_complete = false
// GRAPH: linalg.generic {{.*}}bosc_ame.compute_kind = "elementwise"
// GRAPH-SAME: bosc_ame.rematerializable = false
// GRAPH: func.call @unknown
