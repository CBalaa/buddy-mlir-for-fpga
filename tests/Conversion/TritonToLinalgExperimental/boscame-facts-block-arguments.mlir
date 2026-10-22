// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.facts.mlir 2> %t.facts
// RUN: FileCheck %s --check-prefix=FACTS < %t.facts
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.plain.mlir
// RUN: diff %t.facts.mlir %t.plain.mlir

// The function input and both while-region inputs have argument number zero.
// Their references must retain the owner, region and block instead of merging
// all three into the same block-arg#0 name.
module {
  func.func @argument_owners(%input: tensor<4xf32>, %condition: i1, %upper: index)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %result = scf.while (%before = %seed) : (tensor<4xf32>) -> tensor<4xf32> {
      %next_before = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}
          ins(%before : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.condition(%condition) %next_before : tensor<4xf32>
    } do {
    ^bb0(%after: tensor<4xf32>):
      %next_after = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}
          ins(%after : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %nested = scf.for %iv = %c0 to %upper step %c1
          iter_args(%slot = %next_after) -> tensor<4xf32> {
        %next_slot = linalg.generic {
            indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
            iterator_types = ["parallel"]}
            ins(%slot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %negated = arith.negf %value : f32
            linalg.yield %negated : f32
        } -> tensor<4xf32>
        scf.yield %next_slot : tensor<4xf32>
      }
      scf.yield %nested : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }

  // Repeated operands in the add use the seed several times. Each user must
  // appear once, ordered by the report's compute/IR-operation number keys.
  func.func @another_function(%input: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, f32) {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                         affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%seed, %seed : tensor<4xf32>, tensor<4xf32>)
        outs(%seed : tensor<4xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %added = arith.addf %lhs, %rhs : f32
        linalg.yield %added : f32
    } -> tensor<4xf32>
    %scalar = tensor.extract %seed[%c0] : tensor<4xf32>
    %square = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%seed : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %squared = arith.mulf %value, %value : f32
        linalg.yield %squared : f32
    } -> tensor<4xf32>
    return %sum, %square, %scalar : tensor<4xf32>, tensor<4xf32>, f32
  }
}

// FACTS: LinalgFacts:
// FACTS: compute op#{{[0-9]+}} linalg.generic {{.*}}inputs=[ir-op#[[FUNCTION:[0-9]+]]:func.func.region#0.block#0.arg#0:tensor<4xf32>]
// FACTS: compute op#{{[0-9]+}} linalg.generic {{.*}}inputs=[ir-op#[[WHILE:[0-9]+]]:scf.while.region#0.block#0.arg#0:tensor<4xf32>]
// FACTS: compute op#{{[0-9]+}} linalg.generic {{.*}}inputs=[ir-op#[[WHILE]]:scf.while.region#1.block#0.arg#0:tensor<4xf32>]
// FACTS: compute op#{{[0-9]+}} linalg.generic {{.*}}inputs=[ir-op#{{[0-9]+}}:scf.for.region#0.block#0.arg#1:tensor<4xf32>]
// FACTS: compute op#{{[0-9]+}} linalg.generic {{.*}}inputs=[ir-op#{{[0-9]+}}:func.func.region#0.block#0.arg#0:tensor<4xf32>]
// FACTS-SAME: result-users=[result#0 users=[op#[[SUM:[0-9]+]], op#[[SQUARE:[0-9]+]], ir-op#{{[0-9]+}}:tensor.extract]]
// FACTS: compute op#[[SUM]] linalg.generic results=[tensor<4xf32>] body=[arith.addf]
// FACTS: compute op#[[SQUARE]] linalg.generic results=[tensor<4xf32>] body=[arith.mulf]
