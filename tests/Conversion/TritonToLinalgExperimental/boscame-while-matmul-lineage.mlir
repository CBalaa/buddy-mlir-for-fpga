// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.facts.mlir 2> %t.facts
// RUN: FileCheck %s --check-prefix=FACTS --implicit-check-not='a_origin_id=' --implicit-check-not='b_origin_id=' < %t.facts
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.plain.mlir 2> %t.plain.err
// RUN: test ! -s %t.plain.err
// RUN: diff %t.facts.mlir %t.plain.mlir

module {
  tt.func public @while_matmul_lineage(%a_ptr: !tt.ptr<i8>,
      %b_ptr: !tt.ptr<i8>, %replacement_ptr: !tt.ptr<i8>,
      %opaque: tensor<16x16xi8>, %condition: i1)
      -> tensor<16x16xi32> {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %rows = tt.expand_dims %r {axis = 1 : i32}
        : tensor<16xi32> -> tensor<16x1xi32>
    %cols = tt.expand_dims %r {axis = 0 : i32}
        : tensor<16xi32> -> tensor<1x16xi32>
    %stride = arith.constant dense<16> : tensor<16x1xi32>
    %row_offset = arith.muli %rows, %stride : tensor<16x1xi32>
    %cols_2d = tt.broadcast %cols : tensor<1x16xi32> -> tensor<16x16xi32>
    %a_base = tt.splat %a_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %a_row = tt.addptr %a_base, %row_offset
        : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %a_rows = tt.broadcast %a_row
        : tensor<16x1x!tt.ptr<i8>> -> tensor<16x16x!tt.ptr<i8>>
    %a_ptrs = tt.addptr %a_rows, %cols_2d
        : tensor<16x16x!tt.ptr<i8>>, tensor<16x16xi32>
    %a_load = tt.load %a_ptrs : tensor<16x16x!tt.ptr<i8>>
    %b_base = tt.splat %b_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %b_row = tt.addptr %b_base, %row_offset
        : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %b_rows = tt.broadcast %b_row
        : tensor<16x1x!tt.ptr<i8>> -> tensor<16x16x!tt.ptr<i8>>
    %b_ptrs = tt.addptr %b_rows, %cols_2d
        : tensor<16x16x!tt.ptr<i8>>, tensor<16x16xi32>
    %b_load = tt.load %b_ptrs : tensor<16x16x!tt.ptr<i8>>
    %replacement_base = tt.splat %replacement_ptr
        : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %replacement_row = tt.addptr %replacement_base, %row_offset
        : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %replacement_rows = tt.broadcast %replacement_row
        : tensor<16x1x!tt.ptr<i8>> -> tensor<16x16x!tt.ptr<i8>>
    %replacement_ptrs = tt.addptr %replacement_rows, %cols_2d
        : tensor<16x16x!tt.ptr<i8>>, tensor<16x16xi32>
    %replacement = tt.load %replacement_ptrs
        : tensor<16x16x!tt.ptr<i8>>
    %result:3 = scf.while (%before_a = %a_load, %before_b = %b_load,
        %before_opaque = %opaque)
        : (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
          -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>) {
      scf.condition(%condition) %before_b, %before_a, %before_opaque :
        tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>
    } do {
    ^bb0(%after_b: tensor<16x16xi8>, %after_a: tensor<16x16xi8>,
         %after_opaque: tensor<16x16xi8>):
      scf.yield %after_a, %after_b, %replacement :
        tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>
    }
    %zero = arith.constant dense<0> : tensor<16x16xi32>
    %first = tt.dot %result#0, %result#1, %zero
        : tensor<16x16xi8> * tensor<16x16xi8> -> tensor<16x16xi32>
    %second = tt.dot %result#0, %result#2, %first
        : tensor<16x16xi8> * tensor<16x16xi8> -> tensor<16x16xi32>
    tt.return %second : tensor<16x16xi32>
  }
}

// The condition forwarding permutes the first two while slots. Their
// after-region yields restore each recurrence; the third keeps an opaque init
// path alongside its known replacement load.
// FACTS: compute op#{{[0-9]+}} linalg.matmul {{.*}} a_source_load_origin_ids=array<i64: 1> a_source_lineage_complete=true b_source_load_origin_ids=array<i64: 0> b_source_lineage_complete=true
// FACTS: compute op#{{[0-9]+}} linalg.matmul {{.*}} a_source_load_origin_ids=array<i64: 1> a_source_lineage_complete=true b_source_load_origin_ids=array<i64: 2> b_source_lineage_complete=false
