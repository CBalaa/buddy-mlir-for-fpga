// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.facts.mlir 2> %t.facts
// RUN: FileCheck %s --check-prefix=FACTS --implicit-check-not='a_origin_id=' < %t.facts
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.plain.mlir 2> %t.plain.err
// RUN: test ! -s %t.plain.err
// RUN: diff %t.facts.mlir %t.plain.mlir

module {
  tt.func public @switch_matmul_lineage(
      %a_case_ptr: !tt.ptr<i8>, %a_default_ptr: !tt.ptr<i8>,
      %b_ptr: !tt.ptr<i8>, %selector: index) -> tensor<16x16xi32> {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32}
        : tensor<16xi32>
    %rows = tt.expand_dims %range {axis = 1 : i32}
        : tensor<16xi32> -> tensor<16x1xi32>
    %cols = tt.expand_dims %range {axis = 0 : i32}
        : tensor<16xi32> -> tensor<1x16xi32>
    %stride = arith.constant dense<16> : tensor<16x1xi32>
    %row_offsets = arith.muli %rows, %stride : tensor<16x1xi32>
    %cols_2d = tt.broadcast %cols
        : tensor<1x16xi32> -> tensor<16x16xi32>
    %a_case_base = tt.splat %a_case_ptr
        : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %a_case_rows = tt.addptr %a_case_base, %row_offsets
        : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %a_case_rows_2d = tt.broadcast %a_case_rows
        : tensor<16x1x!tt.ptr<i8>> -> tensor<16x16x!tt.ptr<i8>>
    %a_case_ptrs = tt.addptr %a_case_rows_2d, %cols_2d
        : tensor<16x16x!tt.ptr<i8>>, tensor<16x16xi32>
    %a_case = tt.load %a_case_ptrs : tensor<16x16x!tt.ptr<i8>>
    %a_default_base = tt.splat %a_default_ptr
        : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %a_default_rows = tt.addptr %a_default_base, %row_offsets
        : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %a_default_rows_2d = tt.broadcast %a_default_rows
        : tensor<16x1x!tt.ptr<i8>> -> tensor<16x16x!tt.ptr<i8>>
    %a_default_ptrs = tt.addptr %a_default_rows_2d, %cols_2d
        : tensor<16x16x!tt.ptr<i8>>, tensor<16x16xi32>
    %a_default = tt.load %a_default_ptrs
        : tensor<16x16x!tt.ptr<i8>>
    %a = scf.index_switch %selector -> tensor<16x16xi8>
      case 0 {
        scf.yield %a_case : tensor<16x16xi8>
      }
      default {
        scf.yield %a_default : tensor<16x16xi8>
      }
    %b_base = tt.splat %b_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %b_rows = tt.addptr %b_base, %row_offsets
        : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %b_rows_2d = tt.broadcast %b_rows
        : tensor<16x1x!tt.ptr<i8>> -> tensor<16x16x!tt.ptr<i8>>
    %b_ptrs = tt.addptr %b_rows_2d, %cols_2d
        : tensor<16x16x!tt.ptr<i8>>, tensor<16x16xi32>
    %b = tt.load %b_ptrs : tensor<16x16x!tt.ptr<i8>>
    %zero = arith.constant dense<0> : tensor<16x16xi32>
    %result = tt.dot %a, %b, %zero
        : tensor<16x16xi8> * tensor<16x16xi8> -> tensor<16x16xi32>
    tt.return %result : tensor<16x16xi32>
  }
}

// FACTS: compute op#{{[0-9]+}} linalg.matmul {{.*}} a_source_load_origin_ids=array<i64: 0, 1> a_source_lineage_complete=true b_source_load_origin_ids=array<i64: 2> b_source_lineage_complete=true
