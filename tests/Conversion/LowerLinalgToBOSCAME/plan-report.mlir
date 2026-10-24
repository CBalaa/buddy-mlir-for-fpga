// RUN: split-file %s %t
// RUN: buddy-opt %t/gemm.mlir --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: buddy-opt %t/gemm.mlir --linalg-to-boscame='dma-abi=args_aligned64_nonalias plan-report=true' -o %t.reported 2>%t.plan
// RUN: cmp %t.default %t.reported
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: not buddy-opt %t/copy.mlir --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o /dev/null 2>&1 | FileCheck %s --check-prefix=COPY

// The report reads concrete per-operand choices. One complete tile fits the
// direct AME recipes and requires no staging. An independent DDR copy has no
// v1 recipe and is retained below as an explicit compatibility boundary.
// PLAN: linalg.matmul plan.route={a = {bytes = 4096 : i64, level = "ddr"
// PLAN-SAME: accumulator = {bytes = 16384 : i64, level = "ddr"
// PLAN-SAME: b = {bytes = 4096 : i64, level = "ddr"
// PLAN-SAME: engine = "ame"
// PLAN: high_water = array<i64: 0, 0>
// PLAN-NOT: plan.transfer
// COPY: v1 has no independent DDR-to-DDR copy recipe

//--- gemm.mlir
module {
  func.func @gemm(%numPrograms0: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
                  %numPrograms1: i32 {bosc_ame.program_axis = 1 : i32, bosc_ame.program_role = "num_programs"},
                  %numPrograms2: i32 {bosc_ame.program_axis = 2 : i32, bosc_ame.program_role = "num_programs"},
                  %a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32>) {
    linalg.matmul {bosc_ame.a_origin_id = 101 : i64,
                   bosc_ame.b_origin_id = 202 : i64,
                   cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32>)
    return
  }

}

//--- copy.mlir
module {
  func.func @copy(%input: memref<16xf32>, %output: memref<16xf32>) {
    linalg.generic {
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 16>, masked = false,
                         origin_id = 23 : i64, valid_shape = array<i64: 16>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%output : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}
