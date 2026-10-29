// RUN: buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t.emitted.mlir
// RUN: FileCheck %s < %t.emitted.mlir
// RUN: buddy-opt %t.emitted.mlir --bosc-ame-spm-dma-check --lower-bosc-ame -o /dev/null
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// A static descriptor offset of one i32 is four bytes. Direct AME compute
// access is illegal, so the planner stages C through exact ACC32 copies.
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32, strided<[64, 1], offset: 1>>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32, strided<[64, 1], offset: 1>>)
    return
  }
}

// CHECK-LABEL: func.func @gemm
// CHECK: %[[C:.*]] = bosc_ame.spm.view
// CHECK: bosc_ame.matrix.copy %arg2, %[[C]]
// CHECK-SAME: row_bytes = 256 : i64, rows = 64 : i64
// CHECK: bosc_ame.tile.load_acc
// CHECK: bosc_ame.tile.mma
// CHECK: bosc_ame.tile.store_acc
// CHECK: bosc_ame.matrix.copy %[[C]], %arg2
// CHECK-SAME: source_pitch = 256 : i64
