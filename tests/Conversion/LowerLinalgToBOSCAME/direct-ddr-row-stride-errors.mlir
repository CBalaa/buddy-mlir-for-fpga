// RUN: split-file %s %t
// RUN: not buddy-opt %t/vector.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=VECTOR
// RUN: buddy-opt %t/ame.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t/ame.emitted.mlir
// RUN: FileCheck %s --check-prefix=AME < %t/ame.emitted.mlir
// RUN: buddy-opt %t/ame.emitted.mlir --bosc-ame-spm-dma-check --lower-bosc-ame -o /dev/null
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// A direct vector store advances by the physical row stride.  65 f32 values
// per row put the second row at byte offset 260, which is not a cache line.
// The identity copy is rejected at the V1 DDR-to-DDR boundary. The AME
// case reaches route selection and stages the non-cache-line pitch via ACC32.
//--- vector.mlir
module {
  func.func @copy(%src: memref<64x64xf32>,
                  %dst: memref<64x64xf32, strided<[65, 1]>>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64, 64>,
                         valid_shape = array<i64: 64, 64>, masked = false}
    } : memref<64x64xf32> to memref<64x64xf32, strided<[65, 1]>>
    return
  }
}

// A direct AME accumulator uses the same physical row-pitch contract for
// msce32.m.  The byte stride here is 65 * sizeof(i32) = 260 bytes.
//--- ame.mlir
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32, strided<[65, 1]>>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32, strided<[65, 1]>>)
    return
  }
}

// VECTOR: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// AME-LABEL: func.func @gemm
// AME: %[[C:.*]] = bosc_ame.spm.view
// AME: bosc_ame.matrix.copy %arg2, %[[C]]
// AME-SAME: source_pitch = 260 : i64
// AME-SAME: target_pitch = 256 : i64
// AME: bosc_ame.tile.load_acc
// AME: bosc_ame.tile.mma
// AME: bosc_ame.tile.store_acc
// AME: bosc_ame.matrix.copy %[[C]], %arg2
// AME-SAME: source_pitch = 256 : i64
// AME-SAME: target_pitch = 260 : i64
