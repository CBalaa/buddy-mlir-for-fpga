// RUN: split-file %s %t
// RUN: buddy-opt %t/pitched.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t/pitched.out.mlir
// RUN: FileCheck %s --check-prefix=PITCH --implicit-check-not=bosc_ame.dma.start --implicit-check-not=bosc_ame.spm.view < %t/pitched.out.mlir
// RUN: buddy-opt %t/pitched.out.mlir --lower-bosc-ame | FileCheck %s --check-prefix=LOWERED
// RUN: buddy-opt %t/pitched.out.mlir --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=IR
// RUN: not buddy-opt %t/static-offset.mlir --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=COPY
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// Direct DDR B retains its 128-byte parent pitch for a 64x32 logical tile.
// PITCH-LABEL: func.func @pitched_b
// PITCH-SAME: %[[A:[[:alnum:]_]+]]: memref<64x64xi8>
// PITCH-SAME: %[[B:[[:alnum:]_]+]]: memref<64x32xi8, strided<[1, 128]>>
// PITCH: %[[B_TILE:.*]] = memref.subview %[[B]][0, 0] [64, 32] [1, 1]
// PITCH-SAME: to memref<64x32xi8, strided<[1, 128]>>
// PITCH: bosc_ame.tile.load_b %[[B_TILE]] {bosc_ame.direct_ddr} : memref<64x32xi8, strided<[1, 128]>>
// PITCH: bosc_ame.tile.mma
// PITCH: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr,
// LOWERED-LABEL: llvm.func @pitched_b
// LOWERED: llvm.riscv.bosc.ordered.load.b
// LOWERED: llvm.riscv.bosc.ordered.mma
// LOWERED: llvm.return

// Pin the physical pitch and all three unequal geometry/configuration roles
// after descriptor lowering, where the actual intrinsic operands are literal.
// IR-LABEL: define void @pitched_b(
// IR: %[[ACC:[[:alnum:]_]+]] = call <vscale x 32 x i32> @llvm.riscv.bosc.ordered.load.acc.nxv32i32(ptr {{.*}}, i64 128, i64 64, i64 32, i64 65602)
// IR: %[[A:[[:alnum:]_]+]] = call <vscale x 128 x i8> @llvm.riscv.bosc.ordered.load.a.nxv128i8(ptr {{.*}}, i64 64, i64 64, i64 64, i64 65552)
// IR: %[[B:[[:alnum:]_]+]] = call <vscale x 128 x i8> @llvm.riscv.bosc.ordered.load.b.nxv128i8(ptr {{.*}}, i64 128, i64 32, i64 64, i64 65552)
// IR: %[[C:[[:alnum:]_]+]] = call <vscale x 32 x i32> @llvm.riscv.bosc.ordered.mma.nxv32i32.nxv128i8(<vscale x 32 x i32> %[[ACC]], <vscale x 128 x i8> %[[A]], <vscale x 128 x i8> %[[B]], i64 64, i64 32, i64 64, i64 65552)
// IR: call void @llvm.riscv.bosc.ordered.store.acc.nxv32i32(<vscale x 32 x i32> %[[C]], ptr {{.*}}, i64 128, i64 64, i64 32, i64 65602)

// The original static-offset identity generic normalizes to DDR-to-DDR copy.
// V1 rejects that boundary before endpoint flattening; typed DMA offset proofs
// are covered by PlanEmitValidation.py and the BOSCAME dialect tests.
// COPY: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint

//--- pitched.mlir
module {
  func.func @pitched_b(%a: memref<64x64xi8>,
                       %b: memref<64x32xi8, strided<[1, 128]>>,
                       %c: memref<64x32xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x32xi8, strided<[1, 128]>>)
      outs(%c : memref<64x32xi32>)
    return
  }
}

//--- static-offset.mlir
module {
  func.func @static_offset(%input: memref<64xi8, strided<[1], offset: 64>>,
                           %out: memref<64xi8>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<64xi8, strided<[1], offset: 64>>)
      outs(%out : memref<64xi8>) {
    ^bb0(%x: i8, %unused: i8):
      linalg.yield %x : i8
    }
    return
  }
}
