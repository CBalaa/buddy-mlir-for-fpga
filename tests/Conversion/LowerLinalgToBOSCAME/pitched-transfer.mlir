// RUN: split-file %s %t
// RUN: buddy-opt %t/pitched.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t/pitched.out.mlir
// RUN: FileCheck %s --check-prefix=PITCH --implicit-check-not=bosc_ame.dma.start --implicit-check-not=bosc_ame.spm.view < %t/pitched.out.mlir
// RUN: buddy-opt %t/pitched.out.mlir --lower-bosc-ame | FileCheck %s --check-prefix=LOWERED
// RUN: not buddy-opt %t/static-offset.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=COPY

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
// LOWERED: bosc_ame.intr.mlbe8.m
// LOWERED: llvm.inline_asm has_side_effects
// LOWERED-SAME: mqma.b.mm
// LOWERED: llvm.return

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
