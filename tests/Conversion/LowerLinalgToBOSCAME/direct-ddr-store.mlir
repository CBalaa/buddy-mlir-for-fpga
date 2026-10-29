// RUN: buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t
// RUN: FileCheck %s --check-prefix=IR --implicit-check-not=bosc_ame.dma.start --implicit-check-not=bosc_ame.spm.view < %t
// RUN: buddy-opt %t --lower-bosc-ame | FileCheck %s --check-prefix=LLVM --implicit-check-not=bosc_ame_dma_start
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// All three external operands meet the direct DDR address and layout proofs.
// Enabling SPM/DMA does not force staging when no private storage is needed.
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// IR: bosc_ame.tile.load_acc {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.load_a {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.load_b {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.mma
// IR: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr,
// LLVM: %[[ACC:[[:alnum:]_]+]] = llvm.call_intrinsic "llvm.riscv.bosc.ordered.load.acc"
// LLVM: %[[A:[[:alnum:]_]+]] = llvm.call_intrinsic "llvm.riscv.bosc.ordered.load.a"
// LLVM: %[[B:[[:alnum:]_]+]] = llvm.call_intrinsic "llvm.riscv.bosc.ordered.load.b"
// LLVM: %[[C:[[:alnum:]_]+]] = llvm.call_intrinsic "llvm.riscv.bosc.ordered.mma"(%[[ACC]], %[[A]], %[[B]],
// LLVM: llvm.call_intrinsic "llvm.riscv.bosc.ordered.store.acc"(%[[C]],
