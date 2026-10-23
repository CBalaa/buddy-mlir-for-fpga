// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t
// RUN: FileCheck %s --check-prefix=IR < %t
// RUN: buddy-opt %t --lower-bosc-ame | FileCheck %s --check-prefix=LLVM

// The automatic planner proves the external C store is direct DDR. A and B
// still use the typed SPM/DMA staging path, while the AME accumulator reads and writes C
// itself, so no C DMA request is emitted.
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

// IR-COUNT-2: bosc_ame.dma.start
// IR: bosc_ame.tile.load_acc {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr}
// IR-NOT: bosc_ame.dma.start
// LLVM-COUNT-2: llvm.call @bosc_ame_dma_start
// LLVM: "bosc_ame.intr.msce32.m"
// LLVM-NOT: llvm.call @bosc_ame_dma_start
