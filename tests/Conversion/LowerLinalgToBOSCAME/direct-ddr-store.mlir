// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t
// RUN: FileCheck %s --check-prefix=IR --implicit-check-not=bosc_ame.dma.start --implicit-check-not=bosc_ame.spm.view < %t
// RUN: buddy-opt %t --lower-bosc-ame | FileCheck %s --check-prefix=LLVM --implicit-check-not=bosc_ame_dma_start

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
// LLVM: "bosc_ame.intr.mlce32.m"
// LLVM: "bosc_ame.intr.mlae8.m"
// LLVM: "bosc_ame.intr.mlbe8.m"
// LLVM: mqma.b.mm
// LLVM: "bosc_ame.intr.msce32.m"
