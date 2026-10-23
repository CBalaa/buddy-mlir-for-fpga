// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t
// RUN: FileCheck %s --check-prefix=IR < %t
// RUN: buddy-opt %t --lower-bosc-ame | FileCheck %s --check-prefix=LLVM

// A dense external logical A/B event is materialized directly from DDR.  The
// C load/store is fused into the accumulator, so this complete tile has no DMA
// requests or compiler-owned snapshots.
module attributes {
  bosc_ame.dma.abi = "args_aligned64_nonalias",
  bosc_ame.enable_ame = true,
  bosc_ame.enable_spm_dma = true,
  bosc_ame.placement = "logical",
  bosc_ame.target = "nr-fpga"
} {
  func.func @logical_gemm(
      %a: memref<64x128xi8>,
      %b: memref<128x64xi8, strided<[1, 128]>>,
      %c: memref<64x64xi32>) {
    linalg.matmul {bosc_ame.lower_to_ame,
                   cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<64x128xi8>,
                   memref<128x64xi8, strided<[1, 128]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// IR-NOT: bosc_ame.dma.start
// IR-NOT: memref.alloc
// IR: bosc_ame.tile.load_acc {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.load_a {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.load_b {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.mma
// IR: bosc_ame.tile.load_a {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.load_b {{.*}} {bosc_ame.direct_ddr}
// IR: bosc_ame.tile.mma
// IR: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr}
// LLVM-NOT: llvm.call @bosc_ame_dma_start
// LLVM: "bosc_ame.intr.mlce32.m"
// LLVM: "bosc_ame.intr.mlae8.m"
// LLVM: "bosc_ame.intr.mlbe8.m"
// LLVM: mqma.b.mm
// LLVM: mqma.b.mm
// LLVM: "bosc_ame.intr.msce32.m"
