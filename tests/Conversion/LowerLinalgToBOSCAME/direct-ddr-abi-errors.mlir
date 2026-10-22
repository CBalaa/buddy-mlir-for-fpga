// RUN: not buddy-opt %s --linalg-to-boscame='target=nr-fpga' 2>&1 | FileCheck %s

// Direct external stores use the same aligned, non-aliasing caller promise as
// typed DMA endpoints, even when no input requires SPM staging.
module {
  func.func @store(%src: memref<64xf32>, %dst: memref<64xf32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64>,
                         valid_shape = array<i64: 64>, masked = false}
    } : memref<64xf32> to memref<64xf32>
    return
  }
}

// CHECK: direct DDR AME operands/stores require dma-abi=args_aligned64_nonalias
