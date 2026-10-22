// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s

// Direct stores are rectangular and unmasked.  A masked frontend event must
// be diagnosed rather than silently converted into a full-tile overwrite.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @masked(%src: memref<16xf32>, %dst: memref<16xf32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 16>,
                         valid_shape = array<i64: 15>, masked = true}
    } : memref<16xf32> to memref<16xf32>
    return
  }
}

// CHECK: direct-ddr store does not support masked stores
