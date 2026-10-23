// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s

// A masked provenance event is ineligible for direct forwarding. The
// existing full-copy IR remains available for ordinary DMA staging.
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

// CHECK: bosc_ame.dma.start
// CHECK-NOT: bosc_ame.direct_ddr
