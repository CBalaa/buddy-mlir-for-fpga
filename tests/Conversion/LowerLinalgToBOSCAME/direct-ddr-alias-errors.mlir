// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga store-policy=direct-ddr' 2>&1 | FileCheck %s

// A direct destination that is also a Linalg input has no write barrier in
// the direct vector path.  Definite root aliasing is rejected up front.
module {
  func.func @in_place(%root: memref<64xf32>) {
    memref.copy %root, %root {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64>,
                         valid_shape = array<i64: 64>, masked = false}
    } : memref<64xf32> to memref<64xf32>
    return
  }
}

// CHECK: direct-ddr store destination aliases a Linalg input
