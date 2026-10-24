// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s

// A self-copy has identical addresses, so the contract accepts its aliasing.
// The retained DDR-to-DDR boundary has no independent V1 copy recipe.
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

// CHECK: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
