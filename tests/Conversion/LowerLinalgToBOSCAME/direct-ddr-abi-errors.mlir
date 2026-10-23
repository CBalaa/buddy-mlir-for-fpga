// RUN: buddy-opt %s --linalg-to-boscame='target=nr-fpga' 2>&1 | FileCheck %s

// Without an aligned, non-aliasing caller promise the planner cannot prove
// a direct forwarding route. Ordinary buffer loops retain the copy semantics.
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

// CHECK-NOT: bosc_ame.direct_ddr
// CHECK-NOT: bosc_ame.dma.start
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
