// RUN: not buddy-opt %s --linalg-to-boscame='target=nr-fpga' 2>&1 | FileCheck %s

// The absent disjointness promise leaves source/destination aliasing unknown.
// The contract rejects the potentially overlapping copy before route selection.
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

// CHECK: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
