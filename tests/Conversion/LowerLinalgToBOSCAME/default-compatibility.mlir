// RUN: buddy-opt %s --lower-linalg-to-boscame -o %t.default
// RUN: buddy-opt %s --linalg-to-boscame -o %t.alias
// RUN: diff %t.default %t.alias
// RUN: FileCheck %s < %t.default
// RUN: not buddy-opt %s --lower-linalg-to-boscame='enable-spm-dma=false enable-ame=false' 2>&1 | FileCheck %s --check-prefix=DISABLED

// Both compatibility names use the same contract, planner and emitter.
// Ordinary integer matmul retains its ordered scalar payload. Disabled legacy
// engine overrides are diagnosed rather than silently selecting another path.
module {
  func.func @unmarked(%a: memref<4x4xi32>, %b: memref<4x4xi32>,
                      %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x4xi32>, memref<4x4xi32>)
      outs(%c : memref<4x4xi32>)
    return
  }
}

// CHECK-LABEL: func.func @unmarked
// CHECK-NOT: bosc_ame.tile
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: memref.store
// CHECK-NOT: bosc_ame.tile
// CHECK: return
// DISABLED: remove disabled legacy feature overrides
