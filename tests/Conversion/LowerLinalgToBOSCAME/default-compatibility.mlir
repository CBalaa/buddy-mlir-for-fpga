// RUN: buddy-opt %s --lower-linalg-to-boscame -o %t.default
// RUN: buddy-opt %s --lower-linalg-to-boscame='enable-spm-dma=false enable-ame=false' -o %t.disabled
// RUN: diff %t.default %t.disabled
// RUN: FileCheck %s --check-prefix=LEGACY < %t.default
// RUN: buddy-opt %s --linalg-to-boscame | FileCheck %s --check-prefix=INDEPENDENT

// The historical entry point retains its AME lowering with all new features
// disabled. The independent entry point uses ordinary loops for an unmarked
// matmul, even without explicitly selecting a target or DMA configuration.
module {
  func.func @unmarked(%a: memref<4x4xi32>, %b: memref<4x4xi32>,
                      %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x4xi32>, memref<4x4xi32>)
      outs(%c : memref<4x4xi32>)
    return
  }
}

// LEGACY-LABEL: func.func @unmarked
// LEGACY: bosc_ame.msettype
// LEGACY-NOT: bosc_ame.dma.start
// LEGACY: return

// INDEPENDENT-LABEL: func.func @unmarked
// INDEPENDENT-NOT: bosc_ame.
// INDEPENDENT: scf.for
// INDEPENDENT: memref.load
// INDEPENDENT: arith.muli
// INDEPENDENT: arith.addi
// INDEPENDENT: memref.store
// INDEPENDENT-NOT: bosc_ame.
// INDEPENDENT: return
