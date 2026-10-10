// RUN: buddy-opt %s | FileCheck %s
// RUN: buddy-opt %s | buddy-opt | FileCheck %s
// RUN: buddy-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// The SPM storage space is a memref memory space and the DMA request token is a
// dialect type.  Both use a mnemonic form and must survive a parse/print round
// trip, including the generic form used by tooling that does not know the
// dialect's custom printer.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  // CHECK-LABEL: func.func @ops
  func.func @ops(%ddr: memref<4096xi8>) {
    // CHECK: %[[SPM:[a-zA-Z0-9_]+]] = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64} : memref<4032xi8, #bosc_ame.spm<0>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64} : memref<4032xi8, #bosc_ame.spm<0>>
    // CHECK: %[[LEN:[a-zA-Z0-9_]+]] = arith.constant 4032 : index
    %len = arith.constant 4032 : index
    // CHECK: %[[TOKEN:[a-zA-Z0-9_]+]] = bosc_ame.dma.start %{{[a-zA-Z0-9_]+}}, %[[SPM]], %[[LEN]] {channel = 1 : i32} : (memref<4096xi8>, memref<4032xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<1>
    %token = bosc_ame.dma.start %ddr, %spm, %len {channel = 1 : i32} : (memref<4096xi8>, memref<4032xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<1>
    // CHECK: bosc_ame.dma.wait %[[TOKEN]] : !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %token : !bosc_ame.dma_token<1>
    // CHECK: return
    return
  }
}

// GENERIC: "bosc_ame.spm.view"() <{bank = 0 : i32, offset = 64 : i64}> : () -> memref<4032xi8, #bosc_ame.spm<0>>
// GENERIC: "bosc_ame.dma.start"
// GENERIC-SAME: -> !bosc_ame.dma_token<1>
// GENERIC: "bosc_ame.dma.wait"
// GENERIC-SAME: (!bosc_ame.dma_token<1>) -> ()
