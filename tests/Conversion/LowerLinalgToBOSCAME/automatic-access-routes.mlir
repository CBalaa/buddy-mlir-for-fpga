// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: FileCheck %s --check-prefix=IR < %t

// Each access chooses its route independently. An entry function's ABI proves
// disjoint DDR arguments; a private function has no such caller promise and
// retains a staging barrier even though its descriptors have the same shape.
module {
  func.func @direct(%src: memref<16xi32>, %dst: memref<16xi32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
        logical_shape = array<i64: 16>, valid_shape = array<i64: 16>,
        masked = false}
    } : memref<16xi32> to memref<16xi32>
    return
  }

  func.func private @staged(%src: memref<16xi32>, %dst: memref<16xi32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
        logical_shape = array<i64: 16>, valid_shape = array<i64: 16>,
        masked = false}
    } : memref<16xi32> to memref<16xi32>
    return
  }

  func.func @existing_window() {
    %zero = arith.constant 0 : i32
    %window = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
      : memref<16xi32, #bosc_ame.spm<0>>
    linalg.fill ins(%zero : i32)
      outs(%window : memref<16xi32, #bosc_ame.spm<0>>)
    return
  }
}

// PLAN: buffer-access[{{[0-9]+}}]: op=linalg.generic effect=write{{.*}}route=direct-ddr reason=complete-disjoint-external-store route-bytes=64 route-bank=-1 route-offset=0 route-capacity=0
// PLAN: buffer-access[{{[0-9]+}}]: op=linalg.generic effect=read{{.*}}route=staging reason=direct-store-alias-unproven route-bytes=64 route-bank=1 route-offset=0 route-capacity=64
// PLAN: buffer-access[{{[0-9]+}}]: op=linalg.generic effect=write{{.*}}route=staging reason=direct-store-alias-unproven route-bytes=64 route-bank=1 route-offset=64 route-capacity=64
// PLAN: buffer-access[{{[0-9]+}}]: op=linalg.fill effect=write{{.*}}route=existing-spm reason=existing-spm-window route-bytes=64 route-bank=0 route-offset=256 route-capacity=64

// IR-LABEL: func.func @direct
// IR-COUNT-1: bosc_ame.dma.start
// IR-NOT: bosc_ame.dma.start
// IR: vector.store
// IR-LABEL: func.func private @staged
// IR-COUNT-3: bosc_ame.dma.start
// IR-NOT: bosc_ame.dma.start
// IR: return
