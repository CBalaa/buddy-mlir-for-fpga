// RUN: split-file %s %t
// RUN: not buddy-opt %t/direct.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=COPY
// RUN: not buddy-opt %t/staged.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=COPY
// RUN: buddy-opt %t/window.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t/window.out 2>%t/window.plan
// RUN: FileCheck %s --check-prefix=WINDOW < %t/window.out
// RUN: buddy-opt %t/window.out --bosc-ame-spm-dma-check --lower-bosc-ame -o /dev/null

// Preserve the original functions. Independent external copies have no V1
// recipe; the explicit SPM window retains its bank/offset and initialization.
// Mixed per-access DDR/SPM routes and reasons are covered by PlanRoutes.py.
// COPY: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// WINDOW-LABEL: func.func @existing_window
// WINDOW: bosc_ame.spm.view {bank = 0 : i32,
// WINDOW-SAME: offset = 256 : i64} : memref<16xi32, #bosc_ame.spm<0>>
// WINDOW: vector.store
// WINDOW-NOT: bosc_ame.dma.start

//--- direct.mlir
module {
  func.func @direct(%src: memref<16xi32>, %dst: memref<16xi32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
        logical_shape = array<i64: 16>, valid_shape = array<i64: 16>,
        masked = false}
    } : memref<16xi32> to memref<16xi32>
    return
  }
}

//--- staged.mlir
module {
  func.func private @staged(%src: memref<16xi32>, %dst: memref<16xi32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
        logical_shape = array<i64: 16>, valid_shape = array<i64: 16>,
        masked = false}
    } : memref<16xi32> to memref<16xi32>
    return
  }
}

//--- window.mlir
module {
  func.func @existing_window() {
    %zero = arith.constant 0 : i32
    %window = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
      : memref<16xi32, #bosc_ame.spm<0>>
    linalg.fill ins(%zero : i32)
      outs(%window : memref<16xi32, #bosc_ame.spm<0>>)
    return
  }
}
