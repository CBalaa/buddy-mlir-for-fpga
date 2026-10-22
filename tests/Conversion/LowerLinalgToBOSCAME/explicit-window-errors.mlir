// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s

// The individual windows are valid SPM ranges, but their bank-relative byte
// intervals overlap. The planner must reject the conflicting reservation
// before it places any generated workspace.
module {
  func.func @overlapping_windows() {
    %first = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<0>>
    %second = bosc_ame.spm.view {bank = 0 : i32, offset = 32 : i64}
      : memref<64xi8, #bosc_ame.spm<0>>
    return
  }
}

// CHECK: overlapping explicit SPM windows in bank 0
