// RUN: not buddy-opt --allow-unregistered-dialect %s --bosc-ame-spm-dma-check 2>&1 | FileCheck %s

module {
  func.func @unknown_spm_user() {
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    "test.unknown_spm_user"(%spm)
        : (memref<64xi8, #bosc_ame.spm<0>>) -> ()
    return
  }
}

// CHECK: unknown operation uses SPM memory
