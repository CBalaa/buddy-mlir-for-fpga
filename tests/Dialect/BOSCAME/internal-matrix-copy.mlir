// RUN: buddy-opt %s --split-input-file --verify-diagnostics --bosc-ame-spm-dma-check -o /dev/null

// Internal copies reserve one transient ACC32 and do not require an entry DDR ABI.
module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @same_bank_independent_windows() {
    %src = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 128 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    bosc_ame.matrix.copy %src, %dst {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @different_banks() {
    %src = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<1>>
    bosc_ame.matrix.copy %src, %dst {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<1>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @same_buffer_rejected() {
    %src = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{internal matrix copy requires independent, nonoverlapping complete SPM endpoint ranges}}
    bosc_ame.matrix.copy %src, %src {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @partial_overlap_rejected() {
    %src = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{internal matrix copy requires independent, nonoverlapping complete SPM endpoint ranges}}
    bosc_ame.matrix.copy %src, %dst {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @whole_endpoint_overlap_rejected() {
    %src = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{internal matrix copy requires independent, nonoverlapping complete SPM endpoint ranges}}
    bosc_ame.matrix.copy %src, %dst {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @spm_window_bounds_not_bank_bounds() {
    %src = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 128 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %alias = memref.reinterpret_cast %src to offset: [16], sizes: [2, 16], strides: [16, 1] : memref<2x16xi32, #bosc_ame.spm<0>> to memref<2x16xi32, strided<[16, 1], offset: 16>, #bosc_ame.spm<0>>
    // expected-error@+1 {{internal matrix copy requires static endpoint ranges bounded by their declared SPM windows and banks}}
    bosc_ame.matrix.copy %alias, %dst {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, strided<[16, 1], offset: 16>, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @unsupported_tr_class() {
    %src = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 128 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{supports only one raw ACC32 register}}
    bosc_ame.matrix.copy %src, %dst {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "tr", mtype = 65602 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}
