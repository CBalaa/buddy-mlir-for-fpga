// RUN: buddy-opt %s --split-input-file --verify-diagnostics --bosc-ame-spm-dma-check -o /dev/null

// A legal SPM view does not authorize a reinterpret alias beyond the bank.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @spm_reinterpret_escapes_bank(%src: memref<1x16xi32>) {
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 131008 : i64} : memref<16xi32, #bosc_ame.spm<0>>
    %alias = memref.reinterpret_cast %spm to offset: [16], sizes: [1, 16], strides: [16, 1] : memref<16xi32, #bosc_ame.spm<0>> to memref<1x16xi32, strided<[16, 1], offset: 16>, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy SPM range must be statically bounded inside its bank}}
    bosc_ame.matrix.copy %src, %alias {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x16xi32>, memref<1x16xi32, strided<[16, 1], offset: 16>, #bosc_ame.spm<0>>
    return
  }
}

// -----

// The tile offsets themselves must be included, beyond the descriptor base.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @copy_offset_escapes_spm_bank(%src: memref<2x16xi32>) {
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 131008 : i64} : memref<16xi32, #bosc_ame.spm<0>>
    %alias = memref.reinterpret_cast %spm to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<16xi32, #bosc_ame.spm<0>> to memref<2x16xi32, strided<[16, 1]>, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy SPM range must be statically bounded inside its bank}}
    bosc_ame.matrix.copy %src, %alias {source_offset = 64 : i64, target_offset = 64 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32>, memref<2x16xi32, strided<[16, 1]>, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @copy_offset_escapes_ddr_root(%src: memref<16xi32>) {
    %alias = memref.reinterpret_cast %src to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<16xi32> to memref<2x16xi32, strided<[16, 1]>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy DDR range must stay inside its statically bounded entry root}}
    bosc_ame.matrix.copy %alias, %spm {source_offset = 64 : i64, target_offset = 64 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, strided<[16, 1]>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

// Logical alias bounds cannot authorize reads beyond a static DDR root.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @ddr_reinterpret_escapes_root(%src: memref<16xi32>) {
    %alias = memref.reinterpret_cast %src to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<16xi32> to memref<2x16xi32, strided<[16, 1]>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy DDR range must stay inside its statically bounded entry root}}
    bosc_ame.matrix.copy %alias, %spm {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, strided<[16, 1]>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

// The same oversized view is legal when this actual tile fits the DDR root.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @ddr_tile_fits_oversized_alias(%src: memref<16xi32>) {
    %alias = memref.reinterpret_cast %src to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<16xi32> to memref<2x16xi32, strided<[16, 1]>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    bosc_ame.matrix.copy %alias, %spm {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32, strided<[16, 1]>>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

// SPM bank bounds also apply to the actual tile, not unused rows of an alias.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @spm_tile_fits_oversized_alias(%src: memref<2x16xi32>) {
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 131008 : i64} : memref<16xi32, #bosc_ame.spm<0>>
    %alias = memref.reinterpret_cast %spm to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<16xi32, #bosc_ame.spm<0>> to memref<2x16xi32, strided<[16, 1]>, #bosc_ame.spm<0>>
    bosc_ame.matrix.copy %src, %alias {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<2x16xi32>, memref<2x16xi32, strided<[16, 1]>, #bosc_ame.spm<0>>
    return
  }
}

// -----

// A byte view changes the pointer independently of its descriptor offset.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @byte_view_fits_root(%src: memref<128xi8>) {
    %shift = arith.constant 4 : index
    %view = memref.view %src[%shift][] : memref<128xi8> to memref<1x16xi32>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x16xi32, #bosc_ame.spm<0>>
    bosc_ame.matrix.copy %view, %spm {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x16xi32>, memref<1x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @byte_view_escapes_root(%src: memref<64xi8>) {
    %shift = arith.constant 4 : index
    %view = memref.view %src[%shift][] : memref<64xi8> to memref<1x16xi32>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy DDR range must stay inside its statically bounded entry root}}
    bosc_ame.matrix.copy %view, %spm {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x16xi32>, memref<1x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @spm_byte_view_escapes_bank(%src: memref<1x16xi32>) {
    %shift = arith.constant 64 : index
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 131008 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %view = memref.view %spm[%shift][] : memref<64xi8, #bosc_ame.spm<0>> to memref<1x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy SPM range must be statically bounded inside its bank}}
    bosc_ame.matrix.copy %src, %view {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x16xi32>, memref<1x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

// Natural word alignment alone does not authorize a negative pointer shift.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @negative_byte_shift(%src: memref<128xi8>) {
    %shift = arith.constant -4 : index
    %view = memref.view %src[%shift][] : memref<128xi8> to memref<1x16xi32>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy endpoint byte range cannot be resolved against its backing storage}}
    bosc_ame.matrix.copy %view, %spm {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x16xi32>, memref<1x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

// Individually aligned pointer shifts must not wrap while tracing the root.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @overflow_byte_shifts(%src: memref<8xi8>) {
    %large = arith.constant 9223372036854775804 : index
    %four = arith.constant 4 : index
    %first = memref.view %src[%large][] : memref<8xi8> to memref<8xi8>
    %view = memref.view %first[%four][] : memref<8xi8> to memref<1x1xi32>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x1xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy endpoint byte range cannot be resolved against its backing storage}}
    bosc_ame.matrix.copy %view, %spm {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 4 : i64, source_pitch = 4 : i64, target_pitch = 4 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

// Element-to-byte descriptor offset conversion must reject i64 overflow.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @overflow_descriptor_offset(%src: memref<16xi32>) {
    %alias = memref.reinterpret_cast %src to offset: [2305843009213693952], sizes: [1, 16], strides: [16, 1] : memref<16xi32> to memref<1x16xi32, strided<[16, 1], offset: 2305843009213693952>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{matrix copy endpoint byte range cannot be resolved against its backing storage}}
    bosc_ame.matrix.copy %alias, %spm {source_offset = 0 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x16xi32, strided<[16, 1], offset: 2305843009213693952>>, memref<1x16xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

// Negative offsets relative to an endpoint fail the local rectangle contract.
module attributes {bosc_ame.target = "nr-fpga", contract.abi = {args_aligned_bytes = 64 : i64}} {
  func.func @negative_copy_offset(%src: memref<1x16xi32>) {
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x16xi32, #bosc_ame.spm<0>>
    // expected-error@+1 {{requires matching in-bounds row/column origins}}
    bosc_ame.matrix.copy %src, %spm {source_offset = -4 : i64, target_offset = 0 : i64, rows = 1 : i64, row_bytes = 64 : i64, source_pitch = 64 : i64, target_pitch = 64 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, register_class = "acc", mtype = 65602 : i64} : memref<1x16xi32>, memref<1x16xi32, #bosc_ame.spm<0>>
    return
  }
}
