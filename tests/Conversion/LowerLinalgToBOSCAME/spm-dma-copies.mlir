// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// Rank-one private copies remain unsupported. The harness also derives legal
// DDR boundary copies, which use byte lengths and wait for each DMA start.
// Rank-two internal ACC copy routes are covered by PlanInternalCopies.py.
module {
  func.func @copy_i8() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @copy_i16() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi16, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi16, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xi16, #bosc_ame.spm<0>> to memref<64xi16, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @copy_i32() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi32, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi32, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xi32, #bosc_ame.spm<0>> to memref<64xi32, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @copy_i64() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi64, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi64, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xi64, #bosc_ame.spm<0>> to memref<64xi64, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @copy_f16() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xf16, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xf16, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xf16, #bosc_ame.spm<0>> to memref<64xf16, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @copy_bf16() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xbf16, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xbf16, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xbf16, #bosc_ame.spm<0>> to memref<64xbf16, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @copy_f32() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xf32, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xf32, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xf32, #bosc_ame.spm<0>> to memref<64xf32, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @copy_f64() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xf64, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xf64, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xf64, #bosc_ame.spm<0>> to memref<64xf64, #bosc_ame.spm<1>>
    return
  }
}

// -----

module {
  func.func @staged_identity_generic(%input: memref<16xf32>, %output: memref<16xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%input : memref<16xf32>) outs(%output : memref<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// -----

// The logical rows contain 64 bytes, but the physical pitches differ.
// Only two 64-byte packets are copied; neither source nor destination gaps
// participate. A flat 128-byte copy would read and overwrite the wrong bytes.
module {
  func.func @different_row_pitches() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<192xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<256xi8, #bosc_ame.spm<1>>
    %source_rows = memref.reinterpret_cast %source to offset: [0], sizes: [2, 64], strides: [128, 1]
      : memref<192xi8, #bosc_ame.spm<0>> to memref<2x64xi8, strided<[128, 1]>, #bosc_ame.spm<0>>
    %target_rows = memref.reinterpret_cast %target to offset: [0], sizes: [2, 64], strides: [192, 1]
      : memref<256xi8, #bosc_ame.spm<1>> to memref<2x64xi8, strided<[192, 1]>, #bosc_ame.spm<1>>
    memref.copy %source_rows, %target_rows
      : memref<2x64xi8, strided<[128, 1]>, #bosc_ame.spm<0>>
      to memref<2x64xi8, strided<[192, 1]>, #bosc_ame.spm<1>>
    return
  }
}

// -----

// One bank, one root, two disjoint subviews. Descriptor offsets must be
// retained exactly once when constructing the rank-one DMA endpoints.
module {
  func.func @same_bank_disjoint_subviews() {
    %bank = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<256xi8, #bosc_ame.spm<0>>
    %source = memref.subview %bank[64] [64] [1]
      : memref<256xi8, #bosc_ame.spm<0>> to memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    %target = memref.subview %bank[192] [64] [1]
      : memref<256xi8, #bosc_ame.spm<0>> to memref<64xi8, strided<[1], offset: 192>, #bosc_ame.spm<0>>
    memref.copy %source, %target
      : memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
      to memref<64xi8, strided<[1], offset: 192>, #bosc_ame.spm<0>>
    return
  }
}

// -----

// A typed memref.view shifts the aligned pointer in bytes; its zero descriptor
// offset must not discard or apply the 64-byte base shift twice.
module {
  func.func @shifted_typed_view() {
    %c64 = arith.constant 64 : index
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<128xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<128xi8, #bosc_ame.spm<1>>
    %source_typed = memref.view %source[%c64][]
      : memref<128xi8, #bosc_ame.spm<0>> to memref<16xi32, #bosc_ame.spm<0>>
    %target_typed = memref.view %target[%c64][]
      : memref<128xi8, #bosc_ame.spm<1>> to memref<16xi32, #bosc_ame.spm<1>>
    memref.copy %source_typed, %target_typed
      : memref<16xi32, #bosc_ame.spm<0>> to memref<16xi32, #bosc_ame.spm<1>>
    return
  }
}

// -----

// The valid subviews, rather than logical_shape, set the DMA extent. The
// masked 256-element tail keeps its previously initialized `other` values.
module {
  func.func @masked_valid_subview() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<1024xf32, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<1024xf32, #bosc_ame.spm<1>>
    %source_valid = memref.subview %source[0] [768] [1]
      : memref<1024xf32, #bosc_ame.spm<0>> to memref<768xf32, strided<[1]>, #bosc_ame.spm<0>>
    %target_valid = memref.subview %target[0] [768] [1]
      : memref<1024xf32, #bosc_ame.spm<1>> to memref<768xf32, strided<[1]>, #bosc_ame.spm<1>>
    memref.copy %source_valid, %target_valid {
      bosc_ame.memory = {kind = "load", logical_shape = array<i64: 1024>,
                        masked = true, other = 0.000000e+00 : f32,
                        valid_shape = array<i64: 768>}
    } : memref<768xf32, strided<[1]>, #bosc_ame.spm<0>>
      to memref<768xf32, strided<[1]>, #bosc_ame.spm<1>>
    return
  }
}
