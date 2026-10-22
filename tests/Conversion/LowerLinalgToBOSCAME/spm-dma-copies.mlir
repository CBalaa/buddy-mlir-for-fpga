// RUN: buddy-opt %s --split-input-file --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' --canonicalize | FileCheck %s --implicit-check-not=vector.load --implicit-check-not=vector.store --implicit-check-not=memref.copy --implicit-check-not=linalg.generic

// Pure copies use the DMA data path, including typed copies that do not
// perform arithmetic. Lengths are bytes, and every start is waited immediately.
// CHECK-LABEL: func.func @copy_i8
// CHECK: %[[LEN_I8:.*]] = arith.constant 64 : index
// CHECK: %[[TOKEN_I8:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_I8]]
// CHECK-SAME: (memref<64xi8, {{.*}}#bosc_ame.spm<0>>, memref<64xi8, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_I8]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @copy_i16
// CHECK: %[[LEN_I16:.*]] = arith.constant 128 : index
// CHECK: %[[TOKEN_I16:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_I16]]
// CHECK-SAME: (memref<64xi16, {{.*}}#bosc_ame.spm<0>>, memref<64xi16, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_I16]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @copy_i32
// CHECK: %[[LEN_I32:.*]] = arith.constant 256 : index
// CHECK: %[[TOKEN_I32:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_I32]]
// CHECK-SAME: (memref<64xi32, {{.*}}#bosc_ame.spm<0>>, memref<64xi32, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_I32]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @copy_i64
// CHECK: %[[LEN_I64:.*]] = arith.constant 512 : index
// CHECK: %[[TOKEN_I64:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_I64]]
// CHECK-SAME: (memref<64xi64, {{.*}}#bosc_ame.spm<0>>, memref<64xi64, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_I64]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @copy_f16
// CHECK: %[[LEN_F16:.*]] = arith.constant 128 : index
// CHECK: %[[TOKEN_F16:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_F16]]
// CHECK-SAME: (memref<64xf16, {{.*}}#bosc_ame.spm<0>>, memref<64xf16, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_F16]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @copy_bf16
// CHECK: %[[LEN_BF16:.*]] = arith.constant 128 : index
// CHECK: %[[TOKEN_BF16:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_BF16]]
// CHECK-SAME: (memref<64xbf16, {{.*}}#bosc_ame.spm<0>>, memref<64xbf16, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_BF16]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @copy_f32
// CHECK: %[[LEN_F32:.*]] = arith.constant 256 : index
// CHECK: %[[TOKEN_F32:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_F32]]
// CHECK-SAME: (memref<64xf32, {{.*}}#bosc_ame.spm<0>>, memref<64xf32, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_F32]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @copy_f64
// CHECK: %[[LEN_F64:.*]] = arith.constant 512 : index
// CHECK: %[[TOKEN_F64:.*]] = bosc_ame.dma.start {{.*}}, %[[LEN_F64]]
// CHECK-SAME: (memref<64xf64, {{.*}}#bosc_ame.spm<0>>, memref<64xf64, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[TOKEN_F64]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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

// CHECK-LABEL: func.func @staged_identity_generic
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: %[[COPY:.*]] = bosc_ame.dma.start
// CHECK-SAME: (memref<16xf32, {{.*}}#bosc_ame.spm<0>>, memref<16xf32, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[COPY]]
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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
// CHECK-LABEL: func.func @different_row_pitches
// CHECK: %[[ROW_BYTES:.*]] = arith.constant 64 : index
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [64], strides: [1]
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [64], strides: [1]
// CHECK: %[[ROW0:.*]] = bosc_ame.dma.start {{.*}}, %[[ROW_BYTES]]
// CHECK-NEXT: bosc_ame.dma.wait %[[ROW0]]
// CHECK: memref.reinterpret_cast {{.*}} to offset: [128], sizes: [64], strides: [1]
// CHECK: memref.reinterpret_cast {{.*}} to offset: [192], sizes: [64], strides: [1]
// CHECK: %[[ROW1:.*]] = bosc_ame.dma.start {{.*}}, %[[ROW_BYTES]]
// CHECK-NEXT: bosc_ame.dma.wait %[[ROW1]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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
// CHECK-LABEL: func.func @same_bank_disjoint_subviews
// CHECK: %[[SAME_BYTES:.*]] = arith.constant 64 : index
// CHECK: memref.reinterpret_cast {{.*}} to offset: [64], sizes: [64], strides: [1]
// CHECK: memref.reinterpret_cast {{.*}} to offset: [192], sizes: [64], strides: [1]
// CHECK: %[[DISJOINT:.*]] = bosc_ame.dma.start {{.*}}, %[[SAME_BYTES]]
// CHECK-SAME: (memref<64xi8, {{.*}}#bosc_ame.spm<0>>, memref<64xi8, {{.*}}#bosc_ame.spm<0>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[DISJOINT]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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
// CHECK-LABEL: func.func @shifted_typed_view
// CHECK: %[[SHIFT_BYTES:.*]] = arith.constant 64 : index
// CHECK: %[[SOURCE_VIEW:.*]] = memref.view {{.*}}[%[[SHIFT_BYTES]]][] : memref<128xi8, #bosc_ame.spm<0>> to memref<16xi32, #bosc_ame.spm<0>>
// CHECK: %[[TARGET_VIEW:.*]] = memref.view {{.*}}[%[[SHIFT_BYTES]]][] : memref<128xi8, #bosc_ame.spm<1>> to memref<16xi32, #bosc_ame.spm<1>>
// CHECK: memref.reinterpret_cast %[[SOURCE_VIEW]] to offset: [0], sizes: [16], strides: [1]
// CHECK: memref.reinterpret_cast %[[TARGET_VIEW]] to offset: [0], sizes: [16], strides: [1]
// CHECK: %[[SHIFTED:.*]] = bosc_ame.dma.start {{.*}}, %[[SHIFT_BYTES]]
// CHECK-NEXT: bosc_ame.dma.wait %[[SHIFTED]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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
// CHECK-LABEL: func.func @masked_valid_subview
// CHECK: %[[VALID_BYTES:.*]] = arith.constant 3072 : index
// CHECK: %[[VALID:.*]] = bosc_ame.dma.start {{.*}}, %[[VALID_BYTES]]
// CHECK-SAME: (memref<768xf32, {{.*}}#bosc_ame.spm<0>>, memref<768xf32, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[VALID]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
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
