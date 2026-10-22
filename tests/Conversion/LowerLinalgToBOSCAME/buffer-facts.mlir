// RUN: buddy-opt %s --linalg-to-boscame='plan-report=true dma-abi=args_aligned64_nonalias' -o %t.ir 2>%t.plan
// RUN: buddy-opt %s --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o %t.default
// RUN: cmp %t.default %t.ir
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan

module {
  func.func private @opaque()

  func.func @shared_root_views(%root: memref<32xf32>, %value: f32) {
    %c0 = arith.constant 0 : index
    %left = memref.subview %root[0] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1]>>
    %right = memref.subview %root[16] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: 16>>
    %leftValue = memref.load %left[%c0] : memref<16xf32, strided<[1]>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: 16>>
    memref.store %value, %left[%c0] : memref<16xf32, strided<[1]>>
    func.call @opaque() : () -> ()
    return
  }

  func.func @memory_space_cast_disjoint_views(
      %root: memref<32xf32, strided<[1], offset: ?>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %cast = memref.memory_space_cast %root
      : memref<32xf32, strided<[1], offset: ?>>
        to memref<32xf32, strided<[1], offset: ?>, 1>
    %left = memref.subview %root[%c0] [16] [1]
      : memref<32xf32, strided<[1], offset: ?>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %cast[%c16] [16] [1]
      : memref<32xf32, strided<[1], offset: ?>, 1>
        to memref<16xf32, strided<[1], offset: ?>, 1>
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>, 1>
    return
  }

  func.func @memory_space_cast_overlapping_views(
      %root: memref<32xf32, strided<[1], offset: ?>>) {
    %c0 = arith.constant 0 : index
    %cast = memref.memory_space_cast %root
      : memref<32xf32, strided<[1], offset: ?>>
        to memref<32xf32, strided<[1], offset: ?>, 1>
    %left = memref.subview %root[%c0] [16] [1]
      : memref<32xf32, strided<[1], offset: ?>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %cast[%c0] [16] [1]
      : memref<32xf32, strided<[1], offset: ?>, 1>
        to memref<16xf32, strided<[1], offset: ?>, 1>
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>, 1>
    return
  }

  // Both select arms share the root but describe different views. The
  // selected descriptor is therefore an unknown interval and must not be
  // proven disjoint from the right-hand view.
  func.func @select_different_views(%root: memref<32xf32>, %condition: i1) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %left = memref.subview %root[%c0] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %root[%c16] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %selected = arith.select %condition, %left, %right
      : memref<16xf32, strided<[1], offset: ?>>
    %selectedValue = memref.load %selected[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @direct_ddr_copy(%source: memref<16xf32>, %destination: memref<16xf32>) {
    memref.copy %source, %destination {
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 16>, masked = false,
                         origin_id = 77 : i64, valid_shape = array<i64: 16>}
    } : memref<16xf32> to memref<16xf32>
    return
  }

  func.func @dynamic_view(%root: memref<32xf32>, %offset: index) {
    %c0 = arith.constant 0 : index
    %view = memref.subview %root[%offset] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %baseValue = memref.load %root[%c0] : memref<32xf32>
    %value = memref.load %view[%c0] : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @dynamic_sibling_views(%root: memref<128xf32>, %offset: index) {
    %base = memref.subview %root[%offset] [64] [1]
      : memref<128xf32> to memref<64xf32, strided<[1], offset: ?>>
    %left = memref.subview %base[0] [32] [1]
      : memref<64xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    %right = memref.subview %base[32] [32] [1]
      : memref<64xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<32xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<32xf32, strided<[1], offset: ?>>
    return
  }

  func.func @strided_disjoint_siblings(%root: memref<256xf32>, %offset: index) {
    %base = memref.subview %root[%offset] [64] [2]
      : memref<256xf32> to memref<64xf32, strided<[2], offset: ?>>
    %left = memref.subview %base[0] [8] [2]
      : memref<64xf32, strided<[2], offset: ?>>
        to memref<8xf32, strided<[4], offset: ?>>
    %right = memref.subview %base[16] [8] [2]
      : memref<64xf32, strided<[2], offset: ?>>
        to memref<8xf32, strided<[4], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<8xf32, strided<[4], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<8xf32, strided<[4], offset: ?>>
    return
  }

  func.func @strided_overlapping_siblings(%root: memref<256xf32>, %offset: index) {
    %base = memref.subview %root[%offset] [64] [2]
      : memref<256xf32> to memref<64xf32, strided<[2], offset: ?>>
    %left = memref.subview %base[0] [8] [2]
      : memref<64xf32, strided<[2], offset: ?>>
        to memref<8xf32, strided<[4], offset: ?>>
    %right = memref.subview %base[12] [8] [2]
      : memref<64xf32, strided<[2], offset: ?>>
        to memref<8xf32, strided<[4], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<8xf32, strided<[4], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<8xf32, strided<[4], offset: ?>>
    return
  }

  func.func @strided_dynamic_coefficients(%root: memref<1024xf32>, %offset: index) {
    %evenBase = memref.subview %root[0] [512] [2]
      : memref<1024xf32> to memref<512xf32, strided<[2], offset: 0>>
    %thirdBase = memref.subview %root[0] [342] [3]
      : memref<1024xf32> to memref<342xf32, strided<[3], offset: 0>>
    %even = memref.subview %evenBase[%offset] [8] [1]
      : memref<512xf32, strided<[2], offset: 0>>
        to memref<8xf32, strided<[2], offset: ?>>
    %third = memref.subview %thirdBase[%offset] [8] [1]
      : memref<342xf32, strided<[3], offset: 0>>
        to memref<8xf32, strided<[3], offset: ?>>
    %c0 = arith.constant 0 : index
    %evenValue = memref.load %even[%c0]
      : memref<8xf32, strided<[2], offset: ?>>
    %thirdValue = memref.load %third[%c0]
      : memref<8xf32, strided<[3], offset: ?>>
    return
  }

  func.func @strided_span_overflow(
      %root: memref<2xf32, strided<[4611686018427387904]>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %first = memref.load %root[%c0]
      : memref<2xf32, strided<[4611686018427387904]>>
    %second = memref.load %root[%c1]
      : memref<2xf32, strided<[4611686018427387904]>>
    return
  }

  func.func @root_descriptor_offset_overflow(
      %root: memref<64xf32, strided<[1], offset: 4611686018427387904>>) {
    %left = memref.subview %root[0] [16] [1]
      : memref<64xf32, strided<[1], offset: 4611686018427387904>>
        to memref<16xf32, strided<[1], offset: 4611686018427387904>>
    %right = memref.subview %root[32] [16] [1]
      : memref<64xf32, strided<[1], offset: 4611686018427387904>>
        to memref<16xf32, strided<[1], offset: 4611686018427387936>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: 4611686018427387904>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: 4611686018427387936>>
    return
  }

  func.func @rank2_strided_disjoint_siblings(
      %root: memref<64x64xf32>, %row: index) {
    %base = memref.subview %root[%row, 0] [16, 32] [2, 1]
      : memref<64x64xf32> to memref<16x32xf32, strided<[128, 1], offset: ?>>
    %left = memref.subview %base[0, 0] [4, 16] [1, 1]
      : memref<16x32xf32, strided<[128, 1], offset: ?>>
        to memref<4x16xf32, strided<[128, 1], offset: ?>>
    %right = memref.subview %base[8, 0] [4, 16] [1, 1]
      : memref<16x32xf32, strided<[128, 1], offset: ?>>
        to memref<4x16xf32, strided<[128, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0]
      : memref<4x16xf32, strided<[128, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0]
      : memref<4x16xf32, strided<[128, 1], offset: ?>>
    return
  }

  func.func @rank2_strided_overlapping_siblings(
      %root: memref<64x64xf32>, %row: index) {
    %base = memref.subview %root[%row, 0] [16, 32] [2, 1]
      : memref<64x64xf32> to memref<16x32xf32, strided<[128, 1], offset: ?>>
    %left = memref.subview %base[0, 0] [4, 16] [1, 1]
      : memref<16x32xf32, strided<[128, 1], offset: ?>>
        to memref<4x16xf32, strided<[128, 1], offset: ?>>
    %right = memref.subview %base[2, 0] [4, 16] [1, 1]
      : memref<16x32xf32, strided<[128, 1], offset: ?>>
        to memref<4x16xf32, strided<[128, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0]
      : memref<4x16xf32, strided<[128, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0]
      : memref<4x16xf32, strided<[128, 1], offset: ?>>
    return
  }

  func.func @rank3_strided_disjoint_siblings(
      %root: memref<8x64x64xf32>, %row: index) {
    %left = memref.subview %root[%row, 0, 0] [1, 16, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    %right = memref.subview %root[%row, 32, 0] [1, 16, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0, %c0]
      : memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0, %c0]
      : memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    return
  }

  func.func @rank3_strided_overlapping_siblings(
      %root: memref<8x64x64xf32>, %row: index) {
    %left = memref.subview %root[%row, 0, 0] [1, 16, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    %right = memref.subview %root[%row, 8, 0] [1, 16, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0, %c0]
      : memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0, %c0]
      : memref<1x16x16xf32, strided<[4096, 64, 1], offset: ?>>
    return
  }

  func.func @rank4_strided_disjoint_siblings(
      %root: memref<4x8x16x64xf32>, %batch: index, %row: index) {
    %left = memref.subview %root[%batch, 0, %row, 0] [1, 1, 1, 16]
      [1, 1, 1, 1]
      : memref<4x8x16x64xf32>
        to memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    %right = memref.subview %root[%batch, 4, %row, 0] [1, 1, 1, 16]
      [1, 1, 1, 1]
      : memref<4x8x16x64xf32>
        to memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0, %c0, %c0]
      : memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0, %c0, %c0]
      : memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    return
  }

  func.func @rank4_strided_overlapping_siblings(
      %root: memref<4x8x16x64xf32>, %batch: index, %row: index) {
    %left = memref.subview %root[%batch, 0, %row, 0] [1, 1, 1, 16]
      [1, 1, 1, 1]
      : memref<4x8x16x64xf32>
        to memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    %right = memref.subview %root[%batch, 0, %row, 8] [1, 1, 1, 16]
      [1, 1, 1, 1]
      : memref<4x8x16x64xf32>
        to memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0, %c0, %c0]
      : memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0, %c0, %c0]
      : memref<1x1x1x16xf32, strided<[8192, 1024, 64, 1], offset: ?>>
    return
  }

  func.func @rank3_multiple_dynamic_offsets_disjoint(
      %root: memref<8x64x64xf32>, %row: index, %column: index) {
    %left = memref.subview %root[%row, %column, 0] [1, 1, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    %right = memref.subview %root[%row, %column, 32] [1, 1, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0, %c0]
      : memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0, %c0]
      : memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    return
  }

  func.func @rank3_distinct_dynamic_offsets(
      %root: memref<8x64x64xf32>, %row: index, %column: index,
      %otherColumn: index) {
    %left = memref.subview %root[%row, %column, 0] [1, 1, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    %right = memref.subview %root[%row, %otherColumn, 32] [1, 1, 16] [1, 1, 1]
      : memref<8x64x64xf32>
        to memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0, %c0]
      : memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0, %c0]
      : memref<1x1x16xf32, strided<[4096, 64, 1], offset: ?>>
    return
  }

  func.func @rank2_strided_columns(
      %root: memref<64x128xf32>, %row: index) {
    %base = memref.subview %root[%row, 0] [16, 48] [2, 2]
      : memref<64x128xf32> to memref<16x48xf32, strided<[256, 2], offset: ?>>
    // Column offsets are scaled by the base's nonunit column stride.
    %left = memref.subview %base[0, 1] [1, 4] [1, 2]
      : memref<16x48xf32, strided<[256, 2], offset: ?>>
        to memref<1x4xf32, strided<[256, 4], offset: ?>>
    %right = memref.subview %base[0, 9] [1, 4] [1, 2]
      : memref<16x48xf32, strided<[256, 2], offset: ?>>
        to memref<1x4xf32, strided<[256, 4], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0]
      : memref<1x4xf32, strided<[256, 4], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0]
      : memref<1x4xf32, strided<[256, 4], offset: ?>>
    return
  }

  func.func @rank2_strided_dynamic_coefficients(
      %root: memref<64x64xf32>, %offset: index) {
    %row = memref.subview %root[%offset, 0] [8, 8] [2, 1]
      : memref<64x64xf32> to memref<8x8xf32, strided<[128, 1], offset: ?>>
    // The static bounds are separated, but %offset has different physical
    // coefficients in the two views and therefore cannot cancel.
    %column = memref.subview %root[32, %offset] [8, 8] [1, 2]
      : memref<64x64xf32> to memref<8x8xf32, strided<[64, 2], offset: ?>>
    %c0 = arith.constant 0 : index
    %rowValue = memref.load %row[%c0, %c0]
      : memref<8x8xf32, strided<[128, 1], offset: ?>>
    %columnValue = memref.load %column[%c0, %c0]
      : memref<8x8xf32, strided<[64, 2], offset: ?>>
    return
  }

  // Exact same-value forwarding keeps the view descriptor available for
  // interval analysis through the select.
  func.func @select_same_view(%root: memref<32xf32>, %condition: i1) {
    %c0 = arith.constant 0 : index
    %left = memref.subview %root[0] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1]>>
    %right = memref.subview %root[16] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: 16>>
    %selected = arith.select %condition, %left, %left
      : memref<16xf32, strided<[1]>>
    %selectedValue = memref.load %selected[%c0]
      : memref<16xf32, strided<[1]>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: 16>>
    return
  }

  // memref.transpose is metadata-only: the two singleton views retain the
  // same storage root while their permuted strides still give disjoint
  // physical intervals.
  func.func @transpose_disjoint(%root: memref<8x8xf32>) {
    %transposed = memref.transpose %root (i, j) -> (j, i)
      : memref<8x8xf32> to
        memref<8x8xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %left = memref.subview %transposed[%c0, %c0] [1, 1] [1, 1]
      : memref<8x8xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
        to memref<1x1xf32, strided<[1, 8], offset: ?>>
    %right = memref.subview %transposed[%c4, %c4] [1, 1] [1, 1]
      : memref<8x8xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
        to memref<1x1xf32, strided<[1, 8], offset: ?>>
    %leftValue = memref.load %left[%c0, %c0]
      : memref<1x1xf32, strided<[1, 8], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0]
      : memref<1x1xf32, strided<[1, 8], offset: ?>>
    return
  }

  func.func @transpose_overlapping(%root: memref<8x8xf32>) {
    %transposed = memref.transpose %root (i, j) -> (j, i)
      : memref<8x8xf32> to
        memref<8x8xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
    %c0 = arith.constant 0 : index
    %left = memref.subview %transposed[%c0, %c0] [1, 1] [1, 1]
      : memref<8x8xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
        to memref<1x1xf32, strided<[1, 8], offset: ?>>
    %right = memref.subview %transposed[%c0, %c0] [1, 1] [1, 1]
      : memref<8x8xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
        to memref<1x1xf32, strided<[1, 8], offset: ?>>
    %leftValue = memref.load %left[%c0, %c0]
      : memref<1x1xf32, strided<[1, 8], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0]
      : memref<1x1xf32, strided<[1, 8], offset: ?>>
    return
  }

  func.func @reshape_disjoint_siblings(%root: memref<16xf32>,
                                       %shape: memref<2xi32>) {
    %c0 = arith.constant 0 : index
    %matrix = memref.reshape %root(%shape)
      : (memref<16xf32>, memref<2xi32>) -> memref<4x4xf32>
    %top = memref.subview %matrix[0, 0] [1, 4] [1, 1]
      : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: 0>>
    %bottom = memref.subview %matrix[2, 0] [1, 4] [1, 1]
      : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: 8>>
    %topValue = memref.load %top[%c0, %c0]
      : memref<1x4xf32, strided<[4, 1], offset: 0>>
    %bottomValue = memref.load %bottom[%c0, %c0]
      : memref<1x4xf32, strided<[4, 1], offset: 8>>
    return
  }

  func.func @expand_shape_disjoint(%root: memref<16xf32, strided<[1], offset: ?>>) {
    %matrix = memref.expand_shape %root [[0, 1]] output_shape [4, 4]
      : memref<16xf32, strided<[1], offset: ?>>
        into memref<4x4xf32, strided<[4, 1], offset: ?>>
    %top = memref.subview %matrix[0, 0] [1, 4] [1, 1]
      : memref<4x4xf32, strided<[4, 1], offset: ?>>
        to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %bottom = memref.subview %matrix[2, 0] [1, 4] [1, 1]
      : memref<4x4xf32, strided<[4, 1], offset: ?>>
        to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %topValue = memref.load %top[%c0, %c0]
      : memref<1x4xf32, strided<[4, 1], offset: ?>>
    %bottomValue = memref.load %bottom[%c0, %c0]
      : memref<1x4xf32, strided<[4, 1], offset: ?>>
    return
  }

  func.func @expand_shape_overlapping(%root: memref<16xf32, strided<[1], offset: ?>>) {
    %matrix = memref.expand_shape %root [[0, 1]] output_shape [4, 4]
      : memref<16xf32, strided<[1], offset: ?>>
        into memref<4x4xf32, strided<[4, 1], offset: ?>>
    %top = memref.subview %matrix[0, 0] [1, 4] [1, 1]
      : memref<4x4xf32, strided<[4, 1], offset: ?>>
        to memref<1x4xf32, strided<[4, 1], offset: ?>>
    %overlap = memref.subview %matrix[0, 2] [1, 2] [1, 1]
      : memref<4x4xf32, strided<[4, 1], offset: ?>>
        to memref<1x2xf32, strided<[4, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %topValue = memref.load %top[%c0, %c0]
      : memref<1x4xf32, strided<[4, 1], offset: ?>>
    %overlapValue = memref.load %overlap[%c0, %c0]
      : memref<1x2xf32, strided<[4, 1], offset: ?>>
    return
  }

  func.func @collapse_shape_disjoint(
      %root: memref<4x4xf32, strided<[4, 1], offset: ?>>) {
    %flat = memref.collapse_shape %root [[0, 1]]
      : memref<4x4xf32, strided<[4, 1], offset: ?>>
        into memref<16xf32, strided<[1], offset: ?>>
    %left = memref.subview %flat[0] [4] [1]
      : memref<16xf32, strided<[1], offset: ?>>
        to memref<4xf32, strided<[1], offset: ?>>
    %right = memref.subview %flat[8] [4] [1]
      : memref<16xf32, strided<[1], offset: ?>>
        to memref<4xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<4xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<4xf32, strided<[1], offset: ?>>
    return
  }

  func.func @collapse_shape_overlapping(
      %root: memref<4x4xf32, strided<[4, 1], offset: ?>>) {
    %flat = memref.collapse_shape %root [[0, 1]]
      : memref<4x4xf32, strided<[4, 1], offset: ?>>
        into memref<16xf32, strided<[1], offset: ?>>
    %left = memref.subview %flat[0] [4] [1]
      : memref<16xf32, strided<[1], offset: ?>>
        to memref<4xf32, strided<[1], offset: ?>>
    %overlap = memref.subview %flat[2] [4] [1]
      : memref<16xf32, strided<[1], offset: ?>>
        to memref<4xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<4xf32, strided<[1], offset: ?>>
    %overlapValue = memref.load %overlap[%c0]
      : memref<4xf32, strided<[1], offset: ?>>
    return
  }

  func.func @pitched_column_disjoint(%root: memref<8x64xf32>) {
    %left = memref.subview %root[0, 0] [8, 16] [1, 1]
      : memref<8x64xf32> to memref<8x16xf32, strided<[64, 1]>>
    %right = memref.subview %root[0, 32] [8, 16] [1, 1]
      : memref<8x64xf32> to
        memref<8x16xf32, strided<[64, 1], offset: 32>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0]
      : memref<8x16xf32, strided<[64, 1]>>
    %rightValue = memref.load %right[%c0, %c0]
      : memref<8x16xf32, strided<[64, 1], offset: 32>>
    return
  }

  func.func @pitched_column_overlap(%root: memref<8x64xf32>) {
    %left = memref.subview %root[0, 0] [8, 16] [1, 1]
      : memref<8x64xf32> to memref<8x16xf32, strided<[64, 1]>>
    %overlap = memref.subview %root[0, 8] [8, 16] [1, 1]
      : memref<8x64xf32> to
        memref<8x16xf32, strided<[64, 1], offset: 8>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0]
      : memref<8x16xf32, strided<[64, 1]>>
    %overlapValue = memref.load %overlap[%c0, %c0]
      : memref<8x16xf32, strided<[64, 1], offset: 8>>
    return
  }

}

// PLAN:   buffer-alias-pairs: count={{[1-9][0-9]*}} scope=known-roots-within-function unproven-pairs=unknown status=complete
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@shared_root_views accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@shared_root_views accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@memory_space_cast_disjoint_views accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@memory_space_cast_overlapping_views accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@select_different_views accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@direct_ddr_copy accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=abi-nonalias-entry-arguments
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@dynamic_view accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@dynamic_sibling_views accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@strided_disjoint_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@strided_overlapping_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@strided_dynamic_coefficients accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@strided_span_overflow accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@root_descriptor_offset_overflow accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank2_strided_disjoint_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank2_strided_overlapping_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank3_strided_disjoint_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank3_strided_overlapping_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank4_strided_disjoint_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank4_strided_overlapping_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank3_multiple_dynamic_offsets_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank3_distinct_dynamic_offsets accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank2_strided_columns accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@rank2_strided_dynamic_coefficients accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@select_same_view accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@transpose_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@transpose_overlapping accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reshape_disjoint_siblings accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@expand_shape_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@expand_shape_overlapping accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@collapse_shape_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@collapse_shape_overlapping accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@pitched_column_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-pitched-row-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@pitched_column_overlap accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN:   buffer-accesses: count=
// PLAN: buffer-access[{{[0-9]+}}]: op=memref.load effect=read
// PLAN-SAME: root=known root-id=[[ROOT:[0-9]+]]
// PLAN-SAME: views=[memref.subview] shape=[16] logical-shape=[16] strides=[1]
// PLAN: buffer-access[{{[0-9]+}}]: op=memref.load effect=read
// PLAN-SAME: root=known root-id=[[ROOT]]
// PLAN-SAME: views=[memref.subview] shape=[16] logical-shape=[16] strides=[1]
// PLAN: buffer-access[{{[0-9]+}}]: op=memref.store effect=write
// PLAN-SAME: root=known root-id=
// PLAN: buffer-access[{{[0-9]+}}]: op=func.call effect=unknown
// PLAN-SAME: type=<unknown> memory-space=<unknown> root=unknown
// PLAN: buffer-access[{{[0-9]+}}]: op=linalg.generic effect=read
// PLAN-SAME: root=known root-id=
// PLAN: buffer-access[{{[0-9]+}}]: op=linalg.generic effect=write
// PLAN-SAME: source-origin-id=77 type=memref<16xf32> memory-space=default root=known root-id={{[0-9]+}} views=[] shape=[16] logical-shape=[16]
// PLAN-SAME: transfer=<none> alias=unknown dma=unassessed
// PLAN: buffer-access[{{[0-9]+}}]: op=memref.load effect=read{{.*}}views=[memref.subview]{{.*}}effective-byte-offset-bytes=<unknown>
// PLAN: buffer-access[{{[0-9]+}}]: op=memref.load effect=read{{.*}}root=known root-id={{[0-9]+}} views=[memref.subview, memref.reshape] shape=[1,4]
