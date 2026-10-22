// RUN: buddy-opt %s --linalg-to-boscame='plan-report=true dma-abi=args_aligned64_nonalias' -o /dev/null 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan

module {
  func.func @reinterpret_disjoint(%root: memref<256xf32>) {
    %left = memref.reinterpret_cast %root to offset: [0], sizes: [16],
      strides: [1] : memref<256xf32> to memref<16xf32, strided<[1], offset: 0>>
    %right = memref.reinterpret_cast %root to offset: [32], sizes: [16],
      strides: [1] : memref<256xf32> to memref<16xf32, strided<[1], offset: 32>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0] : memref<16xf32, strided<[1], offset: 0>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: 32>>
    return
  }

  func.func @reinterpret_overlap(%root: memref<256xf32>) {
    %left = memref.reinterpret_cast %root to offset: [8], sizes: [16],
      strides: [1] : memref<256xf32> to memref<16xf32, strided<[1], offset: 8>>
    %right = memref.reinterpret_cast %root to offset: [16], sizes: [16],
      strides: [1] : memref<256xf32> to memref<16xf32, strided<[1], offset: 16>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: 8>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: 16>>
    return
  }

  func.func @reinterpret_dynamic(%root: memref<256xf32>, %offset: index) {
    %view = memref.reinterpret_cast %root to offset: [%offset], sizes: [16],
      strides: [1] : memref<256xf32> to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %baseValue = memref.load %root[%c0] : memref<256xf32>
    %value = memref.load %view[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_dynamic_base_disjoint(
      %root: memref<256xf32>, %offset: index) {
    %base = memref.reinterpret_cast %root to offset: [%offset], sizes: [64],
      strides: [1] : memref<256xf32>
        to memref<64xf32, strided<[1], offset: ?>>
    %left = memref.subview %base[0] [16] [1]
      : memref<64xf32, strided<[1], offset: ?>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %base[32] [16] [1]
      : memref<64xf32, strided<[1], offset: ?>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_distinct_dynamic_bases(
      %root: memref<256xf32>, %leftOffset: index, %rightOffset: index) {
    %leftBase = memref.reinterpret_cast %root to offset: [%leftOffset],
      sizes: [64], strides: [1] : memref<256xf32>
        to memref<64xf32, strided<[1], offset: ?>>
    %rightBase = memref.reinterpret_cast %root to offset: [%rightOffset],
      sizes: [64], strides: [1] : memref<256xf32>
        to memref<64xf32, strided<[1], offset: ?>>
    %left = memref.subview %leftBase[0] [16] [1]
      : memref<64xf32, strided<[1], offset: ?>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %rightBase[32] [16] [1]
      : memref<64xf32, strided<[1], offset: ?>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_dynamic_disjoint(%root: memref<512xf32>, %offset: index) {
    %leftBase = memref.reinterpret_cast %root to offset: [0], sizes: [64],
      strides: [2] : memref<512xf32> to memref<64xf32, strided<[2], offset: 0>>
    %rightBase = memref.reinterpret_cast %root to offset: [128], sizes: [64],
      strides: [2] : memref<512xf32> to memref<64xf32, strided<[2], offset: 128>>
    %left = memref.subview %leftBase[%offset] [8] [1]
      : memref<64xf32, strided<[2], offset: 0>>
        to memref<8xf32, strided<[2], offset: ?>>
    %right = memref.subview %rightBase[%offset] [8] [1]
      : memref<64xf32, strided<[2], offset: 128>>
        to memref<8xf32, strided<[2], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<8xf32, strided<[2], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<8xf32, strided<[2], offset: ?>>
    return
  }

  func.func @reinterpret_dynamic_overlap(%root: memref<256xf32>, %offset: index) {
    %leftBase = memref.reinterpret_cast %root to offset: [0], sizes: [64],
      strides: [2] : memref<256xf32> to memref<64xf32, strided<[2], offset: 0>>
    %rightBase = memref.reinterpret_cast %root to offset: [8], sizes: [64],
      strides: [2] : memref<256xf32> to memref<64xf32, strided<[2], offset: 8>>
    %left = memref.subview %leftBase[%offset] [8] [1]
      : memref<64xf32, strided<[2], offset: 0>>
        to memref<8xf32, strided<[2], offset: ?>>
    %right = memref.subview %rightBase[%offset] [8] [1]
      : memref<64xf32, strided<[2], offset: 8>>
        to memref<8xf32, strided<[2], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<8xf32, strided<[2], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<8xf32, strided<[2], offset: ?>>
    return
  }

  func.func @reinterpret_rank2_dynamic_disjoint(
      %root: memref<1024xf32>, %row: index) {
    %base = memref.reinterpret_cast %root to offset: [16], sizes: [16, 32],
      strides: [64, 1]
      : memref<1024xf32> to memref<16x32xf32, strided<[64, 1], offset: 16>>
    %left = memref.subview %base[%row, 0] [1, 8] [1, 1]
      : memref<16x32xf32, strided<[64, 1], offset: 16>>
        to memref<1x8xf32, strided<[64, 1], offset: ?>>
    %right = memref.subview %base[%row, 16] [1, 8] [1, 1]
      : memref<16x32xf32, strided<[64, 1], offset: 16>>
        to memref<1x8xf32, strided<[64, 1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0, %c0]
      : memref<1x8xf32, strided<[64, 1], offset: ?>>
    %rightValue = memref.load %right[%c0, %c0]
      : memref<1x8xf32, strided<[64, 1], offset: ?>>
    return
  }

  // The second reinterpret replaces the first descriptor offset. Accumulating
  // both offsets would incorrectly classify the two subviews as disjoint.
  func.func @reinterpret_resets_source_offset(
      %root: memref<1024xf32>, %offset: index) {
    %first = memref.reinterpret_cast %root to offset: [256], sizes: [64],
      strides: [1] : memref<1024xf32> to memref<64xf32, strided<[1], offset: 256>>
    %reset = memref.reinterpret_cast %first to offset: [0], sizes: [64],
      strides: [1] : memref<64xf32, strided<[1], offset: 256>>
        to memref<64xf32, strided<[1], offset: 0>>
    %peer = memref.reinterpret_cast %root to offset: [8], sizes: [64],
      strides: [1] : memref<1024xf32> to memref<64xf32, strided<[1], offset: 8>>
    %left = memref.subview %reset[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 0>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %peer[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 8>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  // A byte-shifting view changes the aligned pointer, so this slice must not
  // compare only the reinterpret descriptor offsets.
  func.func @reinterpret_shifted_pointer_unknown(
      %root: memref<1024xi8>, %shift: index, %offset: index) {
    %c0 = arith.constant 0 : index
    %shifted = memref.view %root[%shift][]
      : memref<1024xi8> to memref<128xf32>
    %unshifted = memref.view %root[%c0][]
      : memref<1024xi8> to memref<128xf32>
    %leftBase = memref.reinterpret_cast %shifted to offset: [0], sizes: [64],
      strides: [1] : memref<128xf32> to memref<64xf32, strided<[1], offset: 0>>
    %rightBase = memref.reinterpret_cast %unshifted to offset: [32], sizes: [64],
      strides: [1] : memref<128xf32> to memref<64xf32, strided<[1], offset: 32>>
    %left = memref.subview %leftBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 0>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %rightBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 32>>
        to memref<16xf32, strided<[1], offset: ?>>
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_offset_overflow(
      %root: memref<256xf32>, %offset: index) {
    %leftBase = memref.reinterpret_cast %root to
      offset: [2305843009213693952], sizes: [64], strides: [1]
      : memref<256xf32>
        to memref<64xf32, strided<[1], offset: 2305843009213693952>>
    %rightBase = memref.reinterpret_cast %root to offset: [0], sizes: [64],
      strides: [1] : memref<256xf32> to memref<64xf32, strided<[1], offset: 0>>
    %left = memref.subview %leftBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 2305843009213693952>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %rightBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 0>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  // Reinterpret offsets use the aligned-pointer frame. A known static root
  // descriptor offset is normalized into that frame for mixed comparisons.
  func.func @reinterpret_nonzero_root_offset(
      %root: memref<256xf32, strided<[1], offset: 64>>, %offset: index) {
    %base = memref.reinterpret_cast %root to offset: [64], sizes: [64],
      strides: [1] : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 64>>
    %left = memref.subview %base[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %root[%offset] [16] [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_nonzero_root_offset_disjoint(
      %root: memref<256xf32, strided<[1], offset: 64>>, %offset: index) {
    %leftBase = memref.reinterpret_cast %root to offset: [64], sizes: [64],
      strides: [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 64>>
    %rightBase = memref.reinterpret_cast %root to offset: [96], sizes: [64],
      strides: [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 96>>
    %left = memref.subview %leftBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %rightBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 96>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_nonzero_root_offset_overlap(
      %root: memref<256xf32, strided<[1], offset: 64>>, %offset: index) {
    %leftBase = memref.reinterpret_cast %root to offset: [64], sizes: [64],
      strides: [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 64>>
    %rightBase = memref.reinterpret_cast %root to offset: [72], sizes: [64],
      strides: [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 72>>
    %left = memref.subview %leftBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %rightBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 72>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @mixed_root_reinterpret_disjoint(
      %root: memref<256xf32, strided<[1], offset: 64>>, %offset: index) {
    %left = memref.subview %root[%offset] [16] [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %rightBase = memref.reinterpret_cast %root to offset: [96], sizes: [64],
      strides: [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 96>>
    %right = memref.subview %rightBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 96>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @mixed_root_reinterpret_overlap(
      %root: memref<256xf32, strided<[1], offset: 64>>, %offset: index) {
    %left = memref.subview %root[%offset] [16] [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %rightBase = memref.reinterpret_cast %root to offset: [72], sizes: [64],
      strides: [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 72>>
    %right = memref.subview %rightBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 72>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @mixed_root_reinterpret_equal(
      %root: memref<256xf32, strided<[1], offset: 64>>, %offset: index) {
    %left = memref.subview %root[%offset] [16] [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %rightBase = memref.reinterpret_cast %root to offset: [64], sizes: [64],
      strides: [1]
      : memref<256xf32, strided<[1], offset: 64>>
        to memref<64xf32, strided<[1], offset: 64>>
    %right = memref.subview %rightBase[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_dynamic_root_offset(
      %root: memref<256xf32, strided<[1], offset: ?>>, %offset: index) {
    %base = memref.reinterpret_cast %root to offset: [64], sizes: [64],
      strides: [1] : memref<256xf32, strided<[1], offset: ?>>
        to memref<64xf32, strided<[1], offset: 64>>
    %left = memref.subview %base[%offset] [16] [1]
      : memref<64xf32, strided<[1], offset: 64>>
        to memref<16xf32, strided<[1], offset: ?>>
    %right = memref.subview %root[%offset] [16] [1]
      : memref<256xf32, strided<[1], offset: ?>>
        to memref<16xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    %rightValue = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @reinterpret_rank3(%root: memref<1024xf32>) {
    %view = memref.reinterpret_cast %root to offset: [0], sizes: [4, 4, 4],
      strides: [16, 4, 1]
      : memref<1024xf32> to memref<4x4x4xf32, strided<[16, 4, 1]>>
    %c0 = arith.constant 0 : index
    %baseValue = memref.load %root[%c0] : memref<1024xf32>
    %value = memref.load %view[%c0, %c0, %c0]
      : memref<4x4x4xf32, strided<[16, 4, 1]>>
    return
  }
}

// PLAN: buffer-alias-pairs: count={{[1-9][0-9]*}} scope=known-roots-within-function unproven-pairs=unknown status=complete
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_overlap accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_dynamic accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_dynamic_base_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_distinct_dynamic_bases accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_rank3 accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_dynamic_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_dynamic_overlap accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_rank2_dynamic_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_resets_source_offset accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_shifted_pointer_unknown accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_offset_overflow accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_nonzero_root_offset accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_nonzero_root_offset_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_nonzero_root_offset_overlap accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@mixed_root_reinterpret_disjoint accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=same-root-symbolic-byte-intervals
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@mixed_root_reinterpret_overlap accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@mixed_root_reinterpret_equal accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
// PLAN-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@reinterpret_dynamic_root_offset accesses=[{{[0-9]+}},{{[0-9]+}}] relation=may-alias proof=same-root
