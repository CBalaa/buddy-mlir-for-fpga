// RUN: split-file %s %t
// RUN: buddy-opt %t/aligned-mask.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga ' | FileCheck %s --check-prefix=ALIGNED
// RUN: buddy-opt %t/reversed-clamp.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga ' | FileCheck %s --check-prefix=ALIGNED
// RUN: buddy-opt %t/selected-clamp.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga ' | FileCheck %s --check-prefix=ALIGNED
// RUN: not buddy-opt %t/unaligned-mask.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=UNALIGNED
// RUN: not buddy-opt %t/wrong-upper-bound.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=BAD-BOUND
// RUN: not buddy-opt %t/selected-wrong-upper-bound.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=SELECT-BAD-BOUND

// ALIGNED: bosc_ame.dma.start
// UNALIGNED: dynamic BOSCAME prefix extent must be clamped to the static maximum and provably divisible by a DMA packet
// BAD-BOUND: dynamic BOSCAME prefix extent must be clamped to the static maximum and provably divisible by a DMA packet
// SELECT-BAD-BOUND: dynamic BOSCAME prefix extent must be clamped to the static maximum and provably divisible by a DMA packet

//--- aligned-mask.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @aligned_mask(%source: memref<32xf32>, %limit: i32) {
    %low_bits_cleared = arith.constant -16 : i32
    %maximum = arith.constant 32 : index
    %zero = arith.constant 0 : index
    %aligned = arith.andi %limit, %low_bits_cleared : i32
    %as_index = arith.index_cast %aligned : i32 to index
    %upper = arith.minsi %as_index, %maximum : index
    %extent = arith.maxsi %upper, %zero : index
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %local = memref.alloc() alignment = 64 : memref<32xf32>
    %target_view = memref.subview %local[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %target_view {bosc_ame.memory = {
      kind = "load", logical_shape = array<i64: 32>, masked = true,
      valid_shape = array<i64: -9223372036854775808>}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}

//--- reversed-clamp.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @reversed_clamp(%source: memref<32xf32>, %limit: i32) {
    %low_bits_cleared = arith.constant -16 : i32
    %maximum = arith.constant 32 : index
    %zero = arith.constant 0 : index
    %aligned = arith.andi %limit, %low_bits_cleared : i32
    %as_index = arith.index_cast %aligned : i32 to index
    %lower = arith.maxsi %as_index, %zero : index
    %extent = arith.minsi %lower, %maximum : index
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %local = memref.alloc() alignment = 64 : memref<32xf32>
    %target_view = memref.subview %local[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %target_view {bosc_ame.memory = {
      kind = "load", logical_shape = array<i64: 32>, masked = true,
      valid_shape = array<i64: -9223372036854775808>}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}

//--- wrong-upper-bound.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @wrong_upper_bound(%source: memref<32xf32>, %limit: i32) {
    %low_bits_cleared = arith.constant -16 : i32
    %wrong_maximum = arith.constant 48 : index
    %zero = arith.constant 0 : index
    %aligned = arith.andi %limit, %low_bits_cleared : i32
    %as_index = arith.index_cast %aligned : i32 to index
    %lower = arith.maxsi %as_index, %zero : index
    %extent = arith.minsi %lower, %wrong_maximum : index
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %local = memref.alloc() alignment = 64 : memref<32xf32>
    %target_view = memref.subview %local[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %target_view {bosc_ame.memory = {
      kind = "load", logical_shape = array<i64: 32>, masked = true,
      valid_shape = array<i64: -9223372036854775808>}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}

//--- selected-clamp.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @selected_clamp(%source: memref<32xf32>, %limit: i32) {
    %low_bits_cleared = arith.constant -16 : i32
    %zero_i32 = arith.constant 0 : i32
    %maximum = arith.constant 32 : index
    %zero = arith.constant 0 : index
    %aligned = arith.andi %limit, %low_bits_cleared : i32
    %as_index = arith.index_cast %aligned : i32 to index
    %lower = arith.maxsi %as_index, %zero : index
    %clamped = arith.minsi %lower, %maximum : index
    %predicate = arith.cmpi sgt, %limit, %zero_i32 : i32
    %extent = arith.select %predicate, %clamped, %zero : index
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %local = memref.alloc() alignment = 64 : memref<32xf32>
    %target_view = memref.subview %local[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %target_view {bosc_ame.memory = {
      kind = "load", logical_shape = array<i64: 32>, masked = true,
      valid_shape = array<i64: -9223372036854775808>}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}

//--- selected-wrong-upper-bound.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @selected_wrong_upper_bound(%source: memref<32xf32>, %limit: i32) {
    %low_bits_cleared = arith.constant -16 : i32
    %zero_i32 = arith.constant 0 : i32
    %wrong_maximum = arith.constant 48 : index
    %zero = arith.constant 0 : index
    %aligned = arith.andi %limit, %low_bits_cleared : i32
    %as_index = arith.index_cast %aligned : i32 to index
    %lower = arith.maxsi %as_index, %zero : index
    %wrong_clamped = arith.minsi %lower, %wrong_maximum : index
    %predicate = arith.cmpi sgt, %limit, %zero_i32 : i32
    %extent = arith.select %predicate, %wrong_clamped, %zero : index
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %local = memref.alloc() alignment = 64 : memref<32xf32>
    %target_view = memref.subview %local[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %target_view {bosc_ame.memory = {
      kind = "load", logical_shape = array<i64: 32>, masked = true,
      valid_shape = array<i64: -9223372036854775808>}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}

//--- unaligned-mask.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @unaligned_mask(%source: memref<32xf32>, %limit: i32) {
    %partial_mask = arith.constant -8 : i32
    %maximum = arith.constant 32 : index
    %zero = arith.constant 0 : index
    %partially_aligned = arith.andi %limit, %partial_mask : i32
    %as_index = arith.index_cast %partially_aligned : i32 to index
    %upper = arith.minsi %as_index, %maximum : index
    %extent = arith.maxsi %upper, %zero : index
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %local = memref.alloc() alignment = 64 : memref<32xf32>
    %target_view = memref.subview %local[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %target_view {bosc_ame.memory = {
      kind = "load", logical_shape = array<i64: 32>, masked = true,
      valid_shape = array<i64: -9223372036854775808>}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}
