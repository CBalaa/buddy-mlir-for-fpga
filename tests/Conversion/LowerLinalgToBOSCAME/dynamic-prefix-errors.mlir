// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

//--- strided-peer.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @strided_store(%destination: memref<32xf32>,
                           %source: memref<32xf32, strided<[2]>>,
                           %limit: i32) {
    %zero_i32 = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %sixteen = arith.constant 16 : i32
    %zero = arith.constant 0 : index
    %maximum = arith.constant 32 : index
    %units_hi = arith.minsi %limit, %two : i32
    %units = arith.maxsi %units_hi, %zero_i32 : i32
    %scaled = arith.muli %units, %sixteen : i32
    %as_index = arith.index_cast %scaled : i32 to index
    %upper = arith.minsi %as_index, %maximum : index
    %extent = arith.maxsi %upper, %zero : index
    %destination_view = memref.subview %destination[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32, strided<[2]>> to memref<?xf32, strided<[2]>>
    memref.copy %source_view, %destination_view {bosc_ame.memory = {
      destination = "external_ddr", kind = "store", masked = true}}
      : memref<?xf32, strided<[2]>> to memref<?xf32, strided<[1]>>
    return
  }
}

//--- helper-alias.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func private @overlap_prefix_helper(
      %destination: memref<32xf32, strided<[1], offset: 16>>,
      %source: memref<32xf32, strided<[1], offset: 0>>, %limit: i32) {
    %zero_i32 = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %sixteen = arith.constant 16 : i32
    %zero = arith.constant 0 : index
    %maximum = arith.constant 32 : index
    %units_hi = arith.minsi %limit, %two : i32
    %units = arith.maxsi %units_hi, %zero_i32 : i32
    %scaled = arith.muli %units, %sixteen : i32
    %as_index = arith.index_cast %scaled : i32 to index
    %upper = arith.minsi %as_index, %maximum : index
    %extent = arith.maxsi %upper, %zero : index
    %destination_view = memref.subview %destination[0] [%extent] [1]
      : memref<32xf32, strided<[1], offset: 16>>
      to memref<?xf32, strided<[1], offset: 16>>
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32, strided<[1], offset: 0>>
      to memref<?xf32, strided<[1], offset: 0>>
    memref.copy %source_view, %destination_view {bosc_ame.memory = {
      destination = "external_ddr", kind = "store", masked = true}}
      : memref<?xf32, strided<[1], offset: 0>>
      to memref<?xf32, strided<[1], offset: 16>>
    return
  }

  func.func @overlap_prefix_call(%root: memref<48xf32>, %limit: i32) {
    %source = memref.subview %root[0] [32] [1]
      : memref<48xf32> to memref<32xf32, strided<[1], offset: 0>>
    %destination = memref.subview %root[16] [32] [1]
      : memref<48xf32> to memref<32xf32, strided<[1], offset: 16>>
    func.call @overlap_prefix_helper(%destination, %source, %limit)
      : (memref<32xf32, strided<[1], offset: 16>>,
         memref<32xf32, strided<[1], offset: 0>>, i32) -> ()
    return
  }
}

//--- global-peer.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  memref.global "private" @peer : memref<32xf32> = uninitialized alignment = 64

  func.func @global_peer_store(%destination: memref<32xf32>, %limit: i32) {
    %zero_i32 = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %sixteen = arith.constant 16 : i32
    %zero = arith.constant 0 : index
    %maximum = arith.constant 32 : index
    %units_hi = arith.minsi %limit, %two : i32
    %units = arith.maxsi %units_hi, %zero_i32 : i32
    %scaled = arith.muli %units, %sixteen : i32
    %as_index = arith.index_cast %scaled : i32 to index
    %upper = arith.minsi %as_index, %maximum : index
    %extent = arith.maxsi %upper, %zero : index
    %source = memref.get_global @peer : memref<32xf32>
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    %destination_view = memref.subview %destination[0] [%extent] [1]
      : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %destination_view {bosc_ame.memory = {
      destination = "external_ddr", kind = "store", masked = true}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}

//--- oversized.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @oversized_prefix(%destination: memref<32784xf32>,
                              %source: memref<32784xf32>, %limit: i32) {
    %zero_i32 = arith.constant 0 : i32
    %max_units = arith.constant 2049 : i32
    %sixteen = arith.constant 16 : i32
    %zero = arith.constant 0 : index
    %maximum = arith.constant 32784 : index
    %units_hi = arith.minsi %limit, %max_units : i32
    %units = arith.maxsi %units_hi, %zero_i32 : i32
    %scaled = arith.muli %units, %sixteen : i32
    %as_index = arith.index_cast %scaled : i32 to index
    %upper = arith.minsi %as_index, %maximum : index
    %extent = arith.maxsi %upper, %zero : index
    %destination_view = memref.subview %destination[0] [%extent] [1]
      : memref<32784xf32> to memref<?xf32, strided<[1]>>
    %source_view = memref.subview %source[0] [%extent] [1]
      : memref<32784xf32> to memref<?xf32, strided<[1]>>
    memref.copy %source_view, %destination_view {bosc_ame.memory = {
      destination = "external_ddr", kind = "store", masked = true}}
      : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    return
  }
}
