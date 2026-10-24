// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// A vector result already lives in RVV registers.  A provenance-marked copy
// from its temporary must be fused by retargeting vector.transfer_write to the
// external DDR pointer; introducing an SPM view here would add a needless DMA
// round trip and lose the store destination provenance.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @copy(%src: memref<64xf32>, %dst: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %tmp = memref.alloc() alignment = 64 : memref<64xf32>
    %v = vector.load %src[%c0] : memref<64xf32>, vector<64xf32>
    vector.transfer_write %v, %tmp[%c0] {in_bounds = [true]}
      : vector<64xf32>, memref<64xf32>
    memref.copy %tmp, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64>,
                         valid_shape = array<i64: 64>, masked = false}
    } : memref<64xf32> to memref<64xf32>
    return
  }
}


// A complete two-dimensional transfer may target a pitched external buffer
// when the physical row stride is both non-overlapping and cache-line aligned.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @pitched_copy(%src: memref<2x4xf32>,
                          %dst: memref<2x4xf32, strided<[1024, 1]>>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %tmp = memref.alloc() alignment = 64 : memref<2x4xf32>
    %v = vector.transfer_read %src[%c0, %c0], %zero {in_bounds = [true, true]}
      : memref<2x4xf32>, vector<2x4xf32>
    vector.transfer_write %v, %tmp[%c0, %c0] {in_bounds = [true, true]}
      : vector<2x4xf32>, memref<2x4xf32>
    memref.copy %tmp, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 2, 4>,
                         valid_shape = array<i64: 2, 4>, masked = false}
    } : memref<2x4xf32> to memref<2x4xf32, strided<[1024, 1]>>
    return
  }
}


// A producer that reads the external destination cannot be retargeted across
// the copy: doing so would erase the original read/write ordering.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @destination_read(%dst: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %tmp = memref.alloc() alignment = 64 : memref<64xf32>
    %v = vector.load %dst[%c0] : memref<64xf32>, vector<64xf32>
    vector.transfer_write %v, %tmp[%c0] {in_bounds = [true]}
      : vector<64xf32>, memref<64xf32>
    memref.copy %tmp, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64>,
                         valid_shape = array<i64: 64>, masked = false}
    } : memref<64xf32> to memref<64xf32>
    return
  }
}
