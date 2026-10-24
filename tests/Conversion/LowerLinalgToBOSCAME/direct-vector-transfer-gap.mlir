// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// A pure operation between a vector transfer and its provenance-marked copy
// must not force the already-materialized vector through SPM/DMA staging.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @separated(%src: memref<64xf32>, %dst: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %tmp = memref.alloc() alignment = 64 : memref<64xf32>
    %v = vector.load %src[%c0] : memref<64xf32>, vector<64xf32>
    vector.transfer_write %v, %tmp[%c0] {in_bounds = [true]}
      : vector<64xf32>, memref<64xf32>
    %one = arith.constant 1 : index
    %unused = arith.addi %c0, %one : index
    memref.copy %tmp, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64>,
                         valid_shape = array<i64: 64>, masked = false}
    } : memref<64xf32> to memref<64xf32>
    return
  }
}
