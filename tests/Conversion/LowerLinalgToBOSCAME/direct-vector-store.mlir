// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// A frontend-preserved store event stages the producer into SPM, then writes
// the result directly to its external DDR destination with vector stores.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @copy(%src: memref<64xf32>, %dst: memref<64xf32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64>,
                         valid_shape = array<i64: 64>, masked = false}
    } : memref<64xf32> to memref<64xf32>
    return
  }
}
