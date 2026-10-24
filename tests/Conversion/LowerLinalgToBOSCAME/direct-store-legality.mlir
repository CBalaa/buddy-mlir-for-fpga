// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// A masked provenance event is ineligible for direct forwarding. The
// existing full-copy IR remains available for ordinary DMA staging.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @masked(%src: memref<16xf32>, %dst: memref<16xf32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 16>,
                         valid_shape = array<i64: 15>, masked = true}
    } : memref<16xf32> to memref<16xf32>
    return
  }
}
