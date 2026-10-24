// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// A one-byte store cannot become a legal typed DMA packet by rounding its
// length up: doing so would write addresses outside the original operation.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @scalar(%a: memref<1xi8>, %b: memref<1xi8>, %c: memref<1xi8>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<1xi8>) outs(%c : memref<1xi8>) {
    ^bb0(%x: i8, %y: i8):
      linalg.yield %x : i8
    }
    return
  }
}
