// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// The unrelated root occupies an entire bank. The external C has a proven
// direct route, so only A/B need staging and their two 64 KiB windows fit in
// the other bank. Both selected routes participate in the placement budget.
module {
  func.func @ame_staging_is_in_the_budget(
      %a: memref<64x1024xi8>,
      %b: memref<1024x64xi8, strided<[1, 1024]>>,
      %c: memref<64x64xi32>) {
    %zero = arith.constant 0.0 : f32
    %unrelated = memref.alloc() alignment = 64 : memref<32768xf32>
    linalg.fill ins(%zero : f32) outs(%unrelated : memref<32768xf32>)
    linalg.matmul {bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x1024xi8>,
        memref<1024x64xi8, strided<[1, 1024]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}
