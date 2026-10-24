// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// One 64x128 program of the M=128, N=128, K=1024 workload. B is already in
// physical [N,K] storage. Two reads stage B and A; the legal accumulator
// tiles read and write C directly. No DDR packing is needed.
module {
  func.func @gemm(%a: memref<64x1024xi8>,
                  %b: memref<1024x128xi8, strided<[1, 1024]>>,
                  %c: memref<64x128xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x1024xi8>,
                   memref<1024x128xi8, strided<[1, 1024]>>)
      outs(%c : memref<64x128xi32>)
    return
  }
}




// The output is accepted directly by lower-bosc-ame. In particular no
// runtime call may interrupt the live AME accumulator chain.
