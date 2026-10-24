// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// Fill and unused generic output arguments must not cause a DDR read. An
// accumulator does require its initial value. Scalar broadcasts stay in DDR
// and are loaded before the SPM vector computation begins.
module {
  func.func @fill(%out: memref<16xf32>, %value: f32) {
    linalg.fill ins(%value : f32) outs(%out : memref<16xf32>)
    return
  }

  func.func @copy(%input: memref<16xf32>, %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }

  func.func @accumulate(%input: memref<16xf32>, %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %initial: f32):
      %sum = arith.addf %x, %initial : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func @scalar_broadcast(%input: memref<16xf32>, %scale: memref<f32>,
                              %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> ()>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input, %scale : memref<16xf32>, memref<f32>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %scalar: f32, %unused: f32):
      %product = arith.mulf %x, %scalar : f32
      linalg.yield %product : f32
    }
    return
  }
}
