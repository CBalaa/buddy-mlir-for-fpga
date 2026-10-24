// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// Multiplying an f32 element offset by 16 proves 64-byte alignment. The
// emitted descriptor must preserve that arithmetic proof through DMA checking.
// An arbitrary runtime offset must fail instead of assuming the ABI covers it.

//--- aligned.mlir
module {
  func.func @aligned(%input: memref<4096xf32>, %row: index,
                     %out: memref<16xf32>) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %view = memref.subview %input[%offset] [16] [1]
      : memref<4096xf32> to memref<16xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%view : memref<16xf32, strided<[1], offset: ?>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- unknown.mlir
module {
  func.func @unknown(%input: memref<4096xf32>, %offset: index,
                     %out: memref<16xf32>) {
    %view = memref.subview %input[%offset] [16] [1]
      : memref<4096xf32> to memref<16xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%view : memref<16xf32, strided<[1], offset: ?>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}
