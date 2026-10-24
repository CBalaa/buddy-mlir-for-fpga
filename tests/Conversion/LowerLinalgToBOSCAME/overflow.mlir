// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// A small packed SPM allocation does not make an overflowing physical DDR
// range legal. Shape products, element widths, pitches and descriptor offsets
// must all be checked before creating packets or performing byte arithmetic.

//--- shape.mlir
module {
  func.func @shape(%out: memref<9223372036854775807x2xi8>) {
    %zero = arith.constant 0 : i8
    linalg.fill ins(%zero : i8) outs(%out : memref<9223372036854775807x2xi8>)
    return
  }
}

//--- bytes.mlir
module {
  func.func @bytes(%out: memref<2305843009213693952xf32>) {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%out : memref<2305843009213693952xf32>)
    return
  }
}

//--- span.mlir
module {
  func.func @span(%out: memref<3x64xi8, strided<[4611686018427387904, 1]>>) {
    %zero = arith.constant 0 : i8
    linalg.fill ins(%zero : i8)
      outs(%out : memref<3x64xi8, strided<[4611686018427387904, 1]>>)
    return
  }
}

//--- span-bytes.mlir
module {
  func.func @span_bytes(%out: memref<2x16xf32, strided<[2305843009213693952, 1]>>) {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32)
      outs(%out : memref<2x16xf32, strided<[2305843009213693952, 1]>>)
    return
  }
}

//--- offset-span.mlir
module {
  func.func @offset_span(%out: memref<64xi8, strided<[1], offset: 9223372036854775744>>) {
    %zero = arith.constant 0 : i8
    linalg.fill ins(%zero : i8)
      outs(%out : memref<64xi8, strided<[1], offset: 9223372036854775744>>)
    return
  }
}
