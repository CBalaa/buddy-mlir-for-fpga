// RUN: split-file %s %t
// RUN: not buddy-opt %t/shape.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not buddy-opt %t/bytes.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=BYTES
// RUN: not buddy-opt %t/span.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=SPAN
// RUN: not buddy-opt %t/span-bytes.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=SPAN-BYTES
// RUN: not buddy-opt %t/offset-span.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=SPAN

// A small packed SPM allocation does not make an overflowing physical DDR
// range legal. Shape products, element widths, pitches and descriptor offsets
// must all be checked before creating packets or performing byte arithmetic.
// SHAPE: BOSCAME transfer shape overflows or is empty
// BYTES: BOSCAME transfer byte count overflows
// SPAN: BOSCAME transfer physical span overflows
// SPAN-BYTES: BOSCAME transfer physical byte span overflows

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
