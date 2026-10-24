// RUN: split-file %s %t
// RUN: not buddy-opt %t/static-overlap.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATIC-OVERLAP
// RUN: not buddy-opt %t/symbolic-disjoint.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=SYMBOLIC-DISJOINT
// RUN: not buddy-opt %t/symbolic-overlap.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s --check-prefix=SYMBOLIC-OVERLAP
// RUN: not buddy-opt %t/symbolic-dynamic-reinterpret-disjoint.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=SYMBOLIC-DYNAMIC-DISJOINT

// V1 keeps the original inputs and rejects both actual overlap and symbolic
// separation without a supported contract proof. Static disjoint storage
// and legal endpoint offsets are covered by PlanRoutes.py/PlanEmitValidation.py.
//--- static-overlap.mlir
module {
  func.func @overlapping_reinterpret_views(%root: memref<128xf32>) {
    %source = memref.reinterpret_cast %root to offset: [16], sizes: [32],
      strides: [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: 16>>
    %destination = memref.reinterpret_cast %root to offset: [32], sizes: [32],
      strides: [1] : memref<128xf32> to memref<32xf32, strided<[1], offset: 32>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%source : memref<32xf32, strided<[1], offset: 16>>)
      outs(%destination : memref<32xf32, strided<[1], offset: 32>>) {
    ^bb0(%input: f32, %unused: f32):
      linalg.yield %input : f32
    }
    return
  }
}

//--- symbolic-disjoint.mlir
module {
  // The common runtime offset prevents an absolute interval proof. Its
  // packet alignment is proven separately from the static view separation.
  func.func @symbolic_disjoint_reinterpret_views(%root: memref<256xf32>,
                                                %row: index) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %sourceBase = memref.reinterpret_cast %root to offset: [16], sizes: [64],
      strides: [1] : memref<256xf32> to memref<64xf32, strided<[1], offset: 16>>
    %destinationBase = memref.reinterpret_cast %root to offset: [48], sizes: [64],
      strides: [1] : memref<256xf32> to memref<64xf32, strided<[1], offset: 48>>
    %source = memref.subview %sourceBase[%offset] [32] [1]
      : memref<64xf32, strided<[1], offset: 16>>
        to memref<32xf32, strided<[1], offset: ?>>
    %destination = memref.subview %destinationBase[%offset] [32] [1]
      : memref<64xf32, strided<[1], offset: 48>>
        to memref<32xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%source : memref<32xf32, strided<[1], offset: ?>>)
      outs(%destination : memref<32xf32, strided<[1], offset: ?>>) {
    ^bb0(%input: f32, %unused: f32):
      linalg.yield %input : f32
    }
    return
  }
}

//--- symbolic-overlap.mlir
module {
  func.func @symbolic_overlapping_reinterpret_views(%root: memref<256xf32>,
                                                   %row: index) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %sourceBase = memref.reinterpret_cast %root to offset: [16], sizes: [64],
      strides: [1] : memref<256xf32> to memref<64xf32, strided<[1], offset: 16>>
    %destinationBase = memref.reinterpret_cast %root to offset: [32], sizes: [64],
      strides: [1] : memref<256xf32> to memref<64xf32, strided<[1], offset: 32>>
    %source = memref.subview %sourceBase[%offset] [32] [1]
      : memref<64xf32, strided<[1], offset: 16>>
        to memref<32xf32, strided<[1], offset: ?>>
    %destination = memref.subview %destinationBase[%offset] [32] [1]
      : memref<64xf32, strided<[1], offset: 32>>
        to memref<32xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%source : memref<32xf32, strided<[1], offset: ?>>)
      outs(%destination : memref<32xf32, strided<[1], offset: ?>>) {
    ^bb0(%input: f32, %unused: f32):
      linalg.yield %input : f32
    }
    return
  }
}

//--- symbolic-dynamic-reinterpret-disjoint.mlir
module {
  func.func @symbolic_dynamic_reinterpret_views(%root: memref<256xf32>,
                                                %row: index) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %sourceBase = memref.reinterpret_cast %root to offset: [%offset],
      sizes: [128], strides: [1] : memref<256xf32>
        to memref<128xf32, strided<[1], offset: ?>>
    %destinationBase = memref.reinterpret_cast %root to offset: [%offset],
      sizes: [128], strides: [1] : memref<256xf32>
        to memref<128xf32, strided<[1], offset: ?>>
    %source = memref.subview %sourceBase[0] [32] [1]
      : memref<128xf32, strided<[1], offset: ?>>
        to memref<32xf32, strided<[1], offset: ?>>
    %destination = memref.subview %destinationBase[64] [32] [1]
      : memref<128xf32, strided<[1], offset: ?>>
        to memref<32xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%source : memref<32xf32, strided<[1], offset: ?>>)
      outs(%destination : memref<32xf32, strided<[1], offset: ?>>) {
    ^bb0(%input: f32, %unused: f32):
      linalg.yield %input : f32
    }
    return
  }
}

// STATIC-OVERLAP: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
// SYMBOLIC-DISJOINT: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
// SYMBOLIC-OVERLAP: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
// SYMBOLIC-DYNAMIC-DISJOINT: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
