// RUN: split-file %s %t
// RUN: not buddy-opt %t/disjoint.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=DISJOINT
// RUN: not buddy-opt %t/overlap.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga ' 2>&1 | FileCheck %s --check-prefix=OVERLAP

// The two 8x16 views have 64-byte rows separated by a 256-byte pitch.
// Their bounding spans overlap, while their individual row ranges do not.
// V1 conservatively rejects both this unproved row separation and actual
// overlap at the contract boundary. Legal pitched AME access remains covered
// by pitched-transfer.mlir and PlanEmitValidation.py.

//--- disjoint.mlir
module {
  func.func @pitched_disjoint(%root: memref<8x64xf32>) {
    %input = memref.subview %root[0, 0] [8, 16] [1, 1]
      : memref<8x64xf32> to memref<8x16xf32, strided<[64, 1]>>
    %output = memref.subview %root[0, 32] [8, 16] [1, 1]
      : memref<8x64xf32> to
        memref<8x16xf32, strided<[64, 1], offset: 32>>
    linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                     affine_map<(i, j) -> (i, j)>],
                    iterator_types = ["parallel", "parallel"],
                    bosc_ame.memory = {kind = "store",
                                       destination = "external_ddr",
                                       logical_shape = array<i64: 8, 16>,
                                       valid_shape = array<i64: 8, 16>,
                                       masked = false}}
      ins(%input : memref<8x16xf32, strided<[64, 1]>>)
      outs(%output : memref<8x16xf32, strided<[64, 1], offset: 32>>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }
}

//--- overlap.mlir
module {
  func.func @pitched_overlap(%root: memref<8x64xf32>) {
    %input = memref.subview %root[0, 0] [8, 32] [1, 1]
      : memref<8x64xf32> to memref<8x32xf32, strided<[64, 1]>>
    %output = memref.subview %root[0, 16] [8, 32] [1, 1]
      : memref<8x64xf32> to
        memref<8x32xf32, strided<[64, 1], offset: 16>>
    linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                     affine_map<(i, j) -> (i, j)>],
                    iterator_types = ["parallel", "parallel"],
                    bosc_ame.memory = {kind = "store",
                                       destination = "external_ddr",
                                       logical_shape = array<i64: 8, 32>,
                                       valid_shape = array<i64: 8, 32>,
                                       masked = false}}
      ins(%input : memref<8x32xf32, strided<[64, 1]>>)
      outs(%output : memref<8x32xf32, strided<[64, 1], offset: 16>>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// DISJOINT: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
// OVERLAP: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
