// RUN: split-file %s %t
// RUN: not buddy-opt %t/overlap.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not buddy-opt %t/cross-row-overlap.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=CROSS-ROW
// RUN: not buddy-opt %t/short.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=SHORT
// RUN: not buddy-opt %t/unaligned.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/physical-transpose.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=TRANSPOSE
// RUN: not buddy-opt %t/generic-transpose.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=TRANSPOSE
// RUN: not buddy-opt %t/parallel.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=SERIAL
// RUN: not buddy-opt %t/parallel-caller.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=SERIAL-CALLER

// Rejections must return nonzero; silently keeping an RVV pure copy does not
// implement the DMA contract, and copying across gaps is not a legal fallback.

// Preserved physical fixtures have no logical initialization for their SPM
// roots, so contract content validation runs before target DMA legality. The
// direct parallel SPM-to-SPM copy has no V1 boundary recipe. Typed overlap,
// alignment and asynchronous liveness remain checked by the BOSCAME dialect
// tests; PlanWorkspaceCalls.py/PlanWorkspaceEscapes.py cover workspace safety.
//--- overlap.mlir
module {
  func.func @overlapping_subviews() {
    %bank = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<192xi8, #bosc_ame.spm<0>>
    %source = memref.subview %bank[0] [128] [1]
      : memref<192xi8, #bosc_ame.spm<0>> to memref<128xi8, strided<[1]>, #bosc_ame.spm<0>>
    %target = memref.subview %bank[64] [128] [1]
      : memref<192xi8, #bosc_ame.spm<0>> to memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    memref.copy %source, %target
      : memref<128xi8, strided<[1]>, #bosc_ame.spm<0>>
      to memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    return
  }
}

//--- cross-row-overlap.mlir
// Each individual request has disjoint endpoints. However, row zero's
// destination overwrites row one's source before the second request reads it.
module {
  func.func @cross_row_overlap() {
    %bank = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<320xi8, #bosc_ame.spm<0>>
    %source = memref.reinterpret_cast %bank to offset: [0], sizes: [2, 64], strides: [128, 1]
      : memref<320xi8, #bosc_ame.spm<0>> to memref<2x64xi8, strided<[128, 1]>, #bosc_ame.spm<0>>
    %target = memref.reinterpret_cast %bank to offset: [128], sizes: [2, 64], strides: [128, 1]
      : memref<320xi8, #bosc_ame.spm<0>> to memref<2x64xi8, strided<[128, 1], offset: 128>, #bosc_ame.spm<0>>
    memref.copy %source, %target
      : memref<2x64xi8, strided<[128, 1]>, #bosc_ame.spm<0>>
      to memref<2x64xi8, strided<[128, 1], offset: 128>, #bosc_ame.spm<0>>
    return
  }
}

//--- short.mlir
module {
  func.func @short_copy() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<1>>
    %source_short = memref.subview %source[0] [63] [1]
      : memref<64xi8, #bosc_ame.spm<0>> to memref<63xi8, strided<[1]>, #bosc_ame.spm<0>>
    %target_short = memref.subview %target[0] [63] [1]
      : memref<64xi8, #bosc_ame.spm<1>> to memref<63xi8, strided<[1]>, #bosc_ame.spm<1>>
    memref.copy %source_short, %target_short
      : memref<63xi8, strided<[1]>, #bosc_ame.spm<0>>
      to memref<63xi8, strided<[1]>, #bosc_ame.spm<1>>
    return
  }
}

//--- unaligned.mlir
module {
  func.func @unaligned_source() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<128xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<1>>
    %unaligned = memref.subview %source[1] [64] [1]
      : memref<128xi8, #bosc_ame.spm<0>> to memref<64xi8, strided<[1], offset: 1>, #bosc_ame.spm<0>>
    memref.copy %unaligned, %target
      : memref<64xi8, strided<[1], offset: 1>, #bosc_ame.spm<0>>
      to memref<64xi8, #bosc_ame.spm<1>>
    return
  }
}

//--- physical-transpose.mlir
module {
  func.func @different_physical_axes() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<4096xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<4096xi8, #bosc_ame.spm<1>>
    %columns = memref.reinterpret_cast %source to offset: [0], sizes: [64, 64], strides: [1, 64]
      : memref<4096xi8, #bosc_ame.spm<0>> to memref<64x64xi8, strided<[1, 64]>, #bosc_ame.spm<0>>
    %rows = memref.reinterpret_cast %target to offset: [0], sizes: [64, 64], strides: [64, 1]
      : memref<4096xi8, #bosc_ame.spm<1>> to memref<64x64xi8, strided<[64, 1]>, #bosc_ame.spm<1>>
    memref.copy %columns, %rows
      : memref<64x64xi8, strided<[1, 64]>, #bosc_ame.spm<0>>
      to memref<64x64xi8, strided<[64, 1]>, #bosc_ame.spm<1>>
    return
  }
}

//--- generic-transpose.mlir
module {
  func.func @transposed_indexing_map() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<16x16xf32, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<16x16xf32, #bosc_ame.spm<1>>
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (j, i)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%source : memref<16x16xf32, #bosc_ame.spm<0>>)
      outs(%target : memref<16x16xf32, #bosc_ame.spm<1>>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }
}

//--- parallel.mlir
module {
  func.func @parallel_copy() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<1>>
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      memref.copy %source, %target
        : memref<64xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<1>>
      scf.reduce
    }
    return
  }
}

//--- parallel-caller.mlir
module {
  func.func @copy_helper() {
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<64xi8, #bosc_ame.spm<1>>
    memref.copy %source, %target
      : memref<64xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<1>>
    return
  }
  func.func @parallel_caller() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      func.call @copy_helper() : () -> ()
      scf.reduce
    }
    return
  }
}

// OVERLAP: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CROSS-ROW: kernel contract V22/V23: read or partial inheritance has no initialized content version
// SHORT: kernel contract V22/V23: read or partial inheritance has no initialized content version
// ALIGN: kernel contract V22/V23: read or partial inheritance has no initialized content version
// TRANSPOSE: kernel contract V22/V23: read or partial inheritance has no initialized content version
// SERIAL-CALLER: kernel contract V22/V23: read or partial inheritance has no initialized content version
// SERIAL: v1 planned copy requires exactly one external DDR endpoint
