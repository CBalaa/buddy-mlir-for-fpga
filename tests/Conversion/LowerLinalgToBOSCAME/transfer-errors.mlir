// RUN: split-file %s %t
// RUN: not buddy-opt %t/no-unit-stride.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=STRIDE
// RUN: not buddy-opt %t/negative-stride.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=STRIDE-SIGN
// RUN: not buddy-opt %t/unknown-offset.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/dynamic-shape.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not buddy-opt %t/misaligned.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/misaligned-row.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/overlapping-rows.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not buddy-opt %t/capacity.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=CAPACITY
// RUN: not buddy-opt %t/simultaneous-capacity.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=CAPACITY
// RUN: not buddy-opt %t/element-type.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ELEMENT

// These are transfer-contract failures, before any AME tile is emitted.
// Every rejected command must return nonzero; a diagnostic printed alongside
// apparently successful IR is insufficient.
// STRIDE: BOSCAME transfer has no unit-stride physical axis
// STRIDE-SIGN: BOSCAME transfer requires static positive strides
// SHAPE: BOSCAME transfer shape must be static
// ALIGN: 64 B aligned
// OVERLAP: physical rows overlap
// CAPACITY: BOSCAME SPM workspace capacity exceeded
// ELEMENT: unsupported typed DMA element type

//--- no-unit-stride.mlir
module {
  func.func @copy(%a: memref<16xf32, strided<[2], offset: 0>>,
                  %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<16xf32, strided<[2], offset: 0>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- negative-stride.mlir
module {
  func.func @copy(%a: memref<16xf32, strided<[-1], offset: 15>>,
                  %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<16xf32, strided<[-1], offset: 15>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- unknown-offset.mlir
module {
  func.func @copy(%a: memref<16xf32, strided<[1], offset: ?>>,
                  %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<16xf32, strided<[1], offset: ?>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- dynamic-shape.mlir
module {
  func.func @copy(%a: memref<?xf32>, %out: memref<?xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<?xf32>) outs(%out : memref<?xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- misaligned.mlir
module {
  func.func @copy(%a: memref<16xf32, strided<[1], offset: 1>>,
                  %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<16xf32, strided<[1], offset: 1>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- misaligned-row.mlir
module {
  func.func @copy(%a: memref<2x16xf32, strided<[17, 1], offset: 0>>,
                  %out: memref<2x16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                     affine_map<(i, j) -> (i, j)>],
                    iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<2x16xf32, strided<[17, 1], offset: 0>>)
      outs(%out : memref<2x16xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- overlapping-rows.mlir
module {
  func.func @copy(%a: memref<2x16xf32, strided<[8, 1], offset: 0>>,
                  %out: memref<2x16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                     affine_map<(i, j) -> (i, j)>],
                    iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<2x16xf32, strided<[8, 1], offset: 0>>)
      outs(%out : memref<2x16xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- capacity.mlir
module {
  // 32784 f32 values = 131136 bytes, one 64-byte packet beyond a bank.
  func.func @copy(%a: memref<32784xf32>, %out: memref<32784xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<32784xf32>) outs(%out : memref<32784xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- element-type.mlir
module {
  func.func @copy(%a: memref<128xi4>, %out: memref<128xi4>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<128xi4>) outs(%out : memref<128xi4>) {
    ^bb0(%x: i4, %y: i4):
      linalg.yield %x : i4
    }
    return
  }
}

//--- simultaneous-capacity.mlir
module {
  // Each individual buffer fits a bank, but three simultaneously live
  // 128 KiB buffers cannot share the two banks.
  func.func @add(%a: memref<32768xf32>, %b: memref<32768xf32>,
                 %out: memref<32768xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a, %b : memref<32768xf32>, memref<32768xf32>)
      outs(%out : memref<32768xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }
}
