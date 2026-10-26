// RUN: split-file %s %t
// RUN: not buddy-opt %t/no-unit-stride.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=STRIDE
// RUN: not buddy-opt %t/negative-stride.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=STRIDE-SIGN
// RUN: not buddy-opt %t/unknown-offset.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/dynamic-shape.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not buddy-opt %t/misaligned.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/misaligned-row.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/overlapping-rows.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not buddy-opt %t/capacity.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=CAPACITY
// RUN: buddy-opt %t/simultaneous-capacity.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s --check-prefix=DIRECT --implicit-check-not=bosc_ame.dma.start --implicit-check-not=bosc_ame.spm.view
// RUN: not buddy-opt %t/element-type.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=ELEMENT

// The negative cases fail before any AME tile is emitted.
// Every rejected command must return nonzero; a diagnostic printed alongside
// apparently successful IR is insufficient.

// Identity generics normalize to DDR-to-DDR copies; V1 rejects that boundary
// before the original per-transfer layout/capacity checks. Dynamic identity
// generic is outside the V1 compute subset. The large addition uses direct
// DDR with no SPM capacity demand. Actual route failure and private-capacity
// diagnostics are checked by PlanRoutes.mlir/py and PlanEmitValidation.py.
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

// STRIDE: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// STRIDE-SIGN: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// ALIGN: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// OVERLAP: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// CAPACITY: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// ELEMENT: v1 has no independent DDR-to-DDR copy recipe; a boundary copy requires a planned private endpoint
// SHAPE: v1 vector plan requires rank-one identity pointwise
// DIRECT-LABEL: func.func @add
// DIRECT: vector.load {{.*}} {bosc_ame.direct_ddr}
// DIRECT: vector.load {{.*}} {bosc_ame.direct_ddr}
// DIRECT: arith.addf
// DIRECT: vector.store {{.*}} {bosc_ame.direct_ddr}
