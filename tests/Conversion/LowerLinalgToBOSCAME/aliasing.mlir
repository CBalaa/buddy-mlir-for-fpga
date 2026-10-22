// RUN: split-file %s %t
// RUN: not buddy-opt %t/overlap.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not buddy-opt %t/shared-global.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: buddy-opt %t/disjoint.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s --check-prefix=DISJOINT
// RUN: not buddy-opt %t/unknown.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: buddy-opt %t/dynamic-disjoint.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s --check-prefix=DYNAMIC-DISJOINT
// RUN: not buddy-opt %t/dynamic-overlap.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=DYNAMIC-OVERLAP
// RUN: not buddy-opt %t/dynamic-bases.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=DYNAMIC-BASES
// RUN: not buddy-opt %t/dynamic-size.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=DYNAMIC-SIZE
// RUN: buddy-opt %t/safe.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s

// Distinct aligned views can still overlap even under the nonaliasing root
// ABI. Read-only aliases are legal, and repeated uses of exactly the same
// SSA buffer share one read/write staging window.
// OVERLAP: cannot prove distinct writable views of one DDR root do not alias
// GLOBAL: cannot prove distinct writable views of one DDR root do not alias
// DISJOINT-LABEL: func.func @disjoint
// DISJOINT: bosc_ame.dma.start
// DISJOINT: bosc_ame.dma.start
// DISJOINT: return
// UNKNOWN: cannot prove distinct writable views of one DDR root do not alias
// DYNAMIC-DISJOINT-LABEL: func.func @dynamic_disjoint
// DYNAMIC-DISJOINT: bosc_ame.dma.start
// DYNAMIC-DISJOINT: bosc_ame.dma.start
// DYNAMIC-DISJOINT: return
// DYNAMIC-OVERLAP: cannot prove distinct writable views of one DDR root do not alias
// DYNAMIC-BASES: cannot prove distinct writable views of one DDR root do not alias
// DYNAMIC-SIZE: cannot prove distinct writable views of one DDR root do not alias

//--- overlap.mlir
module {
  func.func @overlap(%root: memref<48xf32>) {
    %input = memref.subview %root[0] [32] [1]
      : memref<48xf32> to memref<32xf32, strided<[1]>>
    %out = memref.subview %root[16] [32] [1]
      : memref<48xf32> to memref<32xf32, strided<[1], offset: 16>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32xf32, strided<[1]>>)
      outs(%out : memref<32xf32, strided<[1], offset: 16>>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- disjoint.mlir
module {
  func.func @disjoint(%root: memref<32xf32>) {
    %input = memref.subview %root[0] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1]>>
    %output = memref.subview %root[16] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: 16>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32, strided<[1]>>)
      outs(%output : memref<16xf32, strided<[1], offset: 16>>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- shared-global.mlir
module {
  memref.global "private" @shared : memref<16xf32> = uninitialized alignment = 64

  func.func @same_global() {
    %input = memref.get_global @shared : memref<16xf32>
    %output = memref.get_global @shared : memref<16xf32>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%output : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- unknown.mlir
module {
  func.func @unknown(%root: memref<32xf32>, %row: index) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %input = memref.subview %root[%offset] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %output = memref.subview %root[0] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1]>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32, strided<[1], offset: ?>>)
      outs(%output : memref<16xf32, strided<[1]>>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- safe.mlir
module {
  func.func @readonly(%root: memref<16xf32>, %out: memref<16xf32>) {
    %left = memref.subview %root[0] [16] [1]
      : memref<16xf32> to memref<16xf32, strided<[1]>>
    %right = memref.subview %root[0] [16] [1]
      : memref<16xf32> to memref<16xf32, strided<[1]>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%left, %right : memref<16xf32, strided<[1]>>,
                           memref<16xf32, strided<[1]>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func @same_buffer(%buffer: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%buffer : memref<16xf32>) outs(%buffer : memref<16xf32>) {
    ^bb0(%x: f32, %initial: f32):
      %sum = arith.addf %x, %initial : f32
      linalg.yield %sum : f32
    }
    return
  }
}

//--- dynamic-disjoint.mlir
module {
  func.func @dynamic_disjoint(%root: memref<128xf32>, %row: index) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %base = memref.subview %root[%offset] [64] [1]
      : memref<128xf32> to memref<64xf32, strided<[1], offset: ?>>
    %input = memref.subview %base[0] [32] [1]
      : memref<64xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    %output = memref.subview %base[32] [32] [1]
      : memref<64xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32xf32, strided<[1], offset: ?>>)
      outs(%output : memref<32xf32, strided<[1], offset: ?>>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- dynamic-overlap.mlir
module {
  func.func @dynamic_overlap(%root: memref<128xf32>, %row: index) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %base = memref.subview %root[%offset] [48] [1]
      : memref<128xf32> to memref<48xf32, strided<[1], offset: ?>>
    %input = memref.subview %base[0] [32] [1]
      : memref<48xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    %output = memref.subview %base[16] [32] [1]
      : memref<48xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32xf32, strided<[1], offset: ?>>)
      outs(%output : memref<32xf32, strided<[1], offset: ?>>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- dynamic-bases.mlir
module {
  func.func @dynamic_bases(%root: memref<128xf32>, %rowA: index,
                           %rowB: index) {
    %c16 = arith.constant 16 : index
    %offsetA = arith.muli %rowA, %c16 : index
    %offsetB = arith.muli %rowB, %c16 : index
    %input = memref.subview %root[%offsetA] [32] [1]
      : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    %output = memref.subview %root[%offsetB] [32] [1]
      : memref<128xf32> to memref<32xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32xf32, strided<[1], offset: ?>>)
      outs(%output : memref<32xf32, strided<[1], offset: ?>>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- dynamic-size.mlir
module {
  func.func @dynamic_size(%root: memref<128xf32>, %row: index,
                          %extent: index) {
    %c16 = arith.constant 16 : index
    %offset = arith.muli %row, %c16 : index
    %base = memref.subview %root[%offset] [%extent] [1]
      : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>>
    %input = memref.subview %base[0] [32] [1]
      : memref<?xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    %output = memref.subview %base[32] [32] [1]
      : memref<?xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32xf32, strided<[1], offset: ?>>)
      outs(%output : memref<32xf32, strided<[1], offset: ?>>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @readonly
// CHECK: vector.load
// CHECK: vector.load
// CHECK: arith.addf {{.*}} : vector<16xf32>
// CHECK: vector.store
// CHECK: return
// CHECK-LABEL: func.func @same_buffer
// CHECK: bosc_ame.spm.view
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.spm.view
// CHECK-NOT: bosc_ame.dma.start
// CHECK: arith.addf {{.*}} : vector<16xf32>
// CHECK: vector.store
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.spm.view
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
