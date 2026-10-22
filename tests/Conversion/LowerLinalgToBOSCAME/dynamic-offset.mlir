// RUN: split-file %s %t
// RUN: buddy-opt %t/aligned.mlir --lower-linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s
// RUN: not buddy-opt %t/unknown.mlir --lower-linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=UNKNOWN

// Multiplying an f32 element offset by 16 proves 64-byte alignment. The
// emitted descriptor must preserve that arithmetic proof through DMA checking.
// An arbitrary runtime offset must fail instead of assuming the ABI covers it.
// CHECK-LABEL: func.func @aligned
// CHECK: arith.muli
// CHECK: memref.subview
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: %[[COPY:.*]] = bosc_ame.dma.start
// CHECK-SAME: (memref<16xf32, {{.*}}#bosc_ame.spm<0>>, memref<16xf32, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[COPY]]
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
// UNKNOWN: endpoint effective address cannot be proven 64 B aligned

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
