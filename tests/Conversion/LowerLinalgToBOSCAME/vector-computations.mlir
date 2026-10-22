// RUN: buddy-opt %s --lower-linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=COPY < %t

// Fill and unused generic output arguments must not cause a DDR read. An
// accumulator does require its initial value. Scalar broadcasts stay in DDR
// and are loaded before the SPM vector computation begins.
module {
  func.func @fill(%out: memref<16xf32>, %value: f32) {
    linalg.fill ins(%value : f32) outs(%out : memref<16xf32>)
    return
  }

  func.func @copy(%input: memref<16xf32>, %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }

  func.func @accumulate(%input: memref<16xf32>, %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %initial: f32):
      %sum = arith.addf %x, %initial : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func @scalar_broadcast(%input: memref<16xf32>, %scale: memref<f32>,
                              %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> ()>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input, %scale : memref<16xf32>, memref<f32>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %scalar: f32, %unused: f32):
      %product = arith.mulf %x, %scalar : f32
      linalg.yield %product : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @fill
// CHECK-NOT: bosc_ame.dma.start
// CHECK: vector.broadcast
// CHECK: vector.store {{.*}} : memref<16xf32, {{.*}}#bosc_ame.spm<0>>, vector<16xf32>
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return

// CHECK-LABEL: func.func @copy
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK: %[[COPY_TOKEN:.*]] = bosc_ame.dma.start
// CHECK-SAME: (memref<16xf32, {{.*}}#bosc_ame.spm<0>>, memref<16xf32, {{.*}}#bosc_ame.spm<1>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[COPY_TOKEN]]
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return

// COPY-LABEL: func.func @copy
// COPY-NOT: vector.load
// COPY-NOT: vector.store
// COPY: return

// CHECK-LABEL: func.func @accumulate
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: vector.load
// CHECK: vector.load
// CHECK: arith.addf {{.*}} : vector<16xf32>
// CHECK: vector.store
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return

// CHECK-LABEL: func.func @scalar_broadcast
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: memref.load {{.*}}[] : memref<f32>
// CHECK: vector.load
// CHECK: vector.broadcast
// CHECK: arith.mulf {{.*}} : vector<16xf32>
// CHECK: vector.store
// CHECK: bosc_ame.dma.start
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
