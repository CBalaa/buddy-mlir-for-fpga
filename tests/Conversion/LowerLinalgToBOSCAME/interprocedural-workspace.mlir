// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t
// RUN: buddy-opt %t --lower-bosc-ame -o /dev/null

// The caller's explicit bank-0 window stays live across the call. Staging in
// the helper must reserve this existing range even though its view occurs in
// another function, and must not overwrite the vector saved by the caller.
module {
  func.func @helper(%input: memref<16xf32>, %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }

  func.func @caller(%input: memref<16xf32>, %temporary: memref<16xf32>,
                    %out: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %two = arith.constant dense<2.0> : vector<16xf32>
    %saved = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<16xf32, #bosc_ame.spm<0>>
    vector.store %two, %saved[%c0]
      : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    func.call @helper(%input, %temporary)
      : (memref<16xf32>, memref<16xf32>) -> ()
    %value = vector.load %saved[%c0]
      : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    vector.store %value, %out[%c0] : memref<16xf32>, vector<16xf32>
    return
  }
}

// CHECK-LABEL: func.func @helper
// CHECK: bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
// CHECK: bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
// CHECK: %[[COPY:.*]] = bosc_ame.dma.start
// CHECK-SAME: (memref<16xf32, {{.*}}#bosc_ame.spm<1>>, memref<16xf32, {{.*}}#bosc_ame.spm<0>>, index)
// CHECK-NEXT: bosc_ame.dma.wait %[[COPY]]
// CHECK: bosc_ame.dma.start
// CHECK-NEXT: bosc_ame.dma.wait
// CHECK-NOT: bosc_ame.dma.start
// CHECK: return
// CHECK-LABEL: func.func @caller
// CHECK: %[[SAVED:.*]] = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
// CHECK: vector.store {{.*}}, %[[SAVED]]
// CHECK: call @helper
// CHECK: vector.load %[[SAVED]]
// CHECK: return
