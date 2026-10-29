// RUN: buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s --implicit-check-not=bosc_ame.spm.view --implicit-check-not=bosc_ame.dma.start
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

// Preserve the original thirteen-input arithmetic chain. Its supported scalar
// DAG uses external RVV operands directly and needs no private bank capacity.
module {
  func.func @large_greedy(
      %x0: memref<16xf32>, %x1: memref<16xf32>,
      %x2: memref<16xf32>, %x3: memref<16xf32>,
      %x4: memref<16xf32>, %x5: memref<16xf32>,
      %x6: memref<16xf32>, %x7: memref<16xf32>,
      %x8: memref<16xf32>, %x9: memref<16xf32>,
      %x10: memref<16xf32>, %x11: memref<16xf32>,
      %x12: memref<16xf32>, %out: memref<16xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7, %x8, %x9,
          %x10, %x11, %x12 : memref<16xf32>, memref<16xf32>,
          memref<16xf32>, memref<16xf32>, memref<16xf32>,
          memref<16xf32>, memref<16xf32>, memref<16xf32>,
          memref<16xf32>, memref<16xf32>, memref<16xf32>,
          memref<16xf32>, memref<16xf32>)
      outs(%out : memref<16xf32>) {
    ^bb0(%v0: f32, %v1: f32, %v2: f32, %v3: f32, %v4: f32,
         %v5: f32, %v6: f32, %v7: f32, %v8: f32, %v9: f32,
         %v10: f32, %v11: f32, %v12: f32, %old: f32):
      %s0 = arith.addf %v0, %v1 : f32
      %s1 = arith.addf %s0, %v2 : f32
      %s2 = arith.addf %s1, %v3 : f32
      %s3 = arith.addf %s2, %v4 : f32
      %s4 = arith.addf %s3, %v5 : f32
      %s5 = arith.addf %s4, %v6 : f32
      %s6 = arith.addf %s5, %v7 : f32
      %s7 = arith.addf %s6, %v8 : f32
      %s8 = arith.addf %s7, %v9 : f32
      %s9 = arith.addf %s8, %v10 : f32
      %s10 = arith.addf %s9, %v11 : f32
      %s11 = arith.addf %s10, %v12 : f32
      linalg.yield %s11 : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @large_greedy
// CHECK-COUNT-13: vector.load {{.*}} {bosc_ame.direct_ddr}
// CHECK-COUNT-12: arith.addf {{.*}} : vector<16xf32>
// CHECK: vector.store {{.*}} {bosc_ame.direct_ddr}
