// RUN: not buddy-opt %s '--linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s

// The projected legacy store is not a concrete buffer-copy boundary.
// The adapter rejects its memory facts before a capacity or bank search.
module {
  func.func @bounded_bank_repair(
      %x0: memref<4688xf32>, %x1: memref<4688xf32>,
      %x2: memref<4688xf32>, %x3: memref<4688xf32>,
      %x4: memref<4688xf32>, %x5: memref<4688xf32>,
      %x6: memref<4688xf32>, %x7: memref<4688xf32>,
      %x8: memref<4688xf32>, %x9: memref<4688xf32>,
      %y0: memref<4672xf32>, %y1: memref<4672xf32>,
      %y2: memref<4672xf32>, %out: memref<4688x4672xf32>) {
    linalg.generic {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
        logical_shape = array<i64: 4688, 4672>,
        valid_shape = array<i64: 4688, 4672>, masked = false},
      indexing_maps = [
        affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>,
        affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>,
        affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>,
        affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>,
        affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>,
        affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (j)>,
        affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7, %x8, %x9,
          %y0, %y1, %y2 : memref<4688xf32>, memref<4688xf32>,
          memref<4688xf32>, memref<4688xf32>, memref<4688xf32>,
          memref<4688xf32>, memref<4688xf32>, memref<4688xf32>,
          memref<4688xf32>, memref<4688xf32>, memref<4672xf32>,
          memref<4672xf32>, memref<4672xf32>)
      outs(%out : memref<4688x4672xf32>) {
    ^bb0(%v0: f32, %v1: f32, %v2: f32, %v3: f32, %v4: f32,
         %v5: f32, %v6: f32, %v7: f32, %v8: f32, %v9: f32,
         %w0: f32, %w1: f32, %w2: f32, %old: f32):
      %s0 = arith.addf %v0, %v1 : f32
      %s1 = arith.addf %s0, %v2 : f32
      %s2 = arith.addf %s1, %v3 : f32
      %s3 = arith.addf %s2, %v4 : f32
      %s4 = arith.addf %s3, %v5 : f32
      %s5 = arith.addf %s4, %v6 : f32
      %s6 = arith.addf %s5, %v7 : f32
      %s7 = arith.addf %s6, %v8 : f32
      %s8 = arith.addf %s7, %v9 : f32
      %s9 = arith.addf %s8, %w0 : f32
      %s10 = arith.addf %s9, %w1 : f32
      %s11 = arith.addf %s10, %w2 : f32
      linalg.yield %s11 : f32
    }
    return
  }
}

// CHECK: legacy memory facts require a concrete buffer copy
