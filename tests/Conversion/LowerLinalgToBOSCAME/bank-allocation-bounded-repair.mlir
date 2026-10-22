// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga store-policy=direct-ddr plan-report=true' -o /dev/null 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: FileCheck %s --check-prefix=DIAG < %t.plan

// This fixture uses heterogeneous projected input shapes to exercise a repair
// that a valid current SPM generic cannot express. The planner report is
// emitted before the SPM generic verifier rejects the operation. Sorted greedy
// placement splits the ten 18,752-byte inputs evenly and cannot place the third
// 18,688-byte input. A feasible packing moves one large input so four large
// and three small windows exactly fill one bank.
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

// PLAN: bank-allocation-search: scope=operation-local groups=1 exact-assignments=0 exact-rejected-assignments=0 bounded-groups=1 repair-groups=1 repair-search-nodes=18 repair-rejected-placements=6 objective=minimize-bank-high-water status=bounded latency=uncalibrated
// PLAN: bank[0]: high-water=112512/131072 bytes
// PLAN: bank[1]: high-water=131072/131072 bytes
// DIAG: error: SPM generic requires matching shapes, element byte widths and identity indexing maps
