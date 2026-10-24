// RUN: buddy-opt %s --linalg-to-boscame='plan-report=true dma-abi=args_aligned64_nonalias' -o %t.reported 2>%t.report
// RUN: buddy-opt %s --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o %t.plain
// RUN: cmp %t.reported %t.plain
// RUN: FileCheck %s < %t.report

module {
  func.func @metadata_free(%source: memref<16xf32>, %destination: memref<16xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%source : memref<16xf32>)
      outs(%destination : memref<16xf32>) {
    ^bb0(%input: f32, %unused: f32):
      %negated = arith.negf %input : f32
      linalg.yield %negated : f32
    }
    return
  }
}

// Standard Linalg semantics, rather than optional frontend graph attributes,
// produce the graph. Reporting only reads the recorded execution decisions.
// CHECK: linalg.generic plan.vector={elements = 16 : i64, engine = "rvv"
// CHECK-SAME: fp_policy = "preserve"
// CHECK-SAME: level = "ddr", operand = 0 : i64
// CHECK-SAME: level = "ddr", operand = 1 : i64
// CHECK-SAME: reason = "pointwise-memory-operand"
