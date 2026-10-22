// RUN: buddy-opt %s --linalg-to-boscame='plan-report=true enable-spm-dma=false enable-ame=false' -o %t.reported 2>%t.report
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=false enable-ame=false' -o %t.plain
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

// The explicit SSA facts exist without frontend graph attributes. They do not
// upgrade compute-graph completeness or enable cross-task residency.
// CHECK: linalg-facts: scope=post-planning
// CHECK: linalg-ssa-facts: scope=explicit-ssa operations={{[1-9][0-9]*}} values={{[1-9][0-9]*}} linalg-ops=1
// CHECK: ir-node[{{[0-9]+}}]: op=linalg.generic {{.*}} results=[] kind=elementwise dps-input-operands=[0] dps-init-operands=[1] init-payload-reads=[false] maps=[{{.*}}] iterators=[parallel] reduction-dimensions=[] payload-ops=[{{[0-9, ]+}}] tensor=false buffer=true payload-memory-effect-free=true
// CHECK: ssa-value[{{[0-9]+}}]: type=memref<16xf32> shape=[16] element-type=f32 source=ir-node#{{[0-9]+}}.region#0.block#0.arg#0 uses=[{{.*}}.operand#0]
// CHECK: compute-graph: status=unavailable tasks=0 dependencies=0 logical-values=0
