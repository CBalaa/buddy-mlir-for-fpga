// RUN: buddy-opt %s --linalg-to-boscame='plan-report=true' -o %t.ir 2>%t.plan
// RUN: buddy-opt %s --linalg-to-boscame='' -o %t.default
// RUN: cmp %t.default %t.ir
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan

module {
  func.func @consistent_origin(%source: memref<16xf32>,
                               %destination: memref<16xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"],
      bosc_ame.memory = {kind = "store",
                         logical_shape = array<i64: 16>, masked = false,
                         origin_id = 77 : i64, valid_shape = array<i64: 16>}
    } ins(%source : memref<16xf32>) outs(%destination : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @conflicting_origin(%source: memref<16xf32>,
                                %destination: memref<16xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"],
      bosc_ame.memory = {kind = "store",
                         logical_shape = array<i64: 16>, masked = false,
                         origin_id = 77 : i64, valid_shape = array<i64: 16>},
      bosc_ame.origin_id = 78 : i64
    } ins(%source : memref<16xf32>) outs(%destination : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// PLAN: ir-node[2]: op=linalg.generic {{.*}}memory-event={kind=copy role=store effect=write origin-id=77 inputs=[0] outputs=[1]}
// PLAN: ir-node[6]: op=linalg.generic {{.*}}memory-event={kind=copy role=store effect=write origin-id=<unknown> inputs=[4] outputs=[5]}
// PLAN: buffer-source-events: count=1 scope=pre-accumulator-fusion
// PLAN: buffer-source-event[0]: function=@consistent_origin source-origin-id=77 op=linalg.generic effect=write
// PLAN: buffer-accesses: count=4 scope=post-planning
// PLAN: buffer-access[1]: op=linalg.generic effect=write effect-index=1 operand=1 source-origin-id=77
// PLAN: buffer-access[3]: op=linalg.generic effect=write effect-index=1 operand=1
// PLAN-NOT: source-origin-id=
// PLAN: memory-dependencies:
