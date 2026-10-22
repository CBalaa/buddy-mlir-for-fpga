// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t
// RUN: FileCheck %s < %t

module {
  func.func @transpose_memref_writer(
      %source: memref<4x4xf32>, %buffer: memref<4x4xf32>,
      %destination: memref<4x4xf32>) {
    %transposed = memref.transpose %buffer (i, j) -> (j, i)
      : memref<4x4xf32> to
        memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
    memref.copy %source, %transposed {
      bosc_ame.memory = {kind = "load", origin_id = 101 : i64}
    } : memref<4x4xf32> to
        memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
    memref.copy %buffer, %destination {
      bosc_ame.memory = {kind = "store", origin_id = 102 : i64}
    } : memref<4x4xf32> to memref<4x4xf32>
    return
  }

  func.func @transpose_exit_consumer(%input: memref<4x4xf32>,
      %other: memref<4x4xf32>, %destination: memref<4x4xf32>,
      %condition: i1, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%iter = %input) -> (memref<4x4xf32>) {
      %selected = scf.if %condition -> (memref<4x4xf32>) {
        scf.yield %iter : memref<4x4xf32>
      } else {
        scf.yield %other : memref<4x4xf32>
      }
      scf.yield %selected : memref<4x4xf32>
    }
    %transposed = memref.transpose %loop (i, j) -> (j, i)
      : memref<4x4xf32> to
        memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
    memref.copy %transposed, %destination {
      bosc_ame.memory = {kind = "store", origin_id = 103 : i64}
    } : memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
        to memref<4x4xf32>
    return
  }
}

// CHECK-LABEL: func.func @transpose_memref_writer
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 101 : i64
// CHECK: memref.copy {{.*}}bosc_ame.compute_origin_id = 102 : i64
// CHECK-SAME: bosc_ame.source_load_origin_ids = array<i64: 101>
// CHECK-LABEL: func.func @transpose_exit_consumer
// CHECK: bosc_ame.loop_carried_slots = [{{.*}}exit_consumer_compute_ids = array<i64: 2>{{.*}}]
