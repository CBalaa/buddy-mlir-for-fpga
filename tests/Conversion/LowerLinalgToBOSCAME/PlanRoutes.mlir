// RUN: buddy-opt %s --kernel-contract-annotate --bosc-ame-plan=plan-report=true --bosc-ame-plan-verify 2>&1 | FileCheck %s --check-prefix=RETRY
// RUN: buddy-opt %s --kernel-contract-annotate --bosc-ame-plan --bosc-ame-emit | FileCheck %s --check-prefix=EMIT --implicit-check-not=bosc_ame.dma.start

// A repeats across two N subtiles, so the traffic rule initially stages it.
// Both banks are occupied by private values live across the matmul. Capacity
// forces a proven direct A route, preserving the two existing windows.
// RETRY: linalg.matmul plan.route={a = {{.*}}level = "ddr"{{.*}}reason = "capacity-fallback-safe-direct"
// RETRY: high_water = array<i64: 131072, 131072>
// EMIT-LABEL: func.func @capacity_retry
// EMIT-COUNT-2: bosc_ame.spm.view
// EMIT: bosc_ame.tile.load_a {{.*}}bosc_ame.direct_ddr
// EMIT: bosc_ame.tile.load_b {{.*}}bosc_ame.direct_ddr
// EMIT: bosc_ame.tile.store_acc {{.*}}bosc_ame.direct_ddr

#identity = affine_map<(d0) -> (d0)>
module attributes {
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
} {
  func.func @capacity_retry(
      %a: memref<32x128xi8>,
      %b: memref<128x128xi8, strided<[1, 128]>>,
      %c: memref<32x128xi32>) {
    %zero = arith.constant 0 : i8
    %first = memref.alloc() alignment = 64 : memref<131072xi8>
    %second = memref.alloc() alignment = 64 : memref<131072xi8>
    linalg.fill ins(%zero : i8) outs(%first : memref<131072xi8>)
    linalg.fill ins(%zero : i8) outs(%second : memref<131072xi8>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%a, %b : memref<32x128xi8>, memref<128x128xi8, strided<[1, 128]>>)
        outs(%c : memref<32x128xi32>)
    linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]}
        ins(%first, %first : memref<131072xi8>, memref<131072xi8>)
        outs(%first : memref<131072xi8>) {
      ^bb0(%lhs: i8, %rhs: i8, %old: i8):
        %next = arith.addi %lhs, %rhs : i8
        linalg.yield %next : i8
    }
    linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]}
        ins(%second, %second : memref<131072xi8>, memref<131072xi8>)
        outs(%second : memref<131072xi8>) {
      ^bb0(%lhs: i8, %rhs: i8, %old: i8):
        %next = arith.addi %lhs, %rhs : i8
        linalg.yield %next : i8
    }
    memref.dealloc %first : memref<131072xi8>
    memref.dealloc %second : memref<131072xi8>
    return
  }
}
