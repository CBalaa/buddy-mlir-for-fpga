// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga bm=64 bn=64 bk=128' -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=DMA < %t
// RUN: FileCheck %s --check-prefix=TILES < %t
// RUN: buddy-opt %t --lower-bosc-ame | FileCheck %s --check-prefix=LOWERED

// One 64x128 program of the M=128, N=128, K=1024 workload. B is already in
// physical [N,K] storage. Two reads stage B and A; the legal accumulator
// tiles read and write C directly. No DDR packing is needed.
module {
  func.func @gemm(%a: memref<64x1024xi8>,
                  %b: memref<1024x128xi8, strided<[1, 1024]>>,
                  %c: memref<64x128xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x1024xi8>,
                   memref<1024x128xi8, strided<[1, 1024]>>)
      outs(%c : memref<64x128xi32>)
    return
  }
}

// CHECK-LABEL: func.func @gemm
// CHECK-SAME: %[[A:[[:alnum:]_]+]]: memref<64x1024xi8>,
// CHECK-SAME: %[[B:[[:alnum:]_]+]]: memref<1024x128xi8, strided<[1, 1024]>>,
// CHECK-SAME: %[[C:[[:alnum:]_]+]]: memref<64x128xi32>
// CHECK-DAG: %[[BYTES_B:.*]] = arith.constant 131072 : index
// CHECK-DAG: %[[BYTES_A:.*]] = arith.constant 65536 : index
// CHECK: %[[SPM_B:.*]] = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<131072xi8, #bosc_ame.spm<0>>
// CHECK: %[[DDR_B:.*]] = memref.reinterpret_cast %[[B]] to offset: {{.*}}, sizes: [131072], strides: [1]
// CHECK: %[[DMA_B_DST:.*]] = memref.reinterpret_cast %[[SPM_B]]
// CHECK: %[[TOKEN_B:.*]] = bosc_ame.dma.start %[[DDR_B]], %[[DMA_B_DST]], %[[BYTES_B]]
// CHECK: bosc_ame.dma.wait %[[TOKEN_B]]
// CHECK: %[[SPM_A:.*]] = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<65536xi8, #bosc_ame.spm<1>>
// CHECK: %[[DDR_A:.*]] = memref.reinterpret_cast %[[A]] to offset: {{.*}}, sizes: [65536], strides: [1]
// CHECK: %[[DMA_A_DST:.*]] = memref.reinterpret_cast %[[SPM_A]]
// CHECK: %[[TOKEN_A:.*]] = bosc_ame.dma.start %[[DDR_A]], %[[DMA_A_DST]], %[[BYTES_A]]
// CHECK: bosc_ame.dma.wait %[[TOKEN_A]]
// CHECK-NOT: bosc_ame.dma.start
// CHECK: bosc_ame.tile.load_acc {{.*}} {bosc_ame.direct_ddr}
// CHECK: bosc_ame.tile.mma
// CHECK: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr}
// CHECK: bosc_ame.tile.load_acc {{.*}} {bosc_ame.direct_ddr}
// CHECK: bosc_ame.tile.mma
// CHECK: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr}
// CHECK-NOT: bosc_ame.tile.mma
// CHECK: return

// DMA-LABEL: func.func @gemm
// DMA-COUNT-2: bosc_ame.dma.start
// DMA-NOT: bosc_ame.dma.start
// DMA: return

// TILES-LABEL: func.func @gemm
// TILES: bosc_ame.tile.load_acc
// TILES-COUNT-16: bosc_ame.tile.mma
// TILES-NOT: bosc_ame.tile.mma
// TILES: bosc_ame.tile.store_acc
// TILES: bosc_ame.tile.load_acc
// TILES-COUNT-16: bosc_ame.tile.mma
// TILES-NOT: bosc_ame.tile.mma
// TILES: bosc_ame.tile.store_acc
// TILES-NOT: bosc_ame.tile.mma
// TILES: return

// The output is accepted directly by lower-bosc-ame. In particular no
// runtime call may interrupt the live AME accumulator chain.
// LOWERED-LABEL: llvm.func @gemm
// LOWERED-COUNT-2: llvm.call @bosc_ame_dma_start
// LOWERED: bosc_ame.intr.mlce32.m
// LOWERED-NOT: llvm.call
// LOWERED: bosc_ame.intr.msce32.m
// LOWERED-NOT: llvm.call
// LOWERED: llvm.return
