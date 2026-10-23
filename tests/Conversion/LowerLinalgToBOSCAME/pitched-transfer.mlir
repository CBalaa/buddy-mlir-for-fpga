// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=COUNT < %t
// RUN: buddy-opt %t --lower-bosc-ame | FileCheck %s --check-prefix=LOWERED

// B has 64 contiguous K bytes per physical row and a 128-byte DDR pitch.
// Its 32 packets pack only the logical bytes into SPM with a 64-byte pitch;
// the gaps are neither transferred nor exposed to the AME consumer.
module {
  func.func @pitched_b(%a: memref<64x64xi8>,
                       %b: memref<64x32xi8, strided<[1, 128]>>,
                       %c: memref<64x32xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x32xi8, strided<[1, 128]>>)
      outs(%c : memref<64x32xi32>)
    return
  }

  // The type's 64-element offset is relative to the aligned pointer.
  // Rebuilding a flat endpoint must retain it exactly once.
  func.func @static_offset(%input: memref<64xi8, strided<[1], offset: 64>>,
                           %out: memref<64xi8>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<64xi8, strided<[1], offset: 64>>)
      outs(%out : memref<64xi8>) {
    ^bb0(%x: i8, %unused: i8):
      linalg.yield %x : i8
    }
    return
  }
}

// CHECK-LABEL: func.func @pitched_b
// CHECK-SAME: %[[A:[[:alnum:]_]+]]: memref<64x64xi8>,
// CHECK-SAME: %[[B:[[:alnum:]_]+]]: memref<64x32xi8, strided<[1, 128]>>,
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SIXTY_FOUR:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[PITCH:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[LAST_DDR:.*]] = arith.constant 3968 : index
// CHECK-DAG: %[[LAST_SPM:.*]] = arith.constant 1984 : index
// CHECK: %[[SPM_B:.*]] = bosc_ame.spm.view {{.*}} : memref<2048xi8, #bosc_ame.spm<1>>
// CHECK: %[[DDR_0:.*]] = memref.reinterpret_cast %[[B]] to offset: [%[[ZERO]]], sizes: [64], strides: [1]
// CHECK: %[[SPM_0:.*]] = memref.reinterpret_cast %[[SPM_B]] to offset: [%[[ZERO]]], sizes: [64], strides: [1]
// CHECK: bosc_ame.dma.start %[[DDR_0]], %[[SPM_0]], %[[SIXTY_FOUR]]
// CHECK: bosc_ame.dma.wait
// CHECK: %[[DDR_1:.*]] = memref.reinterpret_cast %[[B]] to offset: [%[[PITCH]]], sizes: [64], strides: [1]
// CHECK: %[[SPM_1:.*]] = memref.reinterpret_cast %[[SPM_B]] to offset: [%[[SIXTY_FOUR]]], sizes: [64], strides: [1]
// CHECK: bosc_ame.dma.start %[[DDR_1]], %[[SPM_1]], %[[SIXTY_FOUR]]
// CHECK: bosc_ame.dma.wait
// CHECK: %[[DDR_LAST:.*]] = memref.reinterpret_cast %[[B]] to offset: [%[[LAST_DDR]]], sizes: [64], strides: [1]
// CHECK: %[[SPM_LAST:.*]] = memref.reinterpret_cast %[[SPM_B]] to offset: [%[[LAST_SPM]]], sizes: [64], strides: [1]
// CHECK: bosc_ame.dma.start %[[DDR_LAST]], %[[SPM_LAST]], %[[SIXTY_FOUR]]
// CHECK: bosc_ame.dma.wait
// CHECK: %[[PACKED_B:.*]] = memref.reinterpret_cast %[[SPM_B]] to offset: [0], sizes: [64, 32], strides: [1, 64]
// CHECK: bosc_ame.tile.load_b %[[PACKED_B]] : memref<64x32xi8, strided<[1, 64]>, #bosc_ame.spm<1>>
// CHECK: bosc_ame.tile.mma
// CHECK: return

// CHECK-LABEL: func.func @static_offset
// CHECK-SAME: %[[INPUT:[[:alnum:]_]+]]: memref<64xi8, strided<[1], offset: 64>>,
// CHECK: %[[OFFSET:.*]] = arith.constant 64 : index
// CHECK: %[[DDR:.*]] = memref.reinterpret_cast %[[INPUT]] to offset: [%[[OFFSET]]], sizes: [64], strides: [1]
// CHECK: %[[SPM:.*]] = memref.reinterpret_cast
// CHECK: bosc_ame.dma.start %[[DDR]], %[[SPM]], %[[OFFSET]]
// CHECK: bosc_ame.dma.wait
// CHECK: bosc_ame.dma.start {{.*}} : (memref<64xi8, {{.*}}#bosc_ame.spm<0>>, memref<64xi8, {{.*}}#bosc_ame.spm<1>>
// CHECK: bosc_ame.dma.wait
// CHECK: bosc_ame.dma.start
// CHECK: return

// COUNT-LABEL: func.func @pitched_b
// COUNT-COUNT-33: bosc_ame.dma.start
// COUNT-NOT: bosc_ame.dma.start
// COUNT: return
// COUNT-LABEL: func.func @static_offset
// COUNT-COUNT-3: bosc_ame.dma.start
// COUNT-NOT: bosc_ame.dma.start
// COUNT: return

// LOWERED-LABEL: llvm.func @pitched_b
// LOWERED: bosc_ame.intr.mlbe8.m
// LOWERED: llvm.inline_asm has_side_effects
// LOWERED-SAME: mqma.b.mm
// LOWERED: llvm.return
// LOWERED-LABEL: llvm.func @static_offset
// LOWERED: llvm.call @bosc_ame_dma_start
// LOWERED: llvm.return
