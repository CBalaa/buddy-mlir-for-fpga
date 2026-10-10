// RUN: buddy-opt %s --bosc-ame-spm-dma-check -o /dev/null
// RUN: buddy-opt %s --bosc-ame-spm-dma-check --canonicalize --cse --bosc-ame-spm-dma-check | FileCheck %s --check-prefix=STABLE

// The check pass accepts a program that respects channel occupancy and buffer
// lifetime: the source of an in-flight request stays readable, a region proven
// disjoint from the in-flight destination stays usable, both channels may be
// active at once, and a bank can be reused after its request was waited.
// `bosc_ame.dma.wait` has no result and checks its own status, so nothing about
// status use appears here.
//
// The second run also pins the interaction with the generic cleanup passes: a
// DMA request is a real side effect, so neither `--canonicalize` nor `--cse`
// may drop a start, a wait or a token, and the check pass must accept the
// optimised program again.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @dual_channel(%ddr0: memref<4096xi8>, %ddr1: memref<4096xi8>,
                          %spm1: memref<131072xi8, #bosc_ame.spm<1>>) {
    %i = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %v0 = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %v0b = bosc_ame.spm.view {bank = 0 : i32, offset = 4096 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %v1 = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<1>>
    %t0 = bosc_ame.dma.start %ddr0, %v0, %c4096 {channel = 0 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // Reading the source of the in-flight request is allowed.
    %xs = memref.load %ddr0[%i] : memref<4096xi8>
    // Reading an SPM region proven disjoint from the in-flight destination is allowed.
    %xd = memref.load %v0b[%i] : memref<4096xi8, #bosc_ame.spm<0>>
    // A second channel may stage the same source into another bank.
    %t1 = bosc_ame.dma.start %ddr0, %v1, %c4096 {channel = 1 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t1 : !bosc_ame.dma_token<1>
    // A normal wait return means the transfer completed and succeeded, so the
    // destinations are readable immediately.
    %x0 = memref.load %v0[%i] : memref<4096xi8, #bosc_ame.spm<0>>
    %x1 = memref.load %v1[%i] : memref<4096xi8, #bosc_ame.spm<1>>
    // The same bank can be reused once its request has been waited.
    %t2 = bosc_ame.dma.start %ddr1, %v0, %c4096 {channel = 0 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t2 : !bosc_ame.dma_token<0>
    memref.store %xs, %v0b[%i] : memref<4096xi8, #bosc_ame.spm<0>>
    return
  }

  // A module without DMA has nothing to check.
  func.func @no_dma(%m: memref<64xi8>) {
    %i = arith.constant 0 : index
    %x = memref.load %m[%i] : memref<64xi8>
    return
  }
}
// STABLE-LABEL: func.func @dual_channel
// STABLE: bosc_ame.spm.view
// STABLE: bosc_ame.dma.start
// STABLE: bosc_ame.dma.start
// STABLE: bosc_ame.dma.wait
// STABLE: bosc_ame.dma.wait
// STABLE: bosc_ame.dma.start
// STABLE: bosc_ame.dma.wait
// STABLE-LABEL: func.func @no_dma
