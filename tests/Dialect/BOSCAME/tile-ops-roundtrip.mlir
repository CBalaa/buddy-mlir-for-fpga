// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// Round trip of the five `bosc_ame.tile.*` operations and of the layouts they
// accept.  The tiles are deliberately non-square (M=8, K=16, N=32) so that a
// swapped dimension or a hidden 64x64x64 assumption fails here instead of on
// the board.

// CHECK-LABEL: func.func @roundtrip
func.func @roundtrip(%a: memref<8x16xi8, #bosc_ame.spm<0>>,
                     %b: memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>>,
                     %acc: memref<8x32xi32, #bosc_ame.spm<0>>) -> vector<8x32xi32> {
  // CHECK: bosc_ame.tile.load_a %{{.*}} : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  // CHECK: bosc_ame.tile.load_b %{{.*}} : memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  // CHECK: bosc_ame.tile.load_acc %{{.*}} : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
  %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
  // CHECK: bosc_ame.tile.mma %{{.*}}, %{{.*}}, %{{.*}} : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
  %c1 = bosc_ame.tile.mma %tc, %ta, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
  // CHECK: bosc_ame.tile.store_acc %{{.*}}, %{{.*}} : vector<8x32xi32>, memref<8x32xi32, #bosc_ame.spm<0>>
  bosc_ame.tile.store_acc %c1, %acc : vector<8x32xi32>, memref<8x32xi32, #bosc_ame.spm<0>>
  return %c1 : vector<8x32xi32>
}

// A pitched A tile (rows further apart than K) and a pitched accumulator are
// legal: the AME load/store takes an explicit row stride.  The B operand keeps
// its K-contiguous (physical NxK) layout.
// CHECK-LABEL: func.func @pitched_layouts
func.func @pitched_layouts(%a: memref<8x16xi8, strided<[64, 1], offset: 0>, #bosc_ame.spm<0>>,
                           %b: memref<16x32xi8, strided<[1, 64], offset: 0>, #bosc_ame.spm<1>>,
                           %acc: memref<8x32xi32, strided<[128, 1], offset: 0>, #bosc_ame.spm<0>>) {
  // CHECK: bosc_ame.tile.load_a
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, strided<[64, 1], offset: 0>, #bosc_ame.spm<0>> -> vector<8x16xi8>
  // CHECK: bosc_ame.tile.load_b
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 64], offset: 0>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  // CHECK: bosc_ame.tile.load_acc
  %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, strided<[128, 1], offset: 0>, #bosc_ame.spm<0>> -> vector<8x32xi32>
  // CHECK: bosc_ame.tile.mma
  %c1 = bosc_ame.tile.mma %tc, %ta, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
  // CHECK: bosc_ame.tile.store_acc
  bosc_ame.tile.store_acc %c1, %acc : vector<8x32xi32>, memref<8x32xi32, strided<[128, 1], offset: 0>, #bosc_ame.spm<0>>
  return
}

// A chain of MMAs carries the accumulator through SSA: every step returns the
// updated `M x N` value with the same type as its accumulator operand.
// CHECK-LABEL: func.func @accumulator_chain
func.func @accumulator_chain(%a0: memref<8x16xi8, #bosc_ame.spm<0>>,
                             %a1: memref<8x16xi8, #bosc_ame.spm<0>>,
                             %b0: memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>>,
                             %b1: memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>>,
                             %acc: memref<8x32xi32, #bosc_ame.spm<0>>) -> vector<8x32xi32> {
  %ta0 = bosc_ame.tile.load_a %a0 : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  %tb0 = bosc_ame.tile.load_b %b0 : memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
  %c1 = bosc_ame.tile.mma %tc, %ta0, %tb0 : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
  %ta1 = bosc_ame.tile.load_a %a1 : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  %tb1 = bosc_ame.tile.load_b %b1 : memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  // CHECK: bosc_ame.tile.mma %{{.*}}, %{{.*}}, %{{.*}} : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
  %c2 = bosc_ame.tile.mma %c1, %ta1, %tb1 : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
  return %c2 : vector<8x32xi32>
}
