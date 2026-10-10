// RUN: buddy-opt -verify-diagnostics %s --lower-bosc-ame-tile

// The fixed register slots.
//
// All A values share `tr0`, all B values share `tr4` and all accumulators share
// `acc0`, so a program that needs two matrix values of the same role at the
// same time cannot be allocated; it is rejected instead of letting one value
// silently overwrite the other.  The functions are ordered so that the valid
// program is checked before the rejected one (`--lower-bosc-ame-tile` stops at
// the first failure).

module attributes {bosc_ame.target = "nr-fpga"} {
  // Valid: the second A tile is loaded only after the first one has been
  // consumed by the MMA, so the two live ranges do not overlap in `tr0`.  The
  // accumulator chain shares one slot by construction (a `tile.mma` writes its
  // accumulator operand in place).
  func.func @sequential_tiles() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %a0 = memref.reinterpret_cast %buf to offset: [0], sizes: [8, 16], strides: [16, 1] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>>
    %a1 = memref.reinterpret_cast %buf to offset: [128], sizes: [8, 16], strides: [16, 1] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x16xi8, strided<[16, 1], offset: 128>, #bosc_ame.spm<0>>
    %b_buf = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<512xi8, #bosc_ame.spm<1>>
    %b = memref.reinterpret_cast %b_buf to offset: [0], sizes: [16, 32], strides: [1, 16] : memref<512xi8, #bosc_ame.spm<1>> to memref<16x32xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>>
    %acc = memref.view %buf[%c0][] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x32xi32, #bosc_ame.spm<0>>

    %ta0 = bosc_ame.tile.load_a %a0 : memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>> -> vector<8x16xi8>
    %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>> -> vector<16x32xi8>
    %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
    %c1 = bosc_ame.tile.mma %tc, %ta0, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
    %ta1 = bosc_ame.tile.load_a %a1 : memref<8x16xi8, strided<[16, 1], offset: 128>, #bosc_ame.spm<0>> -> vector<8x16xi8>
    %c2 = bosc_ame.tile.mma %c1, %ta1, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
    bosc_ame.tile.store_acc %c2, %acc : vector<8x32xi32>, memref<8x32xi32, #bosc_ame.spm<0>>
    return %c0_i32 : i32
  }

  // Rejected: both A tiles are live when the second one is loaded, and the
  // first version has exactly one A slot (`tr0`).
  func.func @two_live_a_tiles() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %a0 = memref.reinterpret_cast %buf to offset: [0], sizes: [8, 16], strides: [16, 1] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>>
    %a1 = memref.reinterpret_cast %buf to offset: [128], sizes: [8, 16], strides: [16, 1] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x16xi8, strided<[16, 1], offset: 128>, #bosc_ame.spm<0>>
    %b_buf = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<512xi8, #bosc_ame.spm<1>>
    %b = memref.reinterpret_cast %b_buf to offset: [0], sizes: [16, 32], strides: [1, 16] : memref<512xi8, #bosc_ame.spm<1>> to memref<16x32xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>>
    %acc = memref.view %buf[%c0][] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x32xi32, #bosc_ame.spm<0>>

    %ta0 = bosc_ame.tile.load_a %a0 : memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>> -> vector<8x16xi8>
    // expected-error@+1 {{two matrix values of the same role may not be live at the same time}}
    %ta1 = bosc_ame.tile.load_a %a1 : memref<8x16xi8, strided<[16, 1], offset: 128>, #bosc_ame.spm<0>> -> vector<8x16xi8>
    %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>> -> vector<16x32xi8>
    %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
    %c1 = bosc_ame.tile.mma %tc, %ta0, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
    %c2 = bosc_ame.tile.mma %c1, %ta1, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
    bosc_ame.tile.store_acc %c2, %acc : vector<8x32xi32>, memref<8x32xi32, #bosc_ame.spm<0>>
    return %c0_i32 : i32
  }
}
