// RUN: buddy-opt -verify-diagnostics %s

// The tile operations only address SPM.  A plain (DDR) memref has no SPM
// address for the AME loads and stores, so every one of them is rejected -
// including the accumulator store.
func.func @ddr_load_a(%a: memref<8x16xi8>) {
  // expected-error@+1 {{must carry the `#bosc_ame.spm<bank>` storage space}}
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8> -> vector<8x16xi8>
  return
}

func.func @ddr_load_b(%b: memref<16x32xi8>) {
  // expected-error@+1 {{must carry the `#bosc_ame.spm<bank>` storage space}}
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8> -> vector<16x32xi8>
  return
}

func.func @ddr_load_acc(%acc: memref<8x32xi32>) {
  // expected-error@+1 {{must carry the `#bosc_ame.spm<bank>` storage space}}
  %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32> -> vector<8x32xi32>
  return
}

func.func @ddr_store_acc(%acc: memref<8x32xi32>, %value: vector<8x32xi32>) {
  // expected-error@+1 {{must carry the `#bosc_ame.spm<bank>` storage space}}
  bosc_ame.tile.store_acc %value, %acc : vector<8x32xi32>, memref<8x32xi32>
  return
}

// A rank-1 SPM buffer is not a tile: the first version is static 2-D only.
func.func @rank_one(%a: memref<4096xi8, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{must be a 2-D memref}}
  %ta = bosc_ame.tile.load_a %a : memref<4096xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  return
}

// An unranked memref is rejected by the ODS operand constraint before the
// verifier runs (the operand of every tile operation is `AnyMemRef`, i.e. a
// ranked memref), so it needs no case here.
