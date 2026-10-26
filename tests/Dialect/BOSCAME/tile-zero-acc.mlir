// RUN: buddy-opt %s --split-input-file --verify-diagnostics

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @zero(%out: memref<64x64xi32, #bosc_ame.spm<0>>) {
    %z = bosc_ame.tile.zero_acc : vector<64x64xi32>
    bosc_ame.tile.store_acc %z, %out : vector<64x64xi32>, memref<64x64xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

func.func @wrong_element() {
  // expected-error @+1 {{matrix element type 'i16' is not supported for the accumulator tile}}
  %z = bosc_ame.tile.zero_acc : vector<8x8xi16>
  return
}

// -----

func.func @wrong_rank() {
  // expected-error @+1 {{must be a 2-D vector with the real logical tile shape, got rank 1}}
  %z = bosc_ame.tile.zero_acc : vector<8xi32>
  return
}

// -----

func.func @oversized() {
  // expected-error @+1 {{matrix tile dimension 65}}
  %z = bosc_ame.tile.zero_acc : vector<65x8xi32>
  return
}

// -----

func.func @scalable() {
  // expected-error @+1 {{must be a fixed 2-D vector at this level}}
  %z = bosc_ame.tile.zero_acc : vector<[8]x8xi32>
  return
}
