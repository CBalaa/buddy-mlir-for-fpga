// RUN: buddy-opt -verify-diagnostics %s

// Shape, element type and tile size rules of the tile operations.  Every
// function below is invalid in exactly one way, and the diagnostic names the
// reason.  Keep the tiles non-square: a dimension swap must not be able to
// hide behind an M = N = K = 64 problem.

// --- load_a: the memref and the returned matrix must describe the same tile.
func.func @load_a_shape_mismatch(%a: memref<8x16xi8, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{does not match the matrix shape}}
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<16x8xi8>
  return
}

// --- load_b: the memref must be the logical [K, N] of `vector<KxNxi8>`.
func.func @load_b_shape_mismatch(%b: memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>>) {
  // expected-error@+1 {{does not match the matrix shape}}
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>> -> vector<16x16xi8>
  return
}

// --- the matrix value must be a fixed 2-D vector with the real tile shape.
func.func @matrix_rank_one(%a: memref<8x16xi8, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{must be a 2-D vector}}
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<128xi8>
  return
}

func.func @matrix_scalable(%a: memref<8x16xi8, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{must be a fixed 2-D vector}}
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x[16]xi8>
  return
}

// --- element types: the first version implements signed i8 A/B and i32 C.
func.func @load_a_i16(%a: memref<8x16xi16, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{element type 'i16' is not supported for the A tile}}
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi16, #bosc_ame.spm<0>> -> vector<8x16xi16>
  return
}

func.func @load_acc_i8(%acc: memref<8x32xi8, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{element type 'i8' is not supported for the accumulator tile}}
  %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi8, #bosc_ame.spm<0>> -> vector<8x32xi8>
  return
}

func.func @store_acc_f32(%acc: memref<8x32xf32, #bosc_ame.spm<0>>, %value: vector<8x32xf32>) {
  // expected-error@+1 {{element type 'f32' is not supported for the accumulator tile}}
  bosc_ame.tile.store_acc %value, %acc : vector<8x32xf32>, memref<8x32xf32, #bosc_ame.spm<0>>
  return
}

// --- the AME tile capacity is 64 per dimension.
func.func @tile_too_large(%a: memref<128x64xi8, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{is outside the AME tile capacity [1, 64]}}
  %ta = bosc_ame.tile.load_a %a : memref<128x64xi8, #bosc_ame.spm<0>> -> vector<128x64xi8>
  return
}

func.func @mma_k_too_large(%a: memref<8x128xi8, #bosc_ame.spm<0>>,
                           %b: memref<128x32xi8, strided<[1, 128]>, #bosc_ame.spm<1>>) {
  // expected-error@+1 {{is outside the AME tile capacity [1, 64]}}
  %ta = bosc_ame.tile.load_a %a : memref<8x128xi8, #bosc_ame.spm<0>> -> vector<8x128xi8>
  return
}

// --- a dynamic tile shape cannot be compiled into the tile configuration.
func.func @dynamic_shape(%a: memref<?x16xi8, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{must have a static shape}}
  %ta = bosc_ame.tile.load_a %a : memref<?x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  return
}

// --- layouts the hardware load cannot address.
func.func @load_a_k_not_contiguous(%a: memref<8x16xi8, strided<[1, 64]>, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{strides[1] == 1}}
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, strided<[1, 64]>, #bosc_ame.spm<0>> -> vector<8x16xi8>
  return
}

func.func @load_a_rows_overlap(%a: memref<8x16xi8, strided<[8, 1]>, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{strides[0] >= K}}
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, strided<[8, 1]>, #bosc_ame.spm<0>> -> vector<8x16xi8>
  return
}

// --- B is rejected unless it describes the physical [N, K] buffer, i.e. with
// --- K contiguous (strides[0] == 1).
func.func @load_b_row_major(%b: memref<16x32xi8, #bosc_ame.spm<1>>) {
  // expected-error@+1 {{strides[0] == 1}}
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, #bosc_ame.spm<1>> -> vector<16x32xi8>
  return
}

func.func @load_b_pitch_too_small(%b: memref<16x32xi8, strided<[1, 8]>, #bosc_ame.spm<1>>) {
  // expected-error@+1 {{strides[1] >= K}}
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 8]>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  return
}

// --- hasVerifier-style diagnostics for the accumulator memory.
func.func @load_acc_n_not_contiguous(%acc: memref<8x32xi32, strided<[1, 64]>, #bosc_ame.spm<0>>) {
  // expected-error@+1 {{strides[1] == 1}}
  %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, strided<[1, 64]>, #bosc_ame.spm<0>> -> vector<8x32xi32>
  return
}
