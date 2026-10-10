// RUN: buddy-opt -verify-diagnostics %s

// Shape and type rules of `bosc_ame.tile.mma`: A is M x K, B is K x N, C is
// M x N, the inner dimensions must agree, and C_in must have exactly the type
// of C_out.  The tiles are non-square so that a swapped M/N/K is rejected
// instead of accidentally matching.

func.func @k_mismatch(%a: memref<8x16xi8, #bosc_ame.spm<0>>,
                      %b: memref<32x32xi8, strided<[1, 32]>, #bosc_ame.spm<1>>,
                      %c: memref<8x32xi32, #bosc_ame.spm<0>>) {
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  %tb = bosc_ame.tile.load_b %b : memref<32x32xi8, strided<[1, 32]>, #bosc_ame.spm<1>> -> vector<32x32xi8>
  %tc = bosc_ame.tile.load_acc %c : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
  // expected-error@+1 {{inner dimension of A (K = 16) and the outer dimension of B (K = 32) must agree}}
  %r = bosc_ame.tile.mma %tc, %ta, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<32x32xi8> -> vector<8x32xi32>
  return
}

func.func @c_shape_mismatch(%a: memref<8x16xi8, #bosc_ame.spm<0>>,
                            %b: memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>>,
                            %c: memref<16x32xi32, #bosc_ame.spm<0>>) {
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  %tc = bosc_ame.tile.load_acc %c : memref<16x32xi32, #bosc_ame.spm<0>> -> vector<16x32xi32>
  // expected-error@+1 {{C must be an M x N tile with M = 8 (from A) and N = 32 (from B)}}
  %r = bosc_ame.tile.mma %tc, %ta, %tb : vector<16x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<16x32xi32>
  return
}

// The accumulator dependency is explicit in SSA: C_in and C_out must have the
// same type, so a chain of MMAs cannot silently change tile shape or element
// type.
func.func @acc_out_type_mismatch(%a: memref<8x16xi8, #bosc_ame.spm<0>>,
                                 %b: memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>>,
                                 %c: memref<8x32xi32, #bosc_ame.spm<0>>) {
  %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, #bosc_ame.spm<0>> -> vector<8x16xi8>
  %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 16]>, #bosc_ame.spm<1>> -> vector<16x32xi8>
  %tc = bosc_ame.tile.load_acc %c : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
  // expected-error@+1 {{must be identical}}
  %r = bosc_ame.tile.mma %tc, %ta, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<16x32xi32>
  return
}

// The MMA datapath is signed i8 x i8 -> i32; a wider input has no verified
// instruction, register mapping and mtype phase in the first version.  The
// matrix values come in as arguments here so that the MMA check itself is the
// one that fires (an `i16` tile load is rejected by the load verifier).
func.func @lhs_i16(%ta: vector<8x16xi16>, %tb: vector<16x32xi16>,
                   %tc: vector<8x32xi32>) -> vector<8x32xi32> {
  // expected-error@+1 {{element type 'i16' is not supported for the A tile}}
  %r = bosc_ame.tile.mma %tc, %ta, %tb : vector<8x32xi32>, vector<8x16xi16>, vector<16x32xi16> -> vector<8x32xi32>
  return %r : vector<8x32xi32>
}

// An i8 accumulator would need a different accumulator datapath; the first
// version has none, and reading it as i32 would be a reinterpretation.
func.func @acc_i8(%ta: vector<8x16xi8>, %tb: vector<16x32xi8>,
                  %tc: vector<8x32xi8>) -> vector<8x32xi8> {
  // expected-error@+1 {{element type 'i8' is not supported for the accumulator tile}}
  %r = bosc_ame.tile.mma %tc, %ta, %tb : vector<8x32xi8>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi8>
  return %r : vector<8x32xi8>
}
