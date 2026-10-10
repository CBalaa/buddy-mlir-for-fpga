// RUN: buddy-opt %s --lower-bosc-ame | buddy-translate --buddy-to-llvmir | llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,-v,+xboscame -filetype=asm -o - | FileCheck %s --check-prefix=ASM

// The whole pipeline really compiles: `--lower-bosc-ame` produces LLVM dialect
// with the registered `llvm.riscv.bosc.fpga.*` intrinsics, `buddy-translate`
// turns it into LLVM IR and `llc` selects the AME instructions.  A check that
// only inspects the LLVM dialect cannot tell whether the backend knows these
// intrinsics or whether the fixed slots survive register allocation.
//
// The `+xboscame-fpga` convention is *not* passed to llc on the command line:
// the pass writes it onto the lowered function, and the backend then allocates
// the fixed slots from the load intrinsics.

// ASM-LABEL: {{^}}gemm:
// ASM: msettype
// ASM: mlce32.m{{[ \t]+}}acc0
// ASM: mlae8.m{{[ \t]+}}tr0
// ASM: mlbe8.m{{[ \t]+}}tr4
// ASM: mqma.b.mm{{[ \t]+}}acc0, tr0, tr4
// ASM: msce32.m{{[ \t]+}}acc0

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @gemm(%dummy: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %a_buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %b_buf = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<1>>
    %c_buf = bosc_ame.spm.view {bank = 0 : i32, offset = 8192 : i64} : memref<16384xi8, #bosc_ame.spm<0>>
    %a = memref.view %a_buf[%c0][] : memref<4096xi8, #bosc_ame.spm<0>> to memref<64x64xi8, #bosc_ame.spm<0>>
    // The physical [N, K] B buffer, described as the logical [K, N].
    %b = memref.reinterpret_cast %b_buf to offset: [0], sizes: [64, 64], strides: [1, 64] : memref<4096xi8, #bosc_ame.spm<1>> to memref<64x64xi8, strided<[1, 64], offset: 0>, #bosc_ame.spm<1>>
    %c = memref.view %c_buf[%c0][] : memref<16384xi8, #bosc_ame.spm<0>> to memref<64x64xi32, #bosc_ame.spm<0>>
    %acc0 = bosc_ame.tile.load_acc %c : memref<64x64xi32, #bosc_ame.spm<0>> -> vector<64x64xi32>
    %ta = bosc_ame.tile.load_a %a : memref<64x64xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    %tb = bosc_ame.tile.load_b %b : memref<64x64xi8, strided<[1, 64], offset: 0>, #bosc_ame.spm<1>> -> vector<64x64xi8>
    %acc = bosc_ame.tile.mma %acc0, %ta, %tb : vector<64x64xi32>, vector<64x64xi8>, vector<64x64xi8> -> vector<64x64xi32>
    bosc_ame.tile.store_acc %acc, %c : vector<64x64xi32>, memref<64x64xi32, #bosc_ame.spm<0>>
    return %c0_i32 : i32
  }
}
