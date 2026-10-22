// RUN: split-file %s %t
// RUN: buddy-opt %t/valid.mlir | FileCheck %s --check-prefix=VALID
// RUN: buddy-opt -verify-diagnostics %t/missing-abi.mlir
// RUN: buddy-opt -verify-diagnostics %t/internal-root.mlir
// RUN: buddy-opt -verify-diagnostics %t/spm-root.mlir
// RUN: buddy-opt -verify-diagnostics %t/overflow-row-stride.mlir

// Direct tile memory endpoints must have both an external DDR provenance and
// the module ABI promise used to establish alignment and argument disjointness.

//--- valid.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @direct(%a: memref<8x64xi8>, %c: memref<8x32xi32>) {
    %a_view = memref.subview %a[0, 0] [8, 64] [1, 1]
      : memref<8x64xi8> to memref<8x64xi8, strided<[64, 1]>>
    %ta = bosc_ame.tile.load_a %a_view {bosc_ame.direct_ddr}
      : memref<8x64xi8, strided<[64, 1]>> -> vector<8x64xi8>
    %tc = bosc_ame.tile.load_acc %c {bosc_ame.direct_ddr}
      : memref<8x32xi32> -> vector<8x32xi32>
    bosc_ame.tile.store_acc %tc, %c {bosc_ame.direct_ddr}
      : vector<8x32xi32>, memref<8x32xi32>
    return
  }
}

// VALID: bosc_ame.tile.load_a {{.*}} {bosc_ame.direct_ddr}
// VALID: bosc_ame.tile.load_acc {{.*}} {bosc_ame.direct_ddr}
// VALID: bosc_ame.tile.store_acc {{.*}} {bosc_ame.direct_ddr}

//--- missing-abi.mlir
module {
  func.func @missing_abi(%a: memref<8x64xi8>) {
    // expected-error@+1 {{direct DDR tile endpoint requires module attribute `bosc_ame.dma.abi = "args_aligned64_nonalias"`}}
    %ta = bosc_ame.tile.load_a %a {bosc_ame.direct_ddr}
      : memref<8x64xi8> -> vector<8x64xi8>
    return
  }
}

//--- internal-root.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @internal_root() {
    %a = memref.alloc() alignment = 64 : memref<8x64xi8>
    // expected-error@+1 {{direct DDR tile endpoint must trace to an external function entry argument}}
    %ta = bosc_ame.tile.load_a %a {bosc_ame.direct_ddr}
      : memref<8x64xi8> -> vector<8x64xi8>
    return
  }
}

//--- spm-root.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @spm_root(%a: memref<8x64xi8, #bosc_ame.spm<0>>) {
    // expected-error@+1 {{direct DDR tile endpoint must trace to an external function entry argument}}
    %ta = bosc_ame.tile.load_a %a {bosc_ame.direct_ddr}
      : memref<8x64xi8, #bosc_ame.spm<0>> -> vector<8x64xi8>
    return
  }
}

// A static element stride can fit in an index while its byte conversion does
// not.  Reject it before the direct-DDR alignment checks see a wrapped value.
//--- overflow-row-stride.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @overflow_row_stride(
      %c: memref<1x1xi32, strided<[9223372036854775807, 1], offset: 0>>) {
    // expected-error@+1 {{has a layout the AME accumulator load/store cannot address}}
    %tc = bosc_ame.tile.load_acc %c {bosc_ame.direct_ddr}
      : memref<1x1xi32, strided<[9223372036854775807, 1], offset: 0>>
        -> vector<1x1xi32>
    return
  }
}
