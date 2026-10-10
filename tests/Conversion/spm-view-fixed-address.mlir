// RUN: buddy-opt %s --lower-bosc-ame --split-input-file | FileCheck %s

// `bosc_ame.spm.view` compiles the bank base address into the pointer, so the
// pass needs no SPM workspace root argument and no caller-provided base.  These
// cases pin the two fixed bases, a nested subview and a multi-byte element
// offset.  The last case pins that a DDR-only module gains no DMA runtime
// declaration.

// A bank-0 i32 view at byte offset 32: the fixed base is 0x20000000 + 32.  The
// subview adds 17 i32 elements, which the descriptor pointer computation scales
// by the element width (GEP with an i32 element type), never by 1.
// CHECK-LABEL: llvm.func @bank0_i32_view
// CHECK-NOT: memref
// CHECK: %[[BASE:.+]] = llvm.mlir.constant(536870944 : i64)
// CHECK: %[[PTR:.+]] = llvm.inttoptr %[[BASE]] : i64 to !llvm.ptr
// CHECK: llvm.mlir.constant(17 : i64)
// CHECK: llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: bosc_ame.intr.mlae32.m
func.func @bank0_i32_view(%out: memref<4x4xi32>, %stride: i64) {
  %v = bosc_ame.spm.view {bank = 0 : i32, offset = 32 : i64} : memref<16x16xi32, #bosc_ame.spm<0>>
  %sv = memref.subview %v[1, 1] [4, 4] [1, 1] : memref<16x16xi32, #bosc_ame.spm<0>> to memref<4x4xi32, strided<[16, 1], offset: 17>, #bosc_ame.spm<0>>
  %tile = bosc_ame.mlae32.m %sv, %stride : memref<4x4xi32, strided<[16, 1], offset: 17>, #bosc_ame.spm<0>> -> vector<4x4xi32>
  bosc_ame.msae32.m %tile, %out, %stride : vector<4x4xi32>, memref<4x4xi32>
  return
}

// -----

// The second bank is a different fixed window in the same address space:
// 0x20400000, not address space 1 and not a runtime argument.
// CHECK-LABEL: llvm.func @bank1_i8_view
// CHECK-NOT: memref
// CHECK: llvm.mlir.constant(541065216 : i64)
// CHECK: llvm.inttoptr
// CHECK: bosc_ame.intr.mlae8.m
func.func @bank1_i8_view(%out: memref<16x16xi8>, %stride: i64) {
  %v = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<16x16xi8, #bosc_ame.spm<1>>
  %tile = bosc_ame.mlae8.m %v, %stride : memref<16x16xi8, #bosc_ame.spm<1>> -> vector<16x16xi8>
  bosc_ame.msae8.m %tile, %out, %stride : vector<16x16xi8>, memref<16x16xi8>
  return
}

// -----

// A module that never mentions SPM or DMA must not gain the runtime ABI
// declarations; it is still lowered to LLVM by the same pass.
// CHECK-LABEL: llvm.func @no_spm_dma(
// CHECK: llvm.load
// CHECK-NOT: bosc_ame_dma_start
// CHECK-NOT: bosc_ame_dma_wait
func.func @no_spm_dma(%in: memref<64xi8>, %i: index) -> i8 {
  %0 = memref.load %in[%i] : memref<64xi8>
  return %0 : i8
}
