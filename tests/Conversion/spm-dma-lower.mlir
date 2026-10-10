// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s
// RUN: buddy-opt %s --bosc-ame-spm-dma-check --lower-bosc-ame | FileCheck %s
// RUN: buddy-opt %s --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM

// `--lower-bosc-ame` is the single entry point for the AME / SPM / DMA surface.
// It runs the SPM/DMA contract checks itself, compiles `bosc_ame.spm.view` to
// the fixed bank base address, lowers the DMA requests to the runtime ABI and
// finishes the module with the standard LLVM conversion.
//
// There is no SPM workspace root argument: the bank base address is a
// compile-time hardware constant, so the kernel ABI only carries the DDR
// buffers and the generated function has no SPM parameter.
//
// `bosc_ame.dma.wait` has no result: the runtime performs the completion
// barrier and checks the status, and a normal return already means "completed
// and succeeded", so the destination can be consumed immediately.

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @stage(%ddr0: memref<8192xi8>, %ddr1: memref<8192xi8>) {
    %c4096 = arith.constant 4096 : index
    %c1024 = arith.constant 1024 : index
    %i = arith.constant 0 : index
    %v0 = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %t0 = bosc_ame.dma.start %ddr0, %v0, %c4096 {channel = 1 : i32} : (memref<8192xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<1>
    %x = memref.load %v0[%i] : memref<4096xi8, #bosc_ame.spm<0>>
    memref.store %x, %ddr1[%i] : memref<8192xi8>
    %v1 = bosc_ame.spm.view {bank = 1 : i32, offset = 8192 : i64} : memref<4096xi8, #bosc_ame.spm<1>>
    %sv = memref.subview %v1[512] [1024] [1] : memref<4096xi8, #bosc_ame.spm<1>> to memref<1024xi8, strided<[1], offset: 512>, #bosc_ame.spm<1>>
    %t1 = bosc_ame.dma.start %sv, %ddr1, %c1024 {channel = 0 : i32} : (memref<1024xi8, strided<[1], offset: 512>, #bosc_ame.spm<1>>, memref<8192xi8>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t1 : !bosc_ame.dma_token<0>
    %y = memref.load %sv[%i] : memref<1024xi8, strided<[1], offset: 512>, #bosc_ame.spm<1>>
    memref.store %y, %ddr0[%i] : memref<8192xi8>
    return
  }
}

// The function is an `llvm.func`; no SPM memref survives anywhere.
// CHECK-LABEL: llvm.func @stage(
// CHECK-NOT: memref
// The bank-0 view at byte offset 64 is the fixed address 0x20000000 + 64.
// CHECK: %[[B0:[a-zA-Z0-9_]+]] = llvm.mlir.constant(536870976 : i64)
// CHECK: %[[P0:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[B0]] : i64 to !llvm.ptr
// The token is the start call's `i64` result; the wait consumes it and returns
// nothing, because it checks the status itself.
// CHECK: %[[T0:[a-zA-Z0-9_]+]] = llvm.call @bosc_ame_dma_start(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}) : (i32, i64, i64, i64) -> i64
// CHECK: llvm.call @bosc_ame_dma_wait(%{{[a-zA-Z0-9_]+}}, %[[T0]]) : (i32, i64) -> ()
// The staged byte is read back through a pointer derived from the descriptor.
// CHECK: llvm.load
// The bank-1 view at byte offset 8192 plus the 512-byte subview is the fixed
// address 0x20400000 + 8192; the subview offset is folded in exactly once.
// CHECK: llvm.mlir.constant(541073408 : i64)
// CHECK: %[[T1:[a-zA-Z0-9_]+]] = llvm.call @bosc_ame_dma_start(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}) : (i32, i64, i64, i64) -> i64
// CHECK: llvm.call @bosc_ame_dma_wait(%{{[a-zA-Z0-9_]+}}, %[[T1]]) : (i32, i64) -> ()
// CHECK: llvm.load
// CHECK-NOT: bosc_ame.spm
// CHECK-NOT: bosc_ame.dma_token
// CHECK-NOT: bosc_ame.spm.view
// CHECK-NOT: bosc_ame.dma.start
// CHECK-NOT: bosc_ame.dma.wait
// CHECK-NOT: func.func

// The runtime entry points are only declared; the C runtime provides them.
// LLVM: declare i64 @bosc_ame_dma_start(i32, i64, i64, i64)
// LLVM: declare void @bosc_ame_dma_wait(i32, i64)
// LLVM-LABEL: define void @stage(
// The channel is an i32 immediate, the SPM destination is the constant bank
// address folded by the compiler.
// LLVM: call i64 @bosc_ame_dma_start(i32 1, i64 %{{[0-9]+}}, i64 536870976, i64 4096)
// LLVM: call void @bosc_ame_dma_wait(i32 1, i64 %{{[0-9]+}})
// LLVM: call i64 @bosc_ame_dma_start(i32 0, i64 541073920, i64 %{{[0-9]+}}, i64 1024)
// LLVM: call void @bosc_ame_dma_wait(i32 0, i64 %{{[0-9]+}})
