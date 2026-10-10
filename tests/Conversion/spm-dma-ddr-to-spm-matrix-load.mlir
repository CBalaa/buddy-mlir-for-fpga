// RUN: buddy-opt %s --bosc-ame-spm-dma-check -o /dev/null
// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s --check-prefix=PREP
// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s --check-prefix=ABSENT
// RUN: buddy-opt %s --lower-bosc-ame | buddy-translate --buddy-to-llvmir \
// RUN:   | FileCheck %s --check-prefix=LLVM

// The smallest kernel of the first version that exercises the whole feature end
// to end: DDR bytes are staged into an SPM bank by one DMA engine and the
// staged bytes are then consumed by the *existing* matrix load.  `--lower-bosc-ame`
// owns the whole path: the `spm` memory space is compiled into the fixed bank
// address, the DMA request becomes a runtime call, and the matrix load reads the
// staged bytes through the descriptor built from that same address.
//
// The completion protocol is the shipped one, but it lives in the runtime now:
// `bosc_ame.dma.wait` has no result, performs the SRC write-back barrier and
// checks STATUS itself.  A normal return means the transfer completed and
// succeeded, so the matrix load follows directly and a bus error can never be
// consumed as if it were data (the runtime takes its error path instead).
//
// The negative counterpart lives in `spm-dma-check-invalid.mlir`: consuming the
// destination while the request is still in flight is rejected for this same
// matrix load.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @stage_then_load(%ddr: memref<8192xi8>,
                             %out: memref<64x64xi8>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : i64
    %view = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %token = bosc_ame.dma.start %ddr, %view, %c4096 {channel = 0 : i32} : (memref<8192xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %token : !bosc_ame.dma_token<0>
    %tile = bosc_ame.mlae8.m %view, %c64 : memref<4096xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    bosc_ame.msae8.m %tile, %out, %c64 : vector<64x64xi8>, memref<64x64xi8>
    return %c0_i32 : i32
  }
}

// The runtime ABI appears as two declarations plus one call each, and the SPM
// view is the fixed bank address 0x20000000 + 64.  The matrix load consumes the
// descriptor derived from that same address.
// PREP: llvm.func @bosc_ame_dma_start(i32, i64, i64, i64) -> i64
// PREP: llvm.func @bosc_ame_dma_wait(i32, i64)
// PREP: llvm.mlir.constant(536870976 : i64)
// PREP: llvm.call @bosc_ame_dma_start
// PREP: llvm.call @bosc_ame_dma_wait
// PREP: bosc_ame.intr.mlae8.m

// ABSENT-NOT: bosc_ame.spm.view
// ABSENT-NOT: bosc_ame.dma.start
// ABSENT-NOT: bosc_ame.dma.wait
// ABSENT-NOT: dma_token
// ABSENT-NOT: #bosc_ame.spm

// The DMA reaches the runtime, the AME load reads the staged bytes, and the
// wait call has no result: the status check lives inside the runtime.
// LLVM: declare i64 @bosc_ame_dma_start(i32, i64, i64, i64)
// LLVM: declare void @bosc_ame_dma_wait(i32, i64)
// LLVM: [[TOKEN:%.*]] = call i64 @bosc_ame_dma_start(i32 0, i64 %[[SRC:[0-9]+]], i64 536870976, i64 4096)
// LLVM: call void @bosc_ame_dma_wait(i32 0, i64 [[TOKEN]])
// LLVM: call <vscale x 128 x i8> @llvm.riscv.bosc.mlae8.m.nxv128i8.i64(ptr
// LLVM: call void @llvm.riscv.bosc.msae8.m.nxv128i8.i64(