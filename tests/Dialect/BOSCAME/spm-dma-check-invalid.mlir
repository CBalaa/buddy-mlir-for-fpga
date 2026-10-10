// RUN: buddy-opt %s --bosc-ame-spm-dma-check --verify-diagnostics --split-input-file

// Every storage-space rule the pass enforces, one module per rule:
//
//  * at most one request per channel may be in flight (`err_channel_busy`);
//  * the source of an in-flight request may be read but not written
//    (`err_src_write`);
//  * the destination of an in-flight request may neither be written
//    (`err_dst_overlap`) nor read (`err_dst_read`);
//  * a request that is still in flight may not be reachable through an SPM
//    region (`err_region_active`);
//  * a request must be waited before the end of the block that started it
//    (`err_block_end`);
//  * no call with unknown memory effects may run while a request is in flight
//    (`err_call`);
//  * SPM storage may only come from `bosc_ame.spm.view`, not from an allocation
//    (`err_spm_alloc`);
//  * a token must be waited exactly once (`err_token_leak`,
//    `err_token_two_users`);
//  * the existing AME matrix load describes no memory effects of its own, so it
//    is conservatively treated as reading every buffer it is handed: it is
//    rejected while the request is in flight (`err_dst_read_ame`).
//
// Status handling is not checked here: `bosc_ame.dma.wait` has no result and
// the runtime performs the completion barrier and the status check itself.
//
// The layering of these checks is intentional: the local verifier rejects what
// is statically invalid in a single operation, this pass rejects what is
// invalid across operations, and the runtime ABI tests cover the protocol
// itself.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @busy(%ddr0: memref<64xi8>, %ddr1: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %w = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<1>>
    %c = arith.constant 64 : index
    %t0 = bosc_ame.dma.start %ddr0, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{while an earlier request on the same channel is still in flight}}
    %t1 = bosc_ame.dma.start %ddr1, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t1 : !bosc_ame.dma_token<0>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @src_write(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    %i = arith.constant 0 : index
    %x = arith.constant 0 : i8
    %t0 = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{which overlaps the source of the in-flight request on DMA channel 0}}
    memref.store %x, %ddr[%i] : memref<64xi8>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @dst_overlap(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %w = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<1>>
    %c = arith.constant 64 : index
    %t0 = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{that overlaps the in-flight request on DMA channel 0}}
    %t1 = bosc_ame.dma.start %w, %ddr, %c {channel = 1 : i32} : (memref<64xi8, #bosc_ame.spm<1>>, memref<64xi8>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t1 : !bosc_ame.dma_token<1>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @dst_read(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    %i = arith.constant 0 : index
    %t0 = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{which overlaps the destination of the in-flight request on DMA channel 0}}
    %x = memref.load %v[%i] : memref<64xi8, #bosc_ame.spm<0>>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @region_active(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    %true = arith.constant true
    %t0 = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{is reached while the request started on DMA channel 0 is still in flight}}
    scf.if %true {
      scf.yield
    }
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @block_end(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{never waited before the end of this block}}
    %t0 = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    cf.br ^next
  ^next:
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func private @callee(memref<64xi8>)
  func.func @call_while_active(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    %t0 = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{calls code with unknown memory effects while a DMA request is in flight}}
    call @callee(%ddr) : (memref<64xi8>) -> ()
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @spm_alloc() {
    // expected-error@+1 {{allocates SPM bank 0}}
    %a = memref.alloc() : memref<64xi8, #bosc_ame.spm<0>>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @leak(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{produces a DMA token that is never waited}}
    %t = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @two_waits(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{not used by exactly one `bosc_ame.dma.wait`}}
    %t = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}
// -----

// The existing matrix load is the consumer the first version targets, so it has
// to be subject to the in-flight occupancy rule like any other access.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @dst_read_ame(%ddr: memref<8192xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : i64
    %t0 = bosc_ame.dma.start %ddr, %v, %c4096 {channel = 0 : i32} : (memref<8192xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{which overlaps the destination of the in-flight request on DMA channel 0}}
    %tile = bosc_ame.mlae8.m %v, %c64 : memref<4096xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    return
  }
}
