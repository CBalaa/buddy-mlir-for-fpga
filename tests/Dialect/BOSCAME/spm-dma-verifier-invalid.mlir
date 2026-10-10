// RUN: buddy-opt %s --verify-diagnostics --split-input-file

// Operation-local checks: the storage space named by the op must match its
// result type, the byte range must stay inside the bank, DMA endpoints must be
// static 1-D i8 memrefs, the transfer must be a positive multiple of the DMA
// granularity and fit both endpoints, the endpoints must be 64 B aligned and
// non-overlapping, and the request must be provably aligned/non-aliasing on the
// DDR side.  These are deliberately stricter than the storage-space rule: a
// static error here can never be repaired by scheduling.
func.func @bank_mismatch() {
  // expected-error@+1 {{bank 1 does not match the result storage space}}
  %v = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
  return
}
func.func @out_of_bank() {
  // expected-error@+1 {{byte range [131072, 131136) exceeds the 131072-byte SPM bank 0}}
  %v = bosc_ame.spm.view {bank = 0 : i32, offset = 131072 : i64} : memref<64xi8, #bosc_ame.spm<0>>
  return
}
func.func @bad_offset() {
  // expected-error@+1 {{offset must be non-negative}}
  %v = bosc_ame.spm.view {bank = 0 : i32, offset = -64 : i64} : memref<64xi8, #bosc_ame.spm<0>>
  return
}
func.func @ddr_to_ddr(%a: memref<64xi8>, %b: memref<64xi8>) {
  %c = arith.constant 64 : index
  // expected-error@+1 {{at least one endpoint must be SPM storage}}
  %t = bosc_ame.dma.start %a, %b, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8>, index) -> !bosc_ame.dma_token<0>
  return
}
func.func @status_not_token(%i: i64) {
  // expected-error@+1 {{must be Handle of an in-flight DMA request on one channel}}
  "bosc_ame.dma.wait"(%i) : (i64) -> ()
  return
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @dynamic_bytes(%d: memref<64xi8>, %n: index) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    // expected-error@+1 {{`bytes` must fold to a constant number of bytes}}
    %t = bosc_ame.dma.start %d, %v, %n {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
  func.func @bad_granularity(%d: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 96 : index
    // expected-error@+1 {{bytes must be a multiple of the 64-byte DMA granularity, got 96}}
    %t = bosc_ame.dma.start %d, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
  func.func @zero_bytes(%d: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 0 : index
    // expected-error@+1 {{bytes must be positive, got 0}}
    %t = bosc_ame.dma.start %d, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
  func.func @too_long(%d: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 128 : index
    // expected-error@+1 {{transfer of 128 bytes exceeds the src byte range of 64 bytes}}
    %t = bosc_ame.dma.start %d, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
  func.func @spm_unaligned(%d: memref<128xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<128xi8, #bosc_ame.spm<0>>
    %s = memref.subview %v[1] [64] [1] : memref<128xi8, #bosc_ame.spm<0>> to memref<64xi8, strided<[1], offset: 1>, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{SPM endpoint must be 64 B aligned, but its descriptor offset 1 is not}}
    %t = bosc_ame.dma.start %d, %s, %c {channel = 0 : i32} : (memref<128xi8>, memref<64xi8, strided<[1], offset: 1>, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
  func.func @overlap(%d: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<128xi8, #bosc_ame.spm<0>>
    %w = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<128xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{src range [0, 64) and dst range [0, 64) overlap in SPM bank 0}}
    %t = bosc_ame.dma.start %v, %w, %c {channel = 0 : i32} : (memref<128xi8, #bosc_ame.spm<0>>, memref<128xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
}
// -----

func.func @no_abi(%d: memref<64xi8>) {
  %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
  %c = arith.constant 64 : index
  // expected-error@+1 {{a DDR endpoint cannot be proven aligned and non-overlapping; annotate the module with `bosc_ame.dma.abi = "args_aligned64_nonalias"`}}
  %t = bosc_ame.dma.start %d, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
  return
}
// -----

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @token_mismatch(%d: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{result token channel 1 does not match the request channel 0}}
    %t = bosc_ame.dma.start %d, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<1>
    return
  }
}
