// RUN: buddy-opt %s --lower-bosc-ame --verify-diagnostics --split-input-file

// `--lower-bosc-ame` runs the SPM/DMA contract checks and the SPM-value
// rules itself, before any type information is converted.  An SPM value that
// has no static address, a bank-changing cast or a broken DMA request lifetime
// must all be rejected with an actionable diagnostic instead of being lowered
// to a wrong address.

// An SPM memref argument has no statically known position inside its bank: the
// bank base is a fixed hardware address and SPM is addressed through
// `bosc_ame.spm.view`, not through the ABI.
// expected-error@+1 {{takes an SPM memref argument}}
func.func @spm_argument(%spm: memref<131072xi8, #bosc_ame.spm<0>>) {
  return
}

// -----

// A value produced by an operation that is not a view of an existing SPM
// buffer has no SPM address.
func.func private @make_spm() -> memref<64xi8, #bosc_ame.spm<0>>

func.func @spm_from_call() {
  // expected-error@+1 {{produces an SPM-typed value, but only `bosc_ame.spm.view`}}
  %v = call @make_spm() : () -> memref<64xi8, #bosc_ame.spm<0>>
  return
}

// -----

// The bank is part of the storage space, not something a cast may change.
func.func @bank_cast() {
  %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
  // expected-error@+1 {{cannot reinterpret an SPM buffer in bank 0 as bank 1}}
  %c = memref.memory_space_cast %v : memref<64xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<1>>
  return
}

// -----

// SPM is a fixed workspace: it is never allocated.
func.func @spm_alloc() {
  // expected-error@+1 {{allocates SPM bank 0}}
  %a = memref.alloc() : memref<64xi8, #bosc_ame.spm<0>>
  return
}

// -----

// A token is the handle of exactly one request, so it is consumed by exactly
// one wait.  The check names the request whose token escapes.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @two_waits(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{produces a DMA token that is not used by exactly one `bosc_ame.dma.wait`}}
    %t = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

// -----

// A dropped request stays in flight and leaves the channel busy.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @never_waited(%ddr: memref<64xi8>) {
    %v = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<0>>
    %c = arith.constant 64 : index
    // expected-error@+1 {{produces a DMA token that is never waited}}
    %t = bosc_ame.dma.start %ddr, %v, %c {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    return
  }
}

// -----

// A token is a local request handle and may not cross a function boundary.
// expected-error@+1 {{takes a DMA token as an argument}}
func.func @foreign_token(%t: !bosc_ame.dma_token<0>) {
  bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
  return
}
