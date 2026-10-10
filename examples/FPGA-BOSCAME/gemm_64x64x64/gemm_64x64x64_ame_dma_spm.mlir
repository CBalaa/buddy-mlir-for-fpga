//===- gemm_64x64x64_ame_dma_spm.mlir - 64x64x64 INT8 GEMM via SPM + AME -===//
//
// A complete 64x64x64 signed INT8 => INT32 GEMM written directly in the
// BOSCAME dialect and lowered end to end by `buddy-opt --lower-bosc-ame`:
//
//   DDR A/B/Z --DMA0/DMA1--> SPM0/SPM1 --MLAE8/MLBE8--> Tr0/Tr4
//             --MQMA.B.MM--> Acc0 --MSCE32--> SPM0 --DMA1--> DDR C
//
// The module is an `nr-fpga` target, i.e. the validated W8A8 register-file
// convention: register-form `msettilem/msettilen/msettilek` and `msettype`,
// `tr0`/`tr4` for the A/B tile registers, `acc0` for the accumulator, and the
// explicit `bosc_ame.fpga.slot` attributes on the tile loads.
//
// What the pass does with each construct:
//
//   * `bosc_ame.spm.view` describes a byte range of a fixed SPM bank.  The
//     pass compiles `bankBase[bank] + offset` into the pointer, so the kernel
//     takes no SPM workspace argument: the ABI is just the DDR buffers.
//     bank 0 = 0x20000000, bank 1 = 0x20400000.
//   * `bosc_ame.dma.start` / `bosc_ame.dma.wait` stage the operands and the
//     zero accumulator tile from DDR into SPM and write the result back.  The
//     token becomes the runtime call's `i64` result (the effective source
//     address); the wait has no result and the runtime performs the completion
//     barrier and the status check itself, taking its error path on failure.
//     A normal wait return therefore means "completed and succeeded", and the
//     code continues straight into the AME sequence.
//   * `bosc_ame.mlae8.m` / `bosc_ame.mlbe8.m` load the SPM tiles into Tr0/Tr4,
//     `bosc_ame.mqma.b.mm` computes into Acc0, `bosc_ame.msce32.m` writes the
//     accumulator tile back into SPM.
//
// The output buffer is byte-oriented (`memref<16384xi8>`): the 64x64 INT32
// tile is 16384 bytes and the DMA engine moves bytes.  `memref.view` re-types
// the same SPM window as `64x64xi32` for the accumulator load/store, which is
// what the nr-fpga contract requires.  `llvm.emit_c_interface` gives the
// kernel the `_mlir_ciface_` entry point the C launcher calls.
//
// Memory layout in SPM (both banks are 128 KiB):
//
//   SPM0 + 0x0000 : A, 4096 bytes
//   SPM0 + 0x2000 : the 16384-byte INT32 accumulator tile
//   SPM1 + 0x0000 : B, 4096 bytes
//
// The module carries `bosc_ame.dma.abi = "args_aligned64_nonalias"`: the
// caller promises that every DDR pointer argument is 64-byte aligned and that
// distinct pointer arguments do not overlap.  That is what lets the compiler
// accept the DDR endpoints without being able to see their runtime addresses.
//===- ------------------------------------------------------------------===//

module attributes {bosc_ame.target = "nr-fpga",
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @gemm_64x64x64(%a: memref<4096xi8>, %b: memref<4096xi8>,
                           %z: memref<16384xi8>, %c: memref<16384xi8>) -> i32
      attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i64
    %c256 = arith.constant 256 : i64
    %c4096 = arith.constant 4096 : index
    %c16384 = arith.constant 16384 : index
    %c0_idx = arith.constant 0 : index
    %mtype_e8 = arith.constant 65552 : i64
    %mtype_e32 = arith.constant 65602 : i64

    // Fixed SPM windows: A in bank 0, B in bank 1, the accumulator tile in
    // bank 0.  Each view compiles to a constant physical address.
    %spm_a = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %spm_b = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<1>>
    %spm_c = bosc_ame.spm.view {bank = 0 : i32, offset = 8192 : i64} : memref<16384xi8, #bosc_ame.spm<0>>
    // The DMA contract is byte-oriented; the nr-fpga accumulator contract is
    // i32.  Both views describe the very same bank-0 window.
    %spm_acc = memref.view %spm_c[%c0_idx][] : memref<16384xi8, #bosc_ame.spm<0>> to memref<64x64xi32, #bosc_ame.spm<0>>

    // Stage A and B from DDR into the two SPM banks.  DMA0 and DMA1 are
    // independent engines, so both transfers overlap; both are 64-byte
    // aligned and 4096 bytes (= 64 AXI beats).  A normal wait return means the
    // transfer completed and succeeded, so the buffers are usable right away.
    %t0 = bosc_ame.dma.start %a, %spm_a, %c4096 {channel = 0 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    %t1 = bosc_ame.dma.start %b, %spm_b, %c4096 {channel = 1 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t1 : !bosc_ame.dma_token<1>

    // C = 0 for this pass: stage a permanent zero tile into the accumulator
    // region so `mlce32` seeds Acc0 with zeros instead of stale SPM bytes.
    %tz = bosc_ame.dma.start %z, %spm_c, %c16384 {channel = 0 : i32} : (memref<16384xi8>, memref<16384xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %tz : !bosc_ame.dma_token<0>

    // One hardware tile covers the whole problem: M = N = K = 64.
    %_m = bosc_ame.msettilem %c64 : i64
    %_n = bosc_ame.msettilen %c64 : i64
    %_k = bosc_ame.msettilek %c64 : i64

    // Zero the accumulator: load the zeroed tile, then read it back as Acc0.
    %_t32 = bosc_ame.msettype %mtype_e32 : i64
    %acc0 = bosc_ame.mlce32.m %spm_acc, %c256 {bosc_ame.fpga.slot = 0 : i32} : memref<64x64xi32, #bosc_ame.spm<0>> -> vector<4x4xi32>

    // Load the i8 operands into Tr0 (A) and Tr4 (B), accumulate into Acc0.
    %_t8 = bosc_ame.msettype %mtype_e8 : i64
    %tr_a = bosc_ame.mlae8.m %spm_a, %c64 {bosc_ame.fpga.slot = 0 : i32} : memref<4096xi8, #bosc_ame.spm<0>> -> vector<4x4xi8>
    %tr_b = bosc_ame.mlbe8.m %spm_b, %c64 {bosc_ame.fpga.slot = 4 : i32} : memref<4096xi8, #bosc_ame.spm<1>> -> vector<4x4xi8>
    %acc = bosc_ame.mqma.b.mm %acc0, %tr_a, %tr_b : vector<4x4xi32>, vector<4x4xi8>, vector<4x4xi8> -> vector<4x4xi32>

    // Store Acc0 back into SPM bank 0.  Row stride = 64 * sizeof(i32) = 256.
    %_t32b = bosc_ame.msettype %mtype_e32 : i64
    bosc_ame.msce32.m %acc, %spm_acc, %c256 : vector<4x4xi32>, memref<64x64xi32, #bosc_ame.spm<0>>

    // Write the result back to DDR.  A one-dimensional i8 endpoint is the DMA
    // contract; 16384 bytes is a multiple of the 64-byte granularity.
    %t3 = bosc_ame.dma.start %spm_c, %c, %c16384 {channel = 1 : i32} : (memref<16384xi8, #bosc_ame.spm<0>>, memref<16384xi8>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t3 : !bosc_ame.dma_token<1>
    return %c0_i32 : i32
  }
}
