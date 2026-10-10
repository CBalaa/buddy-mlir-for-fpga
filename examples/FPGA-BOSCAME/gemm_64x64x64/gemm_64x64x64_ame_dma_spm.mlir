//===- gemm_64x64x64_ame_dma_spm.mlir - 64x64x64 INT8 GEMM via SPM + AME -===//
//
// One 64x64x64 signed INT8 => INT32 matrix multiplication written in the
// BOSCAME dialect and lowered end to end by `buddy-opt --lower-bosc-ame`:
//
//   DDR A/B/Z --DMA0/DMA1--> SPM0/SPM1 --tile.load_a/b--> tr0/tr4
//             --tile.mma--> acc0 --tile.store_acc--> SPM0 --DMA1--> DDR C
//
// This kernel is *only* the computation: it contains no hardware register
// name, no `mtype` value, no tile size and no stride.  The five
// `bosc_ame.tile.*` operations speak in matrix shapes and element types, and
// the pass derives the rest from them:
//
//   * `bosc_ame.tile.load_a`  reads the A tile from SPM into the tile
//     register file, `bosc_ame.tile.load_b` does the same for B and
//     `bosc_ame.tile.load_acc` seeds the accumulator.  Each one carries a
//     `Read` effect on its SPM buffer.
//   * `bosc_ame.tile.mma` computes `C += A * B` on the AME, returning the
//     updated accumulator as an SSA value, so the accumulation order is
//     explicit in the IR.
//   * `bosc_ame.tile.store_acc` writes the INT32 accumulator tile back into
//     SPM and returns nothing: the result of the computation is the memory it
//     writes.
//
// What the pass adds around them:
//
//   * the register-form configuration (`msettilem/msettilen/msettilek` and
//     `msettype`) that each load/store/mma depends on, in the right order, so
//     no instruction ever runs against configuration left over from a
//     different datapath phase;
//   * the physical row stride of each operand in bytes, taken from the memref
//     layout: A and the accumulator use the leading stride, while B - stored
//     as an NxK buffer with K contiguous but described logically as KxN -
//     uses the trailing stride;
//   * the fixed register slots `tr0`/`tr4`/`acc0` of the current FPGA target,
//     validated against the live ranges of this kernel;
//   * `bosc_ame.spm.view` -> a fixed SPM base address (`bankBase[bank] +
//     offset`), so this kernel has no SPM workspace argument: the ABI is the
//     four DDR buffers;
//   * `bosc_ame.dma.start` / `bosc_ame.dma.wait` -> runtime calls.  The token
//     is the start call's `i64` result; the wait has no result and the runtime
//     performs the completion barrier and the STATUS check itself, taking its
//     error path on failure.  A normal wait return therefore means "completed
//     and succeeded", and the kernel continues straight into the AME sequence.
//
// Memory layout in SPM (both banks are 128 KiB):
//
//   SPM0 + 0x0000 : A, 4096 bytes, row-major [M, K] = [64, 64]
//   SPM1 + 0x0000 : B, 4096 bytes, *physical* [N, K] = [64, 64] with K
//                   contiguous - what the AME B load expects
//   SPM0 + 0x2000 : the 16384-byte INT32 accumulator tile
//
// DMA is byte-oriented, so every endpoint is a 1-D `i8` memref and the tile
// operations read 2-D views of the very same storage:
//
//   * `memref.view` re-types the A and the accumulator windows.  The
//     accumulator changes element type (i8 -> i32), which is exactly what
//     `memref.view` is for; a `memref.reinterpret_cast` cannot change element
//     type and is not used here.
//   * the B view is an i8 -> i8 *layout* change: the storage is the physical
//     [N, K] buffer, and the MMAs want `B[K, N]`, so the same bytes are
//     described with strides [1, K].  The AME load then uses the physical row
//     pitch (strides[1] * sizeof(i8) = 64 bytes), not strides[0].
//
// `Z` is a permanent 16 KiB zero tile staged into the accumulator window: the
// validated FPGA subset has no "zero the accumulator" instruction, so the
// kernel seeds the accumulator with a zero tile and lets `tile.mma` accumulate
// on top of it.
//
// The module carries `bosc_ame.target = "nr-fpga"`, the explicit selection of
// the FPGA AME contract (register-form configuration, fixed tr0/tr4/acc0
// slots, raw i32 accumulator memory).  Tile operations are never lowered
// against an implicitly assumed target.  `bosc_ame.dma.abi =
// "args_aligned64_nonalias"` is the caller's promise that every DDR pointer is
// 64-byte aligned and that distinct pointers do not overlap; `e6_main()`
// honours it with `aligned(4096)` buffers.
//
// `llvm.emit_c_interface` gives the kernel the `_mlir_ciface_gemm_64x64x64`
// entry point the C launcher calls.
//===- ------------------------------------------------------------------===//

module attributes {bosc_ame.target = "nr-fpga",
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @gemm_64x64x64(%a: memref<4096xi8>, %b: memref<4096xi8>,
                           %z: memref<16384xi8>, %c: memref<16384xi8>) -> i32
      attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    %c0_idx = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c16384 = arith.constant 16384 : index

    // Fixed SPM windows: A in bank 0, B in bank 1, the accumulator tile in
    // bank 0.  Each view compiles to a constant physical address
    // (0x20000000, 0x20400000, 0x20002000), so no workspace root argument is
    // passed or needed.
    %spm_a = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %spm_b = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<1>>
    %spm_c = bosc_ame.spm.view {bank = 0 : i32, offset = 8192 : i64} : memref<16384xi8, #bosc_ame.spm<0>>

    // 2-D views for the tile operations.  A is row-major [64, 64]; the
    // accumulator is the same window re-typed as i32; B is the physical
    // [N, K] buffer described as the logical [K, N] with strides [1, 64].
    %a2d = memref.view %spm_a[%c0_idx][] : memref<4096xi8, #bosc_ame.spm<0>> to memref<64x64xi8, #bosc_ame.spm<0>>
    %b2d = memref.reinterpret_cast %spm_b to offset: [0], sizes: [64, 64], strides: [1, 64] : memref<4096xi8, #bosc_ame.spm<1>> to memref<64x64xi8, strided<[1, 64], offset: 0>, #bosc_ame.spm<1>>
    %acc2d = memref.view %spm_c[%c0_idx][] : memref<16384xi8, #bosc_ame.spm<0>> to memref<64x64xi32, #bosc_ame.spm<0>>

    // Stage A and B from DDR into the two SPM banks.  DMA0 and DMA1 are
    // independent engines, so both transfers overlap; both are 64-byte
    // aligned and 4096 bytes (= 64 AXI beats).
    %t0 = bosc_ame.dma.start %a, %spm_a, %c4096 {channel = 0 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    %t1 = bosc_ame.dma.start %b, %spm_b, %c4096 {channel = 1 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t1 : !bosc_ame.dma_token<1>

    // C = 0 for this pass: stage a permanent zero tile into the accumulator
    // region so the accumulator load starts from zeros instead of stale SPM
    // bytes.  The load below may not start before this transfer completed.
    %tz = bosc_ame.dma.start %z, %spm_c, %c16384 {channel = 0 : i32} : (memref<16384xi8>, memref<16384xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %tz : !bosc_ame.dma_token<0>

    // One hardware tile covers the whole problem: M = N = K = 64.  The
    // accumulator is seeded from SPM, then the two operand tiles are loaded
    // and one MMA accumulates A x B into it.
    %acc0 = bosc_ame.tile.load_acc %acc2d : memref<64x64xi32, #bosc_ame.spm<0>> -> vector<64x64xi32>
    %tr_a = bosc_ame.tile.load_a %a2d : memref<64x64xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    %tr_b = bosc_ame.tile.load_b %b2d : memref<64x64xi8, strided<[1, 64], offset: 0>, #bosc_ame.spm<1>> -> vector<64x64xi8>
    %acc = bosc_ame.tile.mma %acc0, %tr_a, %tr_b : vector<64x64xi32>, vector<64x64xi8>, vector<64x64xi8> -> vector<64x64xi32>

    // Write the INT32 accumulator tile back into SPM, then move it to DDR.
    // The one-dimensional i8 endpoint is the DMA contract; 16384 bytes is a
    // multiple of the 64-byte granularity.
    bosc_ame.tile.store_acc %acc, %acc2d : vector<64x64xi32>, memref<64x64xi32, #bosc_ame.spm<0>>
    %t3 = bosc_ame.dma.start %spm_c, %c, %c16384 {channel = 1 : i32} : (memref<16384xi8, #bosc_ame.spm<0>>, memref<16384xi8>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t3 : !bosc_ame.dma_token<1>
    return %c0_i32 : i32
  }
}
