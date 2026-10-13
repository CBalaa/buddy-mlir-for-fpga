//===----------------------------------------------------------------------===//
//
// TTIR -> BOSCAME conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_BOSCAME_CONVERSION_PASSES
#define TRITON_TO_BOSCAME_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToBOSCAME : Pass<"triton-to-boscame", "mlir::ModuleOp"> {
  let summary = "Lower the target Triton matmul regions to the BOSCAME "
                "SPM/DMA/AME datapath";
  let description = [{
    Rewrites the statically shaped signed `i8 x i8 -> i32` `tt.dot` regions of
    a module into the BOSCAME three-level datapath:

    * DDR -> SPM: the `tt.load` operands of the dot are staged into fixed SPM
      windows with DMA transfers. Fully valid, aligned tiles use direct DMA;
      other tiles are first assembled in aligned DDR scratch using masked CPU
      copies, `other` for excluded lanes, and zeros beyond the logical K;
    * SPM -> AME: the staged tiles are loaded into the tile register file and
      accumulated with `bosc_ame.tile.mma`, with `scf.for` K blocking when the
      reduction is wider than one tile; the accumulator stays an SSA value that
      the loops carry;
    * AME result -> SPM -> DDR: the accumulator is written to SPM and DMA'd
      into DDR scratch. The original `previous` addition and masked store run
      in DDR, so excluded original addresses are neither read nor written.
      The CPU never reads, writes, or fills SPM.

    A fourth byte memref is appended for caller-provided, 64 B aligned DDR
    scratch. `bosc_ame.scratch_bytes` records its minimum size; its contents are
    initialized on every program. `llvm.emit_c_interface` requests a generated
    descriptor-pointer wrapper for C callers.

    Everything outside a target region - other dots, scalar code, unrelated
    memory accesses - is left untouched for `--triton-to-linalg-experimental`
    conversion.
  }];
  let constructor = "mlir::triton::createTritonToBOSCAMEPass()";
  let options = [
    Option<"dmaAbi", "dma-abi", "std::string", "\"\"",
           "Explicit caller contract for the DDR pointer arguments.  When set "
           "to `args_aligned64_nonalias` the pass records that contract on the "
           "module and may use DMA transfers for DDR endpoints it can prove "
           "aligned. The contract is required: CPU accesses cannot replace "
           "DMA into SPM. Original operands need no padding; masked or "
           "unaligned rows are packed into DDR scratch before DMA.">,
  ];
}

#endif // TRITON_TO_BOSCAME_CONVERSION_PASSES
