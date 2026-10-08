# `linear_196x768x768` AME/DMA/SPM prototype

This directory contains the first C implementation of the ViT patch
projection for the patch6 hardware.  It is an independent diagnostic image;
the existing FP32 Triton/RVV operator remains the model baseline until the
INT8 quantisation error is measured against the ViT graph.

The implementation uses the hardware limits from `patch6.pdf`:

* `M=196` is split into `64+64+64+4`; `N=K=768` use twelve 64-wide tiles.
* A and B are packed tile-major, so every transfer is one contiguous 64-byte
  aligned DMA request.
* DMA0 stages one complete A panel (the twelve K tiles for one M tile) into
  SPM0 and keeps it resident for all twelve N panels.  DMA1 ping-pongs the B
  tiles in SPM1 while AME consumes the alternate bank.
* Two 16 KiB C buffers in SPM0 are ping-ponged.  AME computes the current C
  tile while DMA0 writes the previous C tile back to DDR; the A panel is not
  evicted for either operation.
* The M=196 tail panel is zero-padded to a full 64-row AME tile.  The padded
  rows therefore write back deterministic zeros without a separate tail clear.
* AME executes signed INT8 `A*B^T+C`; the output is copied back to DDR as
  tile-major INT32 data.
* The C FP32 wrapper uses per-row A and per-output-channel B symmetric scales,
  then dequantises the AME accumulator.  It is deliberately not silently
  substituted into the FP32 graph.

Build the host oracle and the patch6 bare image with:

```sh
make -C examples/FPGA-BOSCAME/vit-base-patch16-224/linear_196x768x768_ame_dma_spm all
```

The bare image is compatible with the patch6 5.5296 MHz UART protocol.  Copy
`build/linear_196x768x768_ame_dma_spm.bin` into the FPGA tester's `test/`
directory and run it on an idle FPGA slot.  The UART report includes the
cycle count and checks the complete output with two independent 64-bit
digests plus approximately 150 exact 768-term samples.  The host
`host-check` still performs the exhaustive element-by-element oracle.  The
single-pass board check avoids spending minutes recomputing 115 million
scalar products on the NH core after the AME kernel has already completed.

The progress values printed by NH are mailbox values from the E6 side:
`0x310/0x311` cover page-table installation, `0x51x` covers the resident A
panel DMA, `0x44x` identifies the B-tile/AME loop, and `0x30` is the output
digest check.  A successful run ends with `PASS` and `errors=0`.

On FPGA4 with `sys_clk=5.5296 MHz`, the current deterministic INT8 test took
`0x200440` cycles, or about **379.46 ms** for the 196×768×768 GEMM.  This is
115,605,504 MACs (about 304.7 MMAC/s, 0.609 GOPS counting multiply and add).
