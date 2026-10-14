# Triton INT8 GEMM on the NH/E6 FPGA

This example lowers `A[128,128] i8 @ B[1024,128] i8ᵀ + C[128,1024] i32`
through the experimental BOSCAME entry. B is physically `[N,K]`, with K
contiguous. The Triton kernel body and its two outer K blocks are unchanged.
Each program computes a 16×16 tile; the launch grid is 8×64×1.

```bash
make ttir
make boscame llvm
make check
make simcheck
```

The frontend requires the Triton/triton-riscv Python environment. The makefile
prefers in-tree Buddy/LLVM tools and accepts `PYTHON`, `BUDDY_OPT`,
`BUDDY_TRANSLATE`, `LLC`, `RISCV_CC` and linker overrides.

The conversion commands used by `make boscame llvm` are:

```bash
buddy-opt build/matmul_i8_128x1024x128/kernel.ttir \
  --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias" \
  -o build/matmul_i8_128x1024x128/kernel.boscame.mlir
buddy-opt build/matmul_i8_128x1024x128/kernel.boscame.mlir \
  --lower-bosc-ame -o build/matmul_i8_128x1024x128/kernel.llvm.mlir
buddy-translate build/matmul_i8_128x1024x128/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/matmul_i8_128x1024x128/kernel.ll
```

Loads independently DMA-stage A, B and previous C into SPM, then copy them
with one-dimensional vectors into ordinary buffers. Standard marked matmuls
are bufferized before their AME lowering. The helper packs A/B/zero C, executes
DMA and AME, then materializes each dot output in ordinary memory. The outer
K loop carries ordinary results. The original previous C is added once, and
stores vector-stage valid results into SPM before DMA writes DDR.

This first version uses ordinary temporary buffers and extra DMA/vector
movement. It does not retain AME accumulators across the outer K loop.
Performance has not been measured. The default `boscame=false` and the legacy
`--triton-to-boscame` entry remain compatible; this example now selects the new
entry by default. See [the compiler contract](../../../docs/TritonToLinalgExperimentalBOSCAME.md).

## ABI and image memory

The kernel C wrapper takes **three typed unranked memref descriptor pointers**
and six i32 grid/program scalars. The unranked descriptor is
`{int64_t rank; void *descriptor;}` and points to a rank-one descriptor
`{allocated, aligned, offset, size, stride}`. All descriptor offsets and sizes
are **elements**: C has 131072 i32 elements, not 524288 i8 elements.

`build.py --adapter experimental-boscame` emits the real bridge to
`_mlir_ciface_triton_matmul_i8_128x1024x128`. It checks root alignment, nonalias
ranges, capacities and the expected physical strides, then executes the full
grid. The launcher's existing scratch facade is retained for compatibility;
the new kernel does not receive or use that scratch pointer.
`build.py --adapter triton-to-boscame` still emits the old four-byte-descriptor
bridge for users of the legacy pass.

The generated ordinary packs are aligned stack allocations and some are DMA
endpoints. `link.ld` places the E6 stack at `0x802e0000..0x802f0000`, inside the
exported `0x80100000..0x80300000` DDR/PMP window, and checks that it does not
overlap C. The NH stack remains in its original window. Shared
`runtime/boscame/nh_e6` boot/platform setup installs PMP and bare addressing;
`runtime/boscame/runtime/bosc_ame_dma.c` supplies DMA and DDR cache policies.

The kernel assembly enables RVV and AME, passes through the existing AME word
encoder and NR instruction/fence checks, and is linked into the NH/E6 image.
Runtime/adapter C compilation keeps RVV disabled so aggregate descriptor copies
cannot introduce unaudited vector instructions. `make check` checks the entry
ABI, absence of heap calls, expected AME instructions, RVV memory instructions,
DDR/PMP layout and the full ELF allowlist. These checks do not execute a board.

## Validation and variants

`make simcheck` runs every program with persistent poisoned SPM, checks DDR
access ranges and guards, and compares all 131072 results with the unchanged
independent NumPy INT32 oracle. The compiler regression suite additionally
uses signed extrema, random nonzero C and i32 overflow; a separate K=256 test
checks internal blocking, a nonzero dot initializer and rectangular masks.
All integer comparisons are exact.

```bash
make AME=0 check       # Existing scalar RISC-V path, separate noame/ directory
make host             # Existing Triton-to-linalg host oracle
make debug            # Progress markers and the launcher's 16-row board oracle
```

| File | Role |
| --- | --- |
| `kernels.py` | Original Triton computation |
| `build.py` | Frontend, fixed shape, typed and legacy adapters, LLVM ABI check, host path |
| `makefile` | New TTIR→BOSCAME→LLVM→assembly→ELF/image pipeline and variants |
| `link.ld` | NH/E6 entries, DDR data and authorized DMA stack |
| `launch.c` | Operands, kernel facade, independent INT32 oracle and NH report |
| `simcheck.py` | Full-grid functional execution, numerical oracle and guards |
| `launch-host.c`, `support.c`, `support.h` | Existing host reference |

**Board execution/readback has not been verified for this new path.** Use the
existing platform runner and compare the readback with the launcher oracle
before making a board-correctness or performance claim.
