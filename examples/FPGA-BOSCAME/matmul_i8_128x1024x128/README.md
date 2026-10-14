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
No baseline performance comparison has been made. The default `boscame=false` and the legacy
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

The generated ordinary packs are aligned stack allocations and DMA sources;
results return through explicit one-dimensional vector copies. `link.ld`
places user DDR at `0x80100000..0x802e0000` in PMP entry 5, and the E6 stack at
`0x802e0000..0x802f0000` in entry 6, checking that C and its guards do not overlap
the stack. Entry 6 keeps the `0x0f` write-through policy in both DDR read/write
phases, so compiler spills do not switch between cached and uncached access.
A diagnostic run with the stack inside entry 5 trapped on the second program:
an A descriptor spill was read as the old boolean value 1, then dereferenced
at address 9. The separated layout addresses that cache-policy hazard.
The NH stack remains in its original window. Shared
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
make boardcheck       # Two full launches with signed extremes, wrapping and guards
```

`make boardcheck` builds
`build/matmul_i8_128x1024x128/boardcheck/matmul_i8_128x1024x128.boardcheck.bin`.
The driver runs two different complete 512-program launches, with 262144
output comparisons against its independent C oracle. A/B include signed-i8
extremes, B retains its asymmetric physical `[N,K]` layout, and nonzero C
exercises i32 wrap. The oracle uses unsigned arithmetic to define that wrap
without signed-overflow undefined behavior. Inputs and before/after DDR
guards are checked after each round; the second launch reuses SPM without
clearing it. The shortened `debug` oracle does not replace this validation.

Upload this image using the usual platform command, then run it from the test
runner directory:

```bash
./run.sh test=test/matmul_i8_128x1024x128.boardcheck.bin
```

Keep the session alive until the terminal application verdict. Read back DDR
after that verdict, starting at `0x80000000` through the outputs and diagnostics;
a 3 MiB dump covers this image. The loader's pre-execution mirror cannot be
used as output evidence. From this example directory, validate the final dump
using the exact ELF used to build the uploaded image:

```bash
python3 ../tools/check_experimental_board.py gemm \
  --elf build/matmul_i8_128x1024x128/boardcheck/matmul_i8_128x1024x128.boardcheck.elf \
  --dump gemm.final-ddr.bin --report gemm.board.json
```

The checker independently computes the full final-round product in i64,
adds initial C and explicitly wraps to i32. Every output bit must match. It
also verifies both rounds' counters and the final input bytes/DDR guards;
missing data or any mismatch returns nonzero. JSON includes ELF/dump hashes
and comparison counts.

| File | Role |
| --- | --- |
| `kernels.py` | Original Triton computation |
| `build.py` | Frontend, fixed shape, typed and legacy adapters, LLVM ABI check, host path |
| `makefile` | New TTIR→BOSCAME→LLVM→assembly→ELF/image pipeline and variants |
| `link.ld` | NH/E6 entries, user DDR data and stable stack cache policy |
| `launch.c` | Operands, kernel facade, two-round guarded INT32 board oracle and NH report |
| `simcheck.py` | Full-grid functional execution, numerical oracle and guards |
| `../tools/check_experimental_board.py` | Independent final-DDR checker using exact ELF symbols |
| `launch-host.c`, `support.c`, `support.h` | Existing host reference |

On 2026-10-14, FPGA4 reported PASS for both full rounds: 256 rows / 262144
output values checked, zero errors and zero guard/input errors. Independent
post-execution DDR comparison matched all 131072 final-round i32 values exactly.
See [the board evidence](../../../docs/validation/experimental-boscame/README.md).
No baseline performance comparison has been made.
