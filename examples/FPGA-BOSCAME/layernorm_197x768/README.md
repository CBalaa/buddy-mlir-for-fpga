# Triton LayerNorm on the NH/E6 FPGA

This directory consumes the checked-in `kernel.frontend.mlir` standard MLIR
Contract. It builds and checks the complete backend image without a Triton
Python installation. The original kernel and frontend exporter live in
`triton-riscv/examples/BOSCAMEFPGA/layernorm_197x768`.
See [the Contract handoff guide](../../../docs/TritonContract.md).

This example lowers the ViT-B/16 LayerNorm operator over a `[197,768] f32`
hidden state through the experimental BOSCAME backend. The fixed problem size (`ROWS=197`, `WIDTH=768`,
`BLOCK=1024`, `EPS=1e-12`) and the one-program-per-row launch grid are
defined in [build.py](build.py). The operator has no dot: the reductions,
elementwise and select operations stay in Linalg and lower to SPM vector code.

The only lowering route in this directory is:

```text
kernel.frontend.mlir -> Contract verify -> bosc-ame-plan -> verified planned IR
     -> bosc-ame-emit -> lower-bosc-ame -> LLVM IR -> RVV assembly -> ELF
```

## Quick start

```bash
make boscame llvm    # Contract -> planned/BOSCAME IR -> LLVM IR
make adapter asm     # typed ABI bridge and the restricted NR assembly
make check           # typed ABI, boundary DMA and NR ELF audits (no board)
make simcheck        # functional SPM/DMA model over all 197 programs
make                 # full pipeline -> build/layernorm_197x768/*.bin
```

`make boscame`, `make llvm` and `make asm` stop at the named stage.
Only ordinary Python and NumPy are needed for the backend helpers and
simulator. Select tools explicitly or through PATH:

```sh
make BUDDY_MLIR_BINARY_DIR=/path/to/buddy-build/bin \
  LLVM_BIN=/path/to/llvm-build/bin PYTHON=python3 all check simcheck
```

To consume a fresh export without copying it into this directory:

```sh
make FRONTEND_IR=/path/to/handoff/kernel.frontend.mlir all check simcheck
```

The input must match this example's fixed entry symbol, dimensions, launch grid,
aligned/disjoint buffer promises and launcher ABI. Generate a new handoff in the
frontend repository with `make ttir` or `build.py --export-contract`; this
makefile starts at Contract verification and has no Python-to-TTIR target.

## Pipeline

The conversion commands used by `make boscame llvm` are:

```bash
buddy-opt build/layernorm_197x768/kernel.frontend.mlir \
  --kernel-contract-verify -o build/layernorm_197x768/kernel.bufferized.mlir
buddy-opt build/layernorm_197x768/kernel.bufferized.mlir \
  --bosc-ame-plan --bosc-ame-plan-verify \
  -o build/layernorm_197x768/kernel.planned.mlir
buddy-opt build/layernorm_197x768/kernel.planned.mlir \
  --bosc-ame-emit -o build/layernorm_197x768/kernel.boscame.mlir
buddy-opt build/layernorm_197x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/layernorm_197x768/kernel.llvm.mlir
buddy-translate build/layernorm_197x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/layernorm_197x768/kernel.ll
```

Contract mode performs integer normalization, static mask specialization and
safe bufferization inside the frontend. `kernel.frontend.mlir` already contains
standard buffer-form Linalg/MemRef/SCF and target-neutral `contract.*` facts;
`kernel.bufferized.mlir` is its verified copy, retained as a familiar artifact
name. No second bufferization is needed. The planner binds actual operands,
selects each DDR/SPM route, rebuilds content versions and dependencies, and
records concrete `plan.*` decisions in `kernel.planned.mlir`. Emission verifies
that plan and expands it mechanically. There is no global storage-policy option.

The contract can be produced by another frontend without Triton or BOSCAME
dialect dependencies. See the [pipeline guide](../../../docs/BOSCAMEKernelContractPipeline.md)
and [regenerated-example validation](../../../docs/BOSCAMEKernelContractValidation.md)
for the interface, resource counts and artifact hashes.

## Memory and DMA

The planner places five 4 KiB float32 roots in SPM: a shared mean/invstd
broadcast buffer, X/centered/normalized/output, Weight, Bias and squares.
Bank 0 uses offsets 0, 4096 and 8192; bank 1 uses 0 and 4096. The bank
high-water marks are 12288 B and 8192 B, totaling **20480 B**. Safe frontend
bufferization preserves centered while squares is produced, then updates the
X/output root in place. No general lifetime-based physical window overlap is
assumed.

The logical tensor has 1024 lanes while only the first 768 are valid. Each
external DMA moves the 3072-byte prefix: X, Weight and Bias read, and Out
writes. Each load first fills all 1024 SPM lanes (X/Bias padding +0, Weight
padding 1). The output stays on SPM/DMA because the store covers only the valid
prefix of the private 1024-lane tile. Each program performs four DMA events,
9216 read bytes and 3072 write bytes; private arrays do not round-trip through
DDR.

| Buffer | DDR location | External DMA |
| --- | --- | --- |
| X | row × 3072 bytes | 3072-byte read |
| Weight | 3072 bytes | 3072-byte read |
| Bias | 3072 bytes | 3072-byte read |
| Out | row × 3072 bytes | 3072-byte write |

Both reductions use all 1024 logical lanes and divide by 768. The centered
prefix keeps the original subtraction; a subview fill writes exact +0 to
`[768,1024)` before squaring. Two ordered f32 reductions use 16-lane chunks
and `vfredosum.vs`, preserving the established reduction order and tolerance.
They write two private host rank-zero buffers, which emission converts to
aligned stack storage. Explicit scalar-load tasks capture the results before
the div/rsqrt chains. Mean and invstd are broadcast through the same SPM root
at different times; this version does not fuse those broadcasts into adjacent
RVV computations.

The plan has 21 tasks, versus 19 in the saved integrated baseline. The new
contract explicitly records two scalar-load events, two host initializations
and exact tail fill dependencies. Its five physical windows replace the
baseline's five windows with a 4 KiB gap: total bank high-water drops from
24576 B to 20480 B. The baseline's register broadcast forwarding is not part
of the current mechanical emitter, so the internal vector traffic differs.
External DMA ranges, floating-point operation order and output bytes remain
unchanged for the recorded full-grid validation.

## ABI and image memory

`build.py --adapter` emits the typed C bridge to
`_mlir_ciface_triton_layernorm_197x768`. The bridge takes **four typed
unranked f32 memref descriptor pointers** and six i32 grid/program scalars,
checks 64-byte root alignment, nonalias ranges, capacities and row-major
`[ROWS,WIDTH]` / `[WIDTH]` strides, then launches all 197 row programs
in reverse order with progress markers.

`link.ld` places user DDR at `0x80100000..0x802e0000` in PMP entry 5, and
the E6 stack at `0x802e0000..0x802f0000` in entry 6. Entry 6 keeps the
`0x0f` write-through policy during both DDR phases, so descriptor or stack
spills cannot be read through a stale cached entry after a policy switch. The
freestanding image links the NH/E6 boot/platform code, the DMA/cache-policy
runtime and `nr_math.c` for `sqrtf`/`rsqrtf`; `-ffp-contract=off` keeps the
square root and division separately rounded.

## Validation

`make simcheck` executes all 197 programs in reversed order with persistent
poisoned SPM in the functional DMA/SPM model. It compares every output with an
independent sequential-f32 oracle at `atol=rtol=2e-5`, checks last-row and
Weight/Bias tail sentinels, and requires exactly the four listed DMA ranges per
program. `--write-output FILE` saves the complete 605184-byte output and
`--compare-output FILE` requires a byte-for-byte match with a previous run.

`make check` builds `bin`, verifies the typed LLVM entry ABI and absence of
heap allocation, requires four `bosc_ame.dma.start` operations, and audits the
complete ELF. The regenerated image contains 1957 instructions, 54 RVV
instructions, 26 vector memory instructions and zero AME instructions. These
are static image counts, not executed instruction or latency measurements.
The board launcher supports two different inputs and 394 row programs, with
302592 independent C-oracle comparisons and DDR/input guards.

The regenerated build passes `make`, `make check` and `make simcheck` with
`rtol=atol=2e-5`. All 197 reversed programs have maximum absolute error zero;
the 605184 output bytes match the saved baseline exactly, SHA256
`ed6261ce27a36b49df83291fe18af274a0892e8db49f4a509957405cbebf8cbe`.
Across the grid, DMA reads 1815552 bytes and writes 605184 bytes. The new
linked-ELF call-chain stack bound is **1168 B**, including aligned host scalar
storage, DMA failure hooks and the trap path, within the unchanged 65536-byte
stack. The reconstructed baseline's 992 B is a separate measurement. See the
[validation record](../../../docs/BOSCAMEKernelContractValidation.md) for hashes
and the audit's assumptions.

ELF/RVV checks establish code generation; the functional simulator establishes
BOSCAME IR execution semantics. Neither proves the absence of backend register
spills. No controlled baseline latency comparison has been made.

Upload the image using the usual platform command, then run it from the test
runner directory:

```bash
./run.sh test=test/layernorm_197x768.bin
```

Keep the runner alive until the terminal application verdict, then read back
DDR from `0x80000000` through the outputs and diagnostics; a 3 MiB dump covers
this image. The loader mirror captured before execution is not output
evidence. From this example directory, validate the final dump against the
exact uploaded ELF:

```bash
python3 ../tools/check_experimental_board.py layernorm \
  --elf build/layernorm_197x768/layernorm_197x768.elf \
  --dump layernorm.final-ddr.bin --report layernorm.board.json
```

The checker recomputes the final round in f32 with the current select-zero
semantics and `2e-5 * (1 + abs(expected))` tolerance, and checks both rounds'
counters, inputs and DDR guards. Stored board reports in
[docs/validation](../../../docs/validation/) predate the current resident-SPM
placement and select-zero variance semantics; they describe their own earlier
images. The current vector-reduction image has not been revalidated on a board.

## Files

| File | Role |
| --- | --- |
| `kernel.frontend.mlir` | Standard Contract input, exported in triton-riscv |
| `build.py` | Triton frontend export, fixed shape, typed ABI adapter and LLVM ABI check |
| `makefile` | TTIR -> BOSCAME -> LLVM -> assembly -> ELF/image pipeline |
| `link.ld` | NH/E6 entries, user DDR windows and stack cache policy |
| `launch.c` | Operands, two-round guarded f32 board oracle and NH report |
| `simcheck.py` | Whole-program functional execution, numerical oracle and guard checks |
| `../tools/restrict_fpga_assembly.py` | Restricted NR instruction stream from the `llc` assembly |
| `../tools/check_nr_elf.py` | Complete-ELF instruction audit |

See the [pipeline guide](../../../docs/BOSCAMEKernelContractPipeline.md)
and [current validation record](../../../docs/BOSCAMEKernelContractValidation.md).
