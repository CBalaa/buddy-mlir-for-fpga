# Triton LayerNorm on the NH/E6 FPGA

This example lowers the ViT-B/16 LayerNorm operator over a `[197,768] f32`
hidden state through the experimental BOSCAME backend. The Triton source is
[kernel.py](kernel.py); the fixed problem size (`ROWS=197`, `WIDTH=768`,
`BLOCK=1024`, `EPS=1e-12`) and the one-program-per-row launch grid are
defined in [build.py](build.py). The operator has no dot: the reductions,
elementwise and select operations stay in Linalg and lower to SPM vector code.

The only lowering route in this directory is:

```text
TTIR -> triton-to-linalg-experimental -> one-shot bufferize
     -> linalg-to-boscame -> lower-bosc-ame -> LLVM IR -> RVV assembly -> ELF
```

## Quick start

```bash
make ttir            # Triton frontend -> kernel.ttir
make boscame llvm    # TTIR -> frontend/bufferized/BOSCAME IR -> LLVM IR
make adapter asm     # typed ABI bridge and the restricted NR assembly
make check           # typed ABI, boundary DMA and NR ELF audits (no board)
make simcheck        # functional SPM/DMA model over all 197 programs
make                 # full pipeline -> build/layernorm_197x768/*.bin
```

`make boscame`, `make llvm` and `make asm` stop at the named stage.
The TTIR export imports Triton, so that step runs with the Triton/triton-riscv
environment. The makefile prefers in-tree Buddy/LLVM tools
and accepts `PYTHON`, `BUDDY_OPT`, `BUDDY_TRANSLATE`, `LLC`, `RISCV_CC` and linker
overrides, for example:

```bash
make boscame llvm simcheck PYTHON=/path/to/boscame/python
```

## Pipeline

The conversion commands used by `make boscame llvm` are:

```bash
buddy-opt build/layernorm_197x768/kernel.ttir \
  --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias" \
  -o build/layernorm_197x768/kernel.frontend.mlir
buddy-opt build/layernorm_197x768/kernel.frontend.mlir \
  --empty-tensor-to-alloc-tensor \
  --one-shot-bufferize="bufferize-function-boundaries=true buffer-alignment=64" \
  --canonicalize --cse -o build/layernorm_197x768/kernel.bufferized.mlir
buddy-opt build/layernorm_197x768/kernel.bufferized.mlir \
  --linalg-to-boscame="store-policy=spm-dma buffer-placement=true" \
  --promote-buffers-to-stack -o build/layernorm_197x768/kernel.boscame.mlir
buddy-opt build/layernorm_197x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/layernorm_197x768/kernel.llvm.mlir
buddy-translate build/layernorm_197x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/layernorm_197x768/kernel.ll
```

`--triton-to-linalg-experimental` materializes the Triton source, mask and
program-axis facts as `bosc_ame.*` attributes on the buffer-form Linalg IR.
`--linalg-to-boscame` reads those facts, analyzes the Linalg computation and
its memory facts, plans root-buffer placement and DMA transfers, then emits the
BOSCAME dialect. `--lower-bosc-ame` lowers the DMA/SPM operations to the
runtime ABI and the surrounding LLVM dialect. See
[the compiler contract](../../../docs/TritonToLinalgExperimentalBOSCAME.md) for
the fact and plan reports, the DMA ABI and the supported accesses.

## Memory and DMA

Placement plans six compiler-managed float32 roots in fixed SPM windows.
Register forwarding removes the shared mean/invstd broadcast root, leaving
five 4 KiB roots: zero padding, X/centered/normalized, Weight, Bias and squares.
The surviving offsets keep each bank's high-water mark at 12 KiB; the unused
window is not compacted. No lifetime overlap or spill search is used.
X -> centered -> normalized -> weighted -> biased is computed in place in the
same root.  The pass option defaults to false; this example enables it
explicitly.

The logical tensor has 1024 lanes while only the first 768 are valid, so each
external DMA moves the 3072-byte prefix: X, Weight and Bias read, and Out
writes. The whole per-program traffic is four DMA events, 9216 read bytes and
3072 write bytes; internal float arrays never round-trip through DDR.

| Buffer | DDR location | External DMA |
| --- | --- | --- |
| X | row x 3072 bytes | 3072-byte read |
| Weight | 3072 bytes | 3072-byte read |
| Bias | 3072 bytes | 3072-byte read |
| Out | row x 3072 bytes | 3072-byte write |

Each load first fills the full 1024-lane SPM buffer (invalid X lanes are 0,
invalid Weight lanes are 1, invalid Bias lanes are 0) and then DMAs only the
valid prefix. Both reductions still use all 1024 logical lanes and divide by
768; the centered select zeros the padded lanes before squaring, so they
contribute zero to the variance.

Both sums load `vector<16xf32>` and use ordered vector reduction, producing
`vfredosum.vs`: each 1024-lane sum takes 64 loop iterations. Initial zero and
reduction results stay in scalar SSA values; the generated kernel has no
scalar scratch allocation or DDR-policy call. Mean and invstd feed reusable
register broadcasts directly, avoiding 16 KiB of SPM reads/writes per row.
The broadcast and reduction operations retain their task IDs and dependencies.
The assembly filter's `--expand-f32-sub` option expands scalar-vector
subtraction into supported instructions without extending the ISA allowlist.

The centered select lowers to the fixed ranges `[0,768)` and `[768,1024)`.
Its graph metadata drops the unused mask input and dependency. The private
width, index and mask tasks are then removed, leaving 19 of the original 22
tasks with their surviving IDs unchanged. A final DCE removes the unused
integer/predicate arrays and lowering artifacts while preserving graph tasks.
This cleanup keeps the floating-point operations and results bit for bit.

Because the centered prefix already occupies the output root, lowering skips
its `[0,768)` self-copy and keeps the `[768,1024)` tail clear. Squaring reuses
one vector load for both operands. Together these remove 10 KiB of explicit
SPM reads/writes per row from the BOSCAME IR. Task 14 retains the inherited
prefix dependency, and task 15 retains its two logical input operands; the
graph still has 19 tasks. These changes preserve the floating operations and
their order. See [the validation record](../../../docs/validation/boscame-local-memory-redundancy/README.md)
for output-byte comparison and generated assembly results.

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

`make check` builds `bin` and then verifies the typed entry LLVM ABI and
the absence of heap allocation in `kernel.ll`, requires four
`bosc_ame.dma.start` operations in `kernel.boscame.mlir`, and audits the
complete ELF (1905 instructions, 54 RVV instructions, 24 vector memory
instructions, zero AME instructions). The launcher runs two different inputs,
all 197 rows in reverse order per round (394 programs), and checks 302592
outputs against its independent C oracle with before/after DDR guards and
input-change checks.

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
| `kernel.py` | Original Triton computation |
| `build.py` | Triton frontend export, fixed shape, typed ABI adapter and LLVM ABI check |
| `makefile` | TTIR -> BOSCAME -> LLVM -> assembly -> ELF/image pipeline |
| `link.ld` | NH/E6 entries, user DDR windows and stack cache policy |
| `launch.c` | Operands, two-round guarded f32 board oracle and NH report |
| `simcheck.py` | Whole-program functional execution, numerical oracle and guard checks |
| `../tools/restrict_fpga_assembly.py` | Restricted NR instruction stream from the `llc` assembly |
| `../tools/check_nr_elf.py` | Complete-ELF instruction audit |

See [the compiler contract](../../../docs/TritonToLinalgExperimentalBOSCAME.md)
and the [root buffer placement validation record](../../../docs/validation/boscame-buffer-placement/README.md).
The [vector reduction validation record](../../../docs/validation/boscame-vector-reduction/README.md)
describes the scalar SSA and register-broadcast changes.
