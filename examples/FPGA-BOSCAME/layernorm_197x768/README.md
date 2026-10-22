# LayerNorm through independent DMA load/store

The example retains the original Triton LayerNorm computation for 197 rows of
768 f32 elements. It contains no dot. The experimental BOSCAME path therefore
uses independent DMA loads/stores while Linalg/Arith/Math handle computation.

```bash
make ttir
make llvm adapter asm
make simcheck
make check bin
```

Use the Triton/triton-riscv Python environment. Buddy/LLVM tools default to the
in-tree builds and can be overridden in the makefile. The compiler pipeline
materializes tensor results, bufferizes them, then plans BOSCAME memory and DMA:

```bash
buddy-opt build/layernorm_197x768/kernel.ttir \
  --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias" \
  -o build/layernorm_197x768/kernel.frontend.mlir
buddy-opt build/layernorm_197x768/kernel.frontend.mlir \
  --empty-tensor-to-alloc-tensor \
  --one-shot-bufferize="bufferize-function-boundaries=true buffer-alignment=64" \
  --canonicalize --cse -o build/layernorm_197x768/kernel.bufferized.mlir
buddy-opt build/layernorm_197x768/kernel.bufferized.mlir \
  --linalg-to-boscame="store-policy=spm-dma" --promote-buffers-to-stack \
  -o build/layernorm_197x768/kernel.boscame.mlir
buddy-opt build/layernorm_197x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/layernorm_197x768/kernel.llvm.mlir
buddy-translate build/layernorm_197x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/layernorm_197x768/kernel.ll
```

The logical tensor has 1024 lanes; the valid DDR prefix has 768 f32 elements,
so **each DMA transfers 3072 bytes**. X and Out have dynamic row offsets
`programId * 768` elements, proved to be `programId * 3072` aligned bytes.

| Buffer | DDR address | Padded SPM lanes |
| --- | --- | --- |
| X | X + row × 768 | 0 |
| Weight | Weight | 1 |
| Bias | Bias | 0 |
| Out | Out + row × 768 | No DDR write |

Each load prepares a 1024-element SPM buffer, initializes its padded lanes
with one-dimensional vectors, and DMA-loads only the valid 768 elements.
Elementwise computation reads/writes SPM with one-dimensional RVV vectors.
Each reduction uses single-lane vector loads from SPM and adds lane zero in
the original f32 order; scalar arithmetic accesses registers or ordinary
scalar temporaries. This preserves per-addition rounding without requiring
unsupported vector slides. Elementwise operations retain 64-byte vectors.
The final result DMA-stores directly from SPM to Out. No 4096-byte user DDR
access, scalar SPM load/store or ordinary-buffer bounce is emitted.

Both reductions still use all 1024 logical lanes. The mean uses zero-valued
masked X lanes, and the current centered select sets padded lanes to zero
before squaring, so they contribute zero to variance. The computation does
not replace the 1024-lane reduction with a 768-lane reduction.

`make adapter` generates a bridge with four typed unranked f32 descriptors
and six grid/program scalars. Descriptor sizes and offsets are elements. The
bridge checks 64-byte root alignment, nonalias ranges, capacities and strides,
then launches all 197 rows. Computation buffers live in bounded SPM views;
ordinary scalar temporaries and descriptors live on an aligned stack. `link.ld` puts user DDR in PMP entry 5 at
`0x80100000..0x802e0000`, and the E6 stack in entry 6 at
`0x802e0000..0x802f0000`. The stack keeps the `0x0f` write-through policy during
both DDR phases. This prevents descriptor spills from being read through a
stale cache entry after a policy switch. The image links the existing DMA/cache-policy
runtime, NH/E6 boot/platform code and freestanding `nr_math.c` implementations
of `sqrtf` and `rsqrtf`. `rsqrtf` rounds the f32 square root and division
separately, with floating-point contraction disabled in the build.

`make simcheck` executes all rows in reversed order with persistent poisoned
SPM. It compares with an independent sequential-f32 oracle that zeros padded
centered lanes and checks
last-row and Weight/Bias tail sentinels plus the exact four DMA ranges per row.
Compiler tests also compare exactly against the unchanged experimental buffer
path; the archived maximum absolute error against a high-precision padded
formula was 2.61e-5 on the deterministic input and describes the earlier
semantics. The select-zero correction changes variance numerics for nonzero
means; the reduction order and valid-lane operations are unchanged.

`make check bin` also audits the complete ELF and creates
`build/layernorm_197x768/layernorm_197x768.bin`. Its launcher executes two
different inputs, all 197 rows per round in reverse order: 394 programs and
302592 output comparisons against an independent select-zero f32 C oracle. SPM is
poisoned before the first launch and retained for the second. Before/after DDR
guards cover every buffer, particularly the last row and Weight/Bias tails;
the launcher also checks inputs for modification.

Upload the image using the usual platform command and invoke the runner from
its own directory:

```bash
./run.sh test=test/layernorm_197x768.bin
```

After the application's terminal verdict, capture final DDR from `0x80000000`
through the output and diagnostics; a 3 MiB dump covers this image. The runner's
loader mirror precedes execution and is not a result dump. From this example
directory, check the final dump against the exact uploaded ELF:

```bash
python3 ../tools/check_experimental_board.py layernorm \
  --elf build/layernorm_197x768/layernorm_197x768.elf \
  --dump layernorm.final-ddr.bin --report layernorm.board.json
```

The stored FPGA5/FPGA6 records below are from binaries built before the
select-zero correction. Their oracle centered padded lanes as `0 - mean`, so
each padded lane contributed `mean²` to variance; those board runs validate
that earlier behavior, not the current kernel. The linked board checker also
uses the current select-zero oracle for future images; the archived reports
below still describe the earlier binaries and were not revalidated. It checks
every final-round output with
`2e-5 * (1 + abs(expected))`, reports exact matches and maximum absolute error,
and checks both rounds' counters plus final inputs/guards. Failures return
nonzero. On 2026-10-14, FPGA5 and FPGA6 each reported PASS for both rounds:
302592 values checked, zero errors and zero guard errors. The independent post-execution DDR check
matched all 151296 final-round values bit for bit with the sequential-f32
oracle (maximum absolute error 0). This is the measured result for these
inputs, not a general floating-point bit-identity guarantee. Against the f64
padded formula, maximum absolute error was 2.993e-5.
Those reports describe the earlier staging implementation and earlier
variance semantics. The direct-SPM revision also passed on FPGA6 (NR patch9)
on 2026-10-15 with those same earlier semantics: all 302592 board comparisons
and 151296 independently read-back final-round values were exact, with zero
input/guard errors. See the [direct-SPM board record](../../../docs/validation/experimental-boscame-direct-spm/README.md).

The direct-SPM path removes the unconditional ordinary-buffer staging.
SPM capacity includes the full 1024-lane buffers and static intermediate values.
No controlled baseline latency comparison has been made. ELF/RVV checks establish code generation; hardware correctness
requires the execution and numerical readback checks above.

See [the compiler contract and tests](../../../docs/TritonToLinalgExperimentalBOSCAME.md).
