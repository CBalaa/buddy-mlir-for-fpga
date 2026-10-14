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
in-tree builds and can be overridden in the makefile. The two compiler stages
are directly connected:

```bash
buddy-opt build/layernorm_197x768/kernel.ttir \
  --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias" \
  -o build/layernorm_197x768/kernel.boscame.mlir
buddy-opt build/layernorm_197x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/layernorm_197x768/kernel.llvm.mlir
buddy-translate build/layernorm_197x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/layernorm_197x768/kernel.ll
```

The logical tensor has 1024 lanes; the valid DDR prefix has 768 f32 elements,
so **each DMA transfers 3072 bytes**. X and Out have dynamic row offsets
`programId * 768` elements, proved to be `programId * 3072` aligned bytes.

| Buffer | DDR address | Padded ordinary lanes |
| --- | --- | --- |
| X | X + row × 768 | 0 |
| Weight | Weight | 1 |
| Bias | Bias | 0 |
| Out | Out + row × 768 | No DDR write |

Each load waits for DMA, then vector-copies SPM to an ordinary 1024-element
buffer. Stores vector-stage only the valid 768 elements into SPM and DMA them
to Out. Neither scalar computation nor ordinary Linalg touches SPM, and no
4096-byte user DDR access is emitted.

Both reductions still use all 1024 logical lanes. In particular, the centered
padded X lanes contribute `mean²` to variance. The compiler preserves this
original TTIR behavior and does not replace it with a 768-lane reduction.

`make adapter` generates a bridge with four typed unranked f32 descriptors
and six grid/program scalars. Descriptor sizes and offsets are elements. The
bridge checks 64-byte root alignment, nonalias ranges, capacities and strides,
then launches all 197 rows. Generated ordinary buffers live on an aligned
stack. `link.ld` puts user DDR in PMP entry 5 at
`0x80100000..0x802e0000`, and the E6 stack in entry 6 at
`0x802e0000..0x802f0000`. The stack keeps the `0x0f` write-through policy during
both DDR phases. This prevents descriptor spills from being read through a
stale cache entry after a policy switch. Stack packs are DMA sources; results
return through explicit vectors. The image links the existing DMA/cache-policy
runtime, NH/E6 boot/platform code and freestanding `nr_math.c` implementations
of `sqrtf` and `rsqrtf`. `rsqrtf` rounds the f32 square root and division
separately, with floating-point contraction disabled in the build.

`make simcheck` executes all rows in reversed order with persistent poisoned
SPM. It compares with an independent padded sequential-f32 oracle and checks
last-row and Weight/Bias tail sentinels plus the exact four DMA ranges per row.
Compiler tests also compare exactly against the unchanged experimental buffer
path; maximum absolute error against a high-precision padded formula was
2.61e-5 on the deterministic input. No arithmetic or reduction order changed.

`make check bin` also audits the complete ELF and creates
`build/layernorm_197x768/layernorm_197x768.bin`. Its launcher executes two
different inputs, all 197 rows per round in reverse order: 394 programs and
302592 output comparisons against an independent padded-f32 C oracle. SPM is
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

This independent NumPy oracle preserves all 1024 padded variance lanes and
f32 rounding. It checks every final-round output with
`2e-5 * (1 + abs(expected))`, reports exact matches and maximum absolute error,
and checks both rounds' counters plus final inputs/guards. Failures return
nonzero. On 2026-10-14, FPGA5 and FPGA6 each reported PASS for both rounds:
302592 values checked, zero errors and zero guard errors. The independent post-execution DDR check
matched all 151296 final-round values bit for bit with the sequential-f32
oracle (maximum absolute error 0). This is the measured result for these
inputs, not a general floating-point bit-identity guarantee. Against the f64
padded formula, maximum absolute error was 2.993e-5.
See [the board evidence](../../../docs/validation/experimental-boscame/README.md).

Ordinary temporary buffers and vector copies add work. Performance is
unmeasured. ELF/RVV checks establish code generation; hardware correctness
requires the execution and numerical readback checks above.

See [the compiler contract and tests](../../../docs/TritonToLinalgExperimentalBOSCAME.md).
