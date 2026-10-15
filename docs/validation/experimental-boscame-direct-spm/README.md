# Experimental BOSCAME direct-SPM validation

Validated on 2026-10-15. This extends the [earlier implementation](../experimental-boscame/README.md):
`tt.load` now leaves data in SPM, and its computing consumers generate the
required register accesses inside the experimental pass. The original TTIR
arithmetic, masks, `other`, reduction sizes and independent oracles are retained.

| Kernel | Board image | Programs in two rounds | Board values checked | Independent final-DDR comparison | Input/guard errors |
| --- | --- | --- | --- | --- | --- |
| LayerNorm 197×768 | FPGA6, NR patch9, 5.5296 MHz | 394 | 302592, all exact | 151296, all bit-identical to sequential f32 | 0 |
| GEMM 128×1024×128 | FPGA4, NR patch7, 5.5296 MHz | 1024 | 262144, all exact | 131072, all exact i32 | 0 |

Both runners and independent DDR checkers returned zero. The runner used its
current default hardware image; the compiler change did not modify board
configuration. The two hardware revisions are recorded separately and are
not used for a latency comparison.

## Generated data flow

LayerNorm has exactly three DDR-to-SPM DMA operations and one SPM-to-DDR DMA
operation per program, each followed by a wait and each transferring exactly
3072 bytes. Its logical 1024-lane SPM buffers retain the original padded
X/Weight/Bias values 0/1/0. Both reductions still include all 1024 lanes.
Elementwise consumers use one-dimensional 64-byte vector operations directly
on SPM. The final producer's SPM buffer is the DMA store source. There is no
ordinary full-tensor load bounce. Five static views use 12288 bytes in bank 0
and 8192 bytes in bank 1; ordinary rank-zero accumulators remain on the stack.

For the required integer GEMM, A has strides `[64,1]`, and logical B has strides
`[1,64]`, corresponding to physical `[N,K]`. Both DMA load buffers are consumed
directly by `tile.load_a/b`. Their ordinary A/B pack and duplicate DMA are
removed. Initial C and dot outputs still use ordinary helper buffers and
explicit vector copies; accumulator registers are materialized at each dot
boundary. General K blocking/layouts can still require consumer-owned vector
snapshots followed by ordinary packing. Nonzero C is added exactly once.

Floating matmuls retain standard Linalg arithmetic. When their operands are
SPM buffers, the matmul consumer snapshots bounded physical intervals through
one-dimensional vectors into ordinary storage, then writes an SPM result back
through vectors. Tests cover both physical B layouts and nonzero original C.

SPM placement and computation lowering occur after one-shot bufferization,
before the strict exit checks. Supported consumers are static rank-one/rank-two
fills, identity-map parallel elementwise computations, vector copies, f32
rank-one sums, integer AME matmuls and the floating matmul boundary above.
Other SPM consumers fail explicitly. Unsupported layouts, short vector tails,
unprovable offsets/masks, unsafe DMA lengths and exhausted workspace budgets
also fail. Scalar SPM accesses, ordinary copies/Linalg accessing SPM, unresolved
integer targets, tensor bridges and unmatched DMA tokens remain prohibited.
The [compiler contract](../../TritonToLinalgExperimentalBOSCAME.md) lists the
supported and rejected memory access forms.

## Source-order f32 reduction

An initial candidate used LLVM's ordered f32 vector reduction. On the tested
patch7 board, all 151296 final outputs instead matched an independent model
that accumulates each 16-lane packet in higher precision and rounds once per
packet. Only 11849 values were bit-identical to the sequential-f32 reference;
maximum absolute difference was 3.1948089599609375e-5. This observation does not
establish the RTL implementation. The candidate met the unchanged tolerance,
but was replaced to preserve per-addition f32 rounding.

Extracting every lane of a wider vector generated `vslidedown.vi`, which the
retained instruction audit rejected. The final implementation reads
`vector<1xf32>` from SPM, extracts lane zero and performs the original scalar
f32 addition with an `scf.for` accumulator. Actual code uses
`vle32.v / vfmv.f.s / fadd.s`, with no vector reduction, slides, scalar SPM
load or policy calls inside the sum loop. The existing instruction allowlist
was not relaxed. Other SPM vector transfers remain 64 bytes.

The final patch9 execution is bit-identical to the independent sequential-f32
oracle for every checked value. Against the independent f64 formula retaining
the TTIR's 1024-lane padding, final-round maximum absolute error is
2.992675160307101e-5. Floating-point contraction is disabled. These results
apply to the tested inputs and freestanding f32 sqrt/division implementation;
they are not a universal bit-identity claim for arbitrary math implementations.

The single-lane sum trades vector throughput for source rounding semantics.
Static SPM intermediates increase workspace consumption, and general matmul
packing, C materialization and floating matmul snapshots still add movement.
No controlled baseline speedup measurement was made. UART cycles are totals
over two launches, without division by two, retained only as diagnostics.

## Execution evidence and final-source binding

Each run poisons SPM once and executes two different input rounds without
clearing it. Checks cover every output, unchanged inputs, before/after DDR
guards, the last row and Weight/Bias tails. LayerNorm runs rows in reverse
order; GEMM includes signed-i8 extrema, asymmetric B and i32 wrap.

After the application verdict, 3 MiB of actual DDR was captured starting at
`0x80000000`. The checker resolves addresses from the exact linked ELF and
recomputes every final-round value independently. A pre-execution loader
mirror is not used as result data.

- LayerNorm: [report](layernorm-fpga6.json), [UART verdict](layernorm-fpga6.uart.txt).
- GEMM: [report](gemm-fpga4.json), [UART verdict](gemm-fpga4.uart.txt).

| Uploaded binary | SHA256 |
| --- | --- |
| LayerNorm | `434d941b8116548f9db5cc66e82e5c70a0807ac46c81675bd3116cb06204a2e0` |
| GEMM full board check | `3a46d5b9e226dd27a0b919d2e4a1bc199d3ea4302a402e5b7bbebbaf0851b784` |

The reports retain ELF and dump hashes. The final compiler regenerated
LayerNorm's BOSCAME/LLVM/assembly byte for byte identically to the executed
image's inputs ([comparison](layernorm-final-equivalence.json)). GEMM's complete
final rebuild, including ELF and binary, is byte-identical to its executed
image ([comparison](gemm-final-equivalence.json)). The retained NR audits passed:
LayerNorm 1953 instructions/49 RVV/0 AME, GEMM 3399 instructions/117 RVV/21 AME.
These audits supplement execution and numerical comparison.

The GEMM board artifact uses the full checking launcher from commit `5a980622`
(SHA256 `3bd9d296f26f269c0594c8367c415477097ee822f38ee5aa26b046a31b212e26`).
It was built separately because the local working-tree launcher has its two
verification calls commented out. That local edit was preserved. A binary
with those checks disabled must not be described as a complete board check.

## Reproduce

```bash
build/bin/buddy-opt kernel.ttir \
  --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias" \
  -o kernel.boscame.mlir
build/bin/buddy-opt kernel.boscame.mlir --lower-bosc-ame -o kernel.llvm.mlir
build/bin/buddy-translate kernel.llvm.mlir --buddy-to-llvmir -o kernel.ll

make -C examples/FPGA-BOSCAME/layernorm_197x768 check bin simcheck
make -C examples/FPGA-BOSCAME/matmul_i8_128x1024x128 boardcheck simcheck
# Use a full-check launcher for GEMM, as described above.
# After uploading the chosen binaries, from the platform runner directory:
./run.sh --fpga=6 test=test/layernorm_197x768.bin
./run.sh --fpga=4 test=test/matmul_i8_128x1024x128.boardcheck.bin
```

Use the Triton Python environment and in-tree tools. Post-execution readback
and checker commands are in the [pipeline documentation](../../TritonToLinalgExperimentalBOSCAME.md#board-execution-and-post-execution-readback).
The raw binary/dump artifacts are retained outside the repository rather than
as large fixtures. Reports and verdict excerpts contain no machine paths.

The [changed-file inventory](changed-files.md) lists this revision's source,
test and documentation responsibilities. Earlier validation reports remain
historical evidence for the previous staging path.

## Regression checks

`ninja -C build buddy-opt` and the final `ninja -C build check-buddy` returned
zero: 239 examples and 426 regression tests passed, with two unsupported tests
in each suite and one expected regression failure. Changed-file pre-commit
also passed. The new SPM computation contract fails with the saved pre-change
compiler, which rejects its SPM Linalg consumers.

The retained full LayerNorm test executes all 197 programs and compares with
both the default experimental path and a separate sequential-f32 TTIR oracle.
The full GEMM and K-blocked/masked tests retain independent integer/wrap
oracles and DDR access bounds. New helper tests cover ordered cancellation,
integer wrap, direct and packed tile layouts, floating dots, invalid layouts,
runtime-math rejection and failure exit codes. Existing DMA wait, scalar-SPM,
alignment, capacity and default/legacy regressions all remain enabled.
