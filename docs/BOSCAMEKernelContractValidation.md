# Regenerated BOSCAME contract/plan/emit validation

The later [zero-accumulator and W8A8 checkpoint](validation/zero-acc-w8a8/README.md)
records the dedicated clear instruction, zero-initialization proofs, emitted
quantization chain, current compilation limits and model-accuracy gaps. The
sections below retain their original checkpoint-specific hashes and counts.

This record compares the saved integrated-pipeline baseline with regenerated
outputs from the neutral contract pipeline. All three examples complete
`make`, `make check` and `make simcheck`. Integer results are exact. LayerNorm
keeps its existing ordered f32 operations and `rtol=atol=2e-5`; the recorded
605184-byte output also matches the baseline byte for byte. These checks do
not establish board latency or validate an uploaded image.

## Routes, workspace and traffic

Counts below come from the actual example artifacts, rather than the smaller
handwritten fixtures in `plan/examples`. A mutually exclusive full/tail pair
counts once when reporting physical SPM windows. Workspace is the sum of bank
high-water marks, including gaps, rather than theoretical live bytes.

| Example | Baseline → new physical windows | Baseline → new workspace | Baseline → new surviving tasks | New bank high-water bytes |
| --- | ---: | ---: | ---: | --- |
| 196 × 768 × 768 i8 matmul | 3 → 2 | 36864 → 20480 B | 12 → 10 | 16384 / 4096 |
| 128 × 1024 × 128 i8 matmul | 0 → 0 | 0 → 0 B | 1 → 1 | 0 / 0 |
| 197 × 768 LayerNorm | 5 → 5 | 24576 → 20480 B | 19 → 21 | 12288 / 8192 |

The 196-row matmul retains private C across all six K iterations. Each iteration
stages A, reads B directly from DDR, and loads/updates/stores C through AME.
Four MMA operations per iteration remain in each full/tail branch. A staging
avoids rereading A for both N subtiles; direct B has the same external bytes as
staged B and avoids its window and DMA descriptors. No independent product P
or RVV product addition remains. Tail padding and four-row C writeback remain.

| 196-row matmul, one complete grid | Baseline | New |
| --- | ---: | ---: |
| Static DMA starts, both branches | 364 | 108 |
| Dynamic DMA starts, 42 programs | 41664 | 9408 |
| DMA read bytes | 5634048 | 1505280 |
| DMA write bytes | 602112 | 602112 |
| Direct AME B read bytes | 0 | 4128768 |
| Total external bytes | 6236160 | 6236160 |

Dynamic counts account for 36 full programs and six tail programs. A full
program now performs `32 + 6*32 + 32 = 256` DMA starts; a tail performs
`4 + 6*4 + 4 = 32`. B reads 16384 bytes in each of six iterations per program.
Descriptor reduction is not a measured hardware speedup.

The 128-row matmul folds five source tasks into one all-DDR matmul. It uses
zero SPM and DMA; its kernel assembly and restricted assembly exactly match
the saved baseline. C is read once, accumulated over two internal K subtiles
and written once per program.

LayerNorm retains four 3072-byte DMA ranges per row: X/Weight/Bias read and
output write. Across 197 programs this is 1815552 read bytes and 605184 write
bytes. The five new SPM roots are X/output, Weight, Bias, squares and a reused
mean/invstd broadcast root. The fresh layout has no baseline 4 KiB gap.
Two rank-zero host reductions, explicit scalar-load events and SPM broadcast
fills replace the baseline's scalar SSA/register broadcast forwarding. The
task counts therefore differ, and internal vector traffic is not preserved.
The complete new ELF has 1957 static instructions, 54 RVV instructions and
26 vector memory instructions, versus 1898/54/24 in the baseline; neither
contains AME instructions. The 196-row ELF has 4661 instructions versus 8354;
both have 136 encoded AME instructions. The 128-row ELF retains 1521
instructions and 34 AME instructions.

## Numerical checks

The recorded baseline and regenerated logs agree:

| Check | Programs / outputs | Result |
| --- | --- | --- |
| 196 matmul launcher input | 42 / 150528 i32 | exact, maximum absolute error 0 |
| 196 signed extremes, nonzero C, i32 wrap, tail first | 42 / 150528 i32 | exact, maximum absolute error 0 |
| 196 same stress input, reverse order | 42 / 150528 i32 | exact, same output as tail first |
| 128 matmul | 32 / 131072 i32 | exact, maximum absolute error 0 |
| LayerNorm, reverse row order | 197 / 151296 f32 | maximum absolute error 0; all 605184 bytes equal baseline |

The strict simulator retains poisoned SPM between programs and checks unchanged
inputs, DDR guards and permitted access ranges. The 196 stress launches cover
signed i8 extrema and i32 wrap. LayerNorm requires exactly the four DMA ranges
per row; its numerical acceptance remains `rtol=atol=2e-5` for the independent
sequential-f32 oracle. This finite-input byte equality does not assert bitwise
equivalence to every frontend's reduction tree or to all NaN representations.

Output SHA256 values:

```text
196 launcher: 63bfd123bec19e912c9d6726e001b8add1106c7249c1d7daa6e4d7b7ffebde49
196 stress:   2a80f3d1fc39b8b6a6cdc7eb0e8c0698e8267ec0909cace42c4d7865d60072e8
128 launcher: 38114b2648726d06482fc92bf727d2376ae93b763c1951c7df6cd16876e45cb2
LayerNorm:    ed6261ce27a36b49df83291fe18af274a0892e8db49f4a509957405cbebf8cbe
```

The 196 simulator prints these hashes directly. The 128 value hashes the
deterministic oracle whose complete output the simulator verifies exactly.
LayerNorm hashes the saved full baseline and regenerated output files.

## Linked E6 stack audit

[check_e6_stack.py](../examples/FPGA-BOSCAME/tools/check_e6_stack.py) reads linked
function symbols and disassembly, follows both sides of conditional branches,
direct calls and tail jumps, and propagates SP arithmetic, alignment and frame
pointer restoration. It counts temporary saves inserted by the AME encoder
from the final machine code. Unknown SP writes, inconsistent stack loops,
recursion and unmapped indirect calls fail the audit. Bounds may include
infeasible branch combinations; they are conservative control-flow bounds.

| Image | Recomputed baseline bound | New bound | New remaining stack |
| --- | ---: | ---: | ---: |
| 196 matmul | 2256 B | 1216 B | 64320 B |
| 128 matmul | 656 B | 656 B | 64880 B |
| LayerNorm | 992 B | 1168 B | 64368 B |

The stack symbols still provide 65536 bytes. The largest 196 chain is boot →
platform E6 main → launcher → adapter → MLIR wrapper → kernel. Its live caller
frames total `16 + 272 + 208 + 48 = 544` bytes; the kernel adds its 544-byte
frame and at most 128 bytes of encoder saves, for 1216 bytes. DMA wait and
failure hooks are also traversed, but do not exceed that bound. For 128,
`16 + 256 + 208 + 48 + 128 = 656` bytes.

LayerNorm's live caller frames total `16 + 560 + 272 + 80 = 928` bytes. Its
64-byte kernel frame, 32-byte alignment adjustment for this call chain, two
64-byte aligned scalar allocations, and 16-byte DMA wait frame give
`928 + 64 + 32 + 64 + 64 + 16 = 1168`. The baseline kernel uses a 48-byte
frame and the same 16-byte wait frame: `928 + 48 + 16 = 992`. The new bound is
measured independently and includes the increased alignment/storage cost.

For both DMA images, the audit explicitly includes the installed
`nh_e6_publish_dma_failure` and default `bosc_ame_dma_default_fail` targets at
every indirect wait failure call. Both hooks add zero stack bytes. The boot
trap handler also adds zero bytes. This target set is an explicit promise
about these fixed launchers, which install the hook before kernel execution;
the tool does not prove an arbitrary writable function pointer immutable.
No interrupt nesting or arbitrary stack corruption is modeled. The direct-only
128 image has no reachable DMA calls.

From the repository root, audit any regenerated default image with:

```bash
example=matmul_i8_196x768x768  # or matmul_i8_128x1024x128 / layernorm_197x768
python3 examples/FPGA-BOSCAME/tools/check_e6_stack.py \
  "examples/FPGA-BOSCAME/$example/build/$example/$example.elf" \
  --objdump llvm/build/bin/llvm-objdump --nm llvm/build/bin/llvm-nm \
  --indirect-call bosc_ame_dma_wait=nh_e6_publish_dma_failure,bosc_ame_dma_default_fail \
  --output stack-audit.json
```

Six regression cases cover branch/call peaks, encoder-like transient saves,
aligned allocas and frame restoration, tail calls, both failure hooks,
unknown SP/unbounded-loop rejection, and recursion rejection.

## Artifact identity and baseline reconstruction

New ELF SHA256 values, matched against the generated NR audit reports:

```text
196: 95335a64afee404cc1171d6561f669629f2b5f622960a1ee133be96067dd0948
128: 26d4b1e5e6d6732f6a6ee209b2e54a5dad6600ff849f7e4353dcf1fbd27e3f4b
LN:  818766ad1504f30b99aed59a28c08b33d2b704a7cea28b40940ce3b8f62c480a
```

The pre-change capture retained TTIR, frontend/bufferized/BOSCAME IR, kernel
assembly, restricted assembly, output/logs and NR audit hashes. It did not
retain the original ELF files. For stack comparison, the saved restricted
assembly was assembled and linked with the unchanged launcher, adapter,
boot/platform, UART and DMA runtime objects and the same linker script; LN
also uses `nr_math.o`. These are reconstructed baseline ELFs, not new-pipeline
outputs. Reproduce using a saved baseline directory in `BASELINE`:

```bash
example=matmul_i8_196x768x768
image="examples/FPGA-BOSCAME/$example/build/$example"
llvm/build/bin/clang --target=riscv64-unknown-elf \
  -march=rv64imafdcv_zicsr_zifencei -mabi=lp64d -mno-relax \
  -c "$BASELINE/$example/kernel.nr.s" -o "$BASELINE/$example/kernel.o"
ld.lld -m elf64lriscv --gc-sections \
  -T "examples/FPGA-BOSCAME/$example/link.ld" \
  -o "$BASELINE/$example/$example.elf" \
  "$image/obj/nh_e6_boot.o" "$image/obj/nh_e6_platform.o" \
  "$image/obj/uart.o" "$image/obj/bosc_ame_dma.o" \
  "$image/obj/launch.o" "$image/obj/adapter.o" \
  "$BASELINE/$example/kernel.o"
```

For LayerNorm, insert `"$image/obj/nr_math.o"` before `launch.o`, preserving
the makefile's object order. Audit the resulting ELF with the same stack tool.
The reconstructed 196 ELF exactly matches its original NR audit hash:
`e947700a825db066481d068782ede692ab0996766d0317141110aa7a534c777c`.
The reconstructed 128/LN hashes are respectively
`26d4b1e5e6d6732f6a6ee209b2e54a5dad6600ff849f7e4353dcf1fbd27e3f4b` and
`107801d2104e934efc6b53b70ba11e6f4ed04a3ea084808f527925bb7b9e97f5`.
They differ from the original recorded hashes
`4f790c53dc28e21128e122d31b97cb3823d3506c89331c31eadbd581cd68cbda` and
`988f3be44a4ca2b37ee6cc2cee3e8bc23ab80adb7c37aa0e9a163d8937599ca5`;
the original ELFs are unavailable for byte comparison. Their reconstructed
NR instruction counts and executable-section sizes match the original audit
records. Only the reconstructed files establish the baseline stack bounds
above; all new bounds come directly from the actual new linked ELFs.

## Repeating validation

Run `make`, `make check` and `make simcheck` in each example, using its Triton
Python environment. The 196 `simcheck` target enables the full stress suite.
For LayerNorm, save and compare complete outputs explicitly:

```bash
python3 examples/FPGA-BOSCAME/layernorm_197x768/simcheck.py \
  --module examples/FPGA-BOSCAME/layernorm_197x768/build/layernorm_197x768/kernel.boscame.mlir \
  --write-output layernorm-new.bin --compare-output layernorm-baseline.bin
```

Native planner/emitter tests separately cover malformed plans, direct/staged
capacity fallback, DMA units/pitches, source mappings, migration diagnostics,
both compatibility pass names and rejection of legacy options. Emit regression
tests include the complete emit→`lower-bosc-ame` chain; otherwise constant
descriptor arithmetic could pass emission but fail final SPM bounds checks.
Run their lit tests and the stack audit tests independently of example checks.
The repository-wide `check-buddy`, formatting and `git diff --check` remain
separate validation steps; this record does not substitute for their final run.

The final integrated run of `ninja -C build check-buddy -j2` completed with
239 example tests and 599 regression tests passing, one expected failure and
four unsupported tests, with no failures. Changed-file pre-commit and
`git diff --check` also passed. The design package separately passed all six
fixture checks, deterministic regeneration and 40 positive/mutation tests.
All three production examples were regenerated with the final compiler and
reran `make`, `make check` and `make simcheck`; their linked ELF hashes and
stack bounds remain those recorded above. The final LayerNorm comparison
again matched all 605184 baseline output bytes.

## Exact matrix boundary transfers

The 2026-10-25 matrix-transfer extension preserves a legal pure DMA plan and
otherwise tries an exact raw ACC32 copy before the existing segmented
DMA/RVV fallback. It changes transport only: no arithmetic, FP conversion,
AME floating-point computation or floating-point reassociation is added.

The production changes are in `MatrixTransfer.h`, `BOSCAME.td`, the plan /
verify / emit implementation, `TileOps.cpp`, `BOSCAMETileLowering.cpp`,
`SpmDmaCheck.cpp` and the strict buffer simulator. A plan explicitly records
endpoint layouts, byte pitches/offsets, ordered tiles, one ACC32, mtype,
total/temporary bytes and the selection reason. `plan-report` exposes those
same decisions and the separate SPM window/bank budget. Emission only expands
the verified tiles.

| Boundary fixture | Selection and exact byte ranges | Result |
| --- | --- | --- |
| f32 `[64,16]`, pitches 788 / 64 B | One 64-row, 64-B/row matrix tile; 4096 B in either direction | PASS |
| f32 `[64,5]`, pitches 788 / 64 B | One 64-row, 20-B/row matrix tile; 1280 B in either direction | PASS |
| Independent descriptor offsets, byte-view shifts and destination pitch | Each effective pointer is formed once; source/target pitches remain independent | PASS |
| `[65,69]` f32 | Four tiles per direction: `(64,256)`, `(64,20)`, `(1,256)`, `(1,20)` in rows/bytes | PASS |
| One 276-B row, unused parent pitches 64 / 128 B | 256-B and 20-B tiles; second byte offset 256 remains in row zero | PASS |
| Matching i8/i16/i32/i64/f32/f64 logical storage | Whole logical elements copied with raw e32 accesses | PASS |
| One module with legal DMA, matrix and true rank-one RVV | Independent automatic choices, exact completion boundaries | PASS |
| Invalid raw32 base/row/pitch geometry | Existing legal fallback, or the specific no-16-bit-RVV diagnostic | PASS |

`PlanMatrixTransfers.py` runs actual annotate → plan → verify → emit →
SPM/DMA check → LLVM lowering. Full, tail and mixed-compute fixtures also run
`buddy-translate`, optimized `llc`, the fixed-register AME encoder, strict NR
assembly restriction and target-object assembly. The generated bundle keeps
`fence`, tile sizes, mtype, `mlce32.m acc0`, `msce32.m acc0` and the final
fence together, with ACC0 and memory clobbers. No runtime call or exposed
matrix value exists inside the pair.

The independent simulator compares raw bytes, including positive/negative
zero and infinity, normal/subnormal values, qNaN and sNaN payloads. It checks
each valid external byte exactly once per direction, unchanged guards and
readonly inputs, independent offsets/pitches, column-major physical layouts
and repeated poisoned-SPM runs. Plan mutations reject missing, reordered,
overlapping or incomplete tiles, wrong layouts/configuration, byte accounting
and completion. Standalone checker/lowering negatives cover pending DMA,
live TR, ACC and loop-carried matrix values. `matrix-copy-bounds.mlir` adds
13 split-input sections for actual bank/root spans, view shifts, tile offsets
and overflow. Existing DMA and direct compute checks are retained.

The real `attention_qk_12x197x197x64` frontend was exported again. Its original
788-B input boundary now receives `engine="matrix"` and
`reason="dma-illegal-exact-acc32-rectangle"` before planning stops at its
independent private-to-private initialization copy. A read-only input slice
extracted from that actual bufferized module, with QK arithmetic and output
stores removed only for isolation, passes plan/verify/emit/lower. All four
full/tail branches select matrix input copies. Strict simulation of five
representative programs, including both tails and the last head, verifies
exact input ranges, readonly DDR/guards and zeroed B padding. This is input
transport evidence, not a successful QK numerical or complete-model test.
That describes the boundary-extension checkpoint; the subsequent QK storage
normalization and explicit internal-copy implementation are recorded below.

All three existing standalone examples were rebuilt and reran `make check`
and `make simcheck`. Their frontend through kernel assembly artifacts are
byte-identical to the saved pre-extension baseline. The 128-row matmul checks
32 programs / 131072 i32 outputs; the 196-row case checks three complete
42-program grids / 150528 outputs each, including signed extremes, nonzero C,
i32 wrap, tail-first and reverse order. LayerNorm checks 197 programs /
151296 f32 outputs at unchanged `rtol=atol=2e-5`, with maximum absolute error
zero. Its four exact 3072-B DMA ranges per row are unchanged. Linked stack
bounds remain 656 B, 1216 B and 1168 B respectively, below the unchanged
65536-B capacity.

Final repository checks pass: 239 example tests, 604 regression tests, one
expected failure and four unsupported tests, with no unexpected failures.
Task-scoped pre-commit and both working-tree/index diff whitespace checks
pass. Reproduce the transport tests with the configured Python environment:

```bash
python tests/Conversion/LowerLinalgToBOSCAME/PlanMatrixTransfers.py /tmp/matrix-copy-tests
python tests/Conversion/LowerLinalgToBOSCAME/PlanBoundaryTransfers.py /tmp/boundary-copy-tests
ninja -C build check-buddy -j2
```

Hardware evidence remains the ordinary matrix-memory byte-stride contract in
the supplied specification and NR reference sources. There is no new board
run, arbitrary-unaligned-base claim, other-width matrix-store support, CSR
capacity probe or hardware performance result. Effective bases, offsets,
pitches and row bytes require 4-B alignment; one tile is conservatively
bounded by 64 rows / 256 B per row / 16384 B. f16/bf16 are explicitly excluded
from this closed matrix-copy type set. Existing full-line SPM/padding
initialization is preserved; this extension does not establish the behavior
of uninitialized partial SPM lines.

## QK storage normalization and internal copies

The later implementation separates two changes that need independent
evidence. In QK, a tensor storage normalization replaces an existing floating
add's zero-splat tensor read with its exact scalar constant and destination-
passes the single-use matmul result. The helper clones the original
`arith.addf` operand mapping, signed-zero bits and fastmath flags; existing
canonicalization can then put the constant on the right. The add remains
after the zero-initialized matmul. `MatmulConverter::skipC` is unchanged; this
is neither floating accumulator fusion nor permission to discard `+0`.

| Actual QK frontend, all four full/tail branches | Before → after |
| --- | ---: |
| Private allocations | 20 → 16 |
| Copies | 16 → 12 |
| Private product-initialization copies | 4 → 0 |
| Contract tasks | 40 → 36 |
| Matmuls / floating adds | 4 / 4 → 4 / 4 |
| Planned bank high-water bytes after normalization | 5120 / 5120 |

The regenerated QK completes frontend, neutral verification, planning, plan
verification, emission, target lowering, LLVM translation and RISC-V object
assembly. `build-result.json` records `PASS` at `stage="asm"`; this is a
compilation result, separate from full-grid simulation. The portable
[final build report](validation/qk-storage-internal-copy/qk-final-build-report.json)
and [artifact/source hashes](validation/qk-storage-internal-copy/artifact-source-hashes.json)
identify that exact output, including its byte-identical emitted simulation
input. The old and new real
branch paths were compared at program coordinates `(0,0,0)`, `(0,12,0)`,
`(12,0,0)` and `(12,12,0)`. The saved
[old/new frontend comparison](validation/qk-storage-internal-copy/old-new-branch-report.json)
and [old/emitted comparison](validation/qk-storage-internal-copy/old-emitted-branch-report.json)
record checks of the complete 465708-element C array after those four programs.
The programs write 441 elements: 409 non-NaN results match bit for bit and 32
NaN results have matching classification. The other 465267 elements retain
their initial C values. Exact access rectangles, unchanged inputs and DDR
guards also pass. NaN classification equality does not claim NaN payload
equality for arithmetic.
The [complete-grid report](validation/qk-storage-internal-copy/qk-full-simcheck-report.json)
records a completed strict simulation of all 2028 programs in reverse order.
All 465708 f32 outputs match the independent ordered-f32 oracle bit for bit,
with maximum absolute error zero. The same Machine keeps poisoned SPM across
the whole grid; readonly A/B, exact logical DDR bounds and guards pass.
This is one complete normal-input run, separate from the four-program
signed-zero/NaN/Inf comparison. It does not establish full-model accuracy
or hardware performance.
The [evidence README](validation/qk-storage-internal-copy/README.md) retains
small real before/after IR excerpts, the explicit scalar-broadcast plan and
[storage counts](validation/qk-storage-internal-copy/storage-change-report.json).

Genuine internal logical-buffer copies have a separate
`role="internal",engine="matrix",recipe="spm-acc32-spm"` plan. It records
both `source_endpoint/target_endpoint` dictionaries (operand, bank, window
offset, capacity and complete physical envelope), independent layouts and
strides, ordered raw ACC32 tiles, `temporary_window="none"`, zero temporary
bytes, valid byte total and the final synchronous completion. The recorded
reason is `nonoverlapping-spm-exact-acc32-rectangle`. The verifier recomputes
the endpoint ranges and coverage from actual operands and placements.

Each endpoint must belong to a distinct bounded private SPM root. The full
rectangular envelope, including pitch holes, must fit its window and bank;
same-bank envelopes must not overlap. Disjoint aliases of one root remain
unsupported. Eligible static rank-two i8/i16/i32/i64/f32/f64 storage shares a
physical unit axis and permits exact four-byte-aligned raw32 accesses. No
additional SPM window is allocated. Each tile temporarily uses ACC0 only
inside its fence/configuration/load/store/fence bundle; no physical TR/ACC
format conversion or cross-operation/register residence is introduced.

`PlanInternalCopies.py` runs real annotate → plan → verify → emit → checker
→ lowering and strict raw-byte simulation. It independently checks all six
logical types, NaN payloads, signed zero, infinities, independent descriptor
offsets and pitches, axis-zero layouts and a multi-tile `[65,69]` rectangle.
The source is read again after the copy, poisoned-SPM runs preserve untouched
padding and guards, and access tracking checks both owned windows. Negative
plans cover malformed recipe, endpoint, layout, tile, register configuration,
temporary storage and completion; alias/overlap, window/bank capacity,
unsupported rank/type, live ACC and pending DMA remain rejected. Standalone
`internal-matrix-copy.mlir` also verifies the physical dual-SPM operation.

All three existing standalone examples again complete `make`, `make check`
and `make simcheck`. The portable
[existing-example summary](validation/qk-storage-internal-copy/existing-examples-summary.json)
records successful
linked-ELF and stack audits: 656 B for the 128-row matmul, 1216 B for the
196-row matmul and 1168 B for LayerNorm, each below the unchanged 65536-B
capacity. Integer programs retain exact signed/wrap stress results; LayerNorm
retains its ordered f32 operations and `rtol=atol=2e-5`. These checks do not
change the earlier three-example numerical acceptance criteria.

The [verification checkpoint](validation/qk-storage-internal-copy/verification-checkpoint.json)
records the later full `check-buddy`: 239 example tests with two unsupported,
609 regression tests with two unsupported and one expected failure, and no
unexpected failures. All 40 design checks pass. Internal copies with dynamic
type-level descriptor offsets are rejected during planning and by independent
verification/emission, before target materialization, rather than receiving
an unproved physical range.

Reproduce the independent internal-copy regression with:

```bash
python tests/Conversion/LowerLinalgToBOSCAME/PlanInternalCopies.py /tmp/internal-copy-tests
build/bin/buddy-opt tests/Dialect/BOSCAME/internal-matrix-copy.mlir \
  --split-input-file --verify-diagnostics --bosc-ame-spm-dma-check -o /dev/null
```

The QK normalization does not add K-loop tail specialization, loop invariant
capture, indexed/scalar external memory access or new mathematical recipes.
Hardware evidence remains the cited matrix-memory contract and reference
programs; this checkpoint includes no new board execution, arbitrary TR/ACC
interchange, register-residency guarantee or hardware performance result.
