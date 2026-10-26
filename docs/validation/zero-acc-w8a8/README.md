# Zero accumulators and emitted W8A8 Linear validation

This checkpoint covers the new accumulator clear and the existing ViT dense
Linear rewrite. It separates target encoding, emitted-kernel simulation,
graph/ABI checks and model accuracy. There is no new board run or hardware
performance measurement.

## Semantics and generated code

`bosc_ame.tile.zero_acc` has no input accumulator or memory operand. Its
lowering uses the restricted same-register `mwsub.h.mm` form, clear mode
`0x21`, explicit M/N and fences. Subsequent signed-i8 MMA uses its existing
`0x10010` mode. The encoder, dialect verifier, production ISA validator and
strict simulator reject general accumulator subtraction. Existing board
evidence covers Acc0–Acc7 and 1×1/2×3/8×8; larger production tiles are covered
by simulation, including 32×64 and 64×64. See
[the zero-accumulator contract](../../BOSCAMEZeroAccumulator.md).

Before contract graph construction, a proven immutable zero-copy becomes a
local fill. This preserves exact zero bits, coverage and memory ordering;
V19 remains enforced. A private i32 matmul initialization can then receive
`deferred-zero` and a reciprocal initialization task link. Verification
re-proves its aliases, observers, lifetime and complete coverage. For a
statically nonempty external K loop, only `iv == lowerBound` clears; later
iterations load the stored partial sum. Each branch completes its own
clear/load, MMA and store before the next DMA/runtime call. Padding needed
for input or output transfers is preserved.

The default ViT replacement reuses the existing W8A8 graph transformation:
73 dense calls become 219 quantize/overwrite-i8-matmul/dequantize calls.
The explicit `--fp32-reference` mode retains the floating dense path; its
ordinary patch projection is still the pre-existing quantized composition.
It is therefore not a pure FP32 HuggingFace baseline. The generic frontend
does not silently quantize floating matmuls.

Five dedicated overwrite kernels express zero initialization in neutral IR.
Their adapters neither initialize nor read DDR C. The original five
accumulating kernels retain `C += A*B`, including nonzero C and i32 wrap.
Bias and residual additions execute after ordered dequantization.

Quantization uses per-row activation and per-output-channel weight scales,
physical `[N,K]` weights, signed `[-127,127]`, and scale 1 for an all-zero row.
Rounding truncates the clamped f32 value, tests its fractional magnitude
against 0.5 and applies a signed correction. This implements ties away from
zero without rounding a value immediately below 0.5 upward during an
intermediate f32 addition. Casts and saturation run in emitted kernels.
Dequantization retains `(float(acc) * row_scale) * column_scale` order.

Dynamic scalar operands, singleton boundary snapshots, ordered max-number
reduction, common-lane mixed-width pointwise expressions and proven
row-broadcast/contiguous scale access are explicitly represented. Exact f32
absolute value clears the sign bit; it does not scalarize to `fabsf` calls.
The existing f32 subtract expansion and a new restricted immediate-merge
expansion generate only allowed instructions. The merge saves/restores its
scratch GPR in a 16-byte temporary stack slot; the ISA whitelist is unchanged.

## Numerical and structural evidence

The portable JSON files in this directory identify source and generated
artifact hashes. The classifier-chain report executes candidate operations
in emitted IR; NumPy only prepares inputs and supplies independent oracles.

- [All 40 compilation results](vit-build-report.json): 37 objects, three
  compilation blockers, with hashes for every generated stage.
- [Production ISA audit](vit-production-object-audit.json): all 37 objects
  pass, including adjacent AME fences and the unchanged instruction whitelist.
- [Strict simulation results](vit-simcheck-report.json): 31 passing cases,
  17485 programs and 11211403 output comparisons; six numerical runs explicitly
  not executed, plus the three compilation blockers.
- [Four-round classifier chain](classifier-chain-report.json),
  [graph rewrite](graph-rewrite-report.json) and
  [source identities](source-hashes.json).
- [All 40 native ABI smoke checks](abi-smoke-report.json),
  [model summary](model-summary.json) and
  [unverified model accuracy](accuracy-status.json).
- [Generated zero object](zero-generated-object-audit.json) and
  [standalone examples and stacks](existing-examples-report.json).

| Check | Completed scope | Result |
| --- | --- | --- |
| Zero target tests | Poisoned contents, repeated clear, clear after MMA, subsequent MMA, tails and guards | PASS |
| Zero planner tests | Nonzero loop lower bound, multiple K steps, padded 4×40 tail, aliases/observers, incomplete tasks and malformed plans | PASS |
| Default graph rewrite | 73 dense rewrites, 219 external calls, rejection of accumulating-kernel fallback | PASS; structural/ABI only |
| Native launcher smoke | All 40 cases | PASS; no numerical execution |
| Six quantizers | 2359 complete-grid programs | Exact quantized values and f32 scales |
| Five overwrite matmuls | Three rounds each, 1812 programs total | Exact i32 oracle, independent of old C |
| Classifier Linear chain | Four rounds, including persistent dirty workspace, changed token scale, all-zero activation and channel, forward/reverse/tail-first order | Exact integer and ordered quantized-f32 oracles |
| Four dequantizers | Complete grids | Exact ordered f32 oracle |
| Floating matmul layout regressions | Column-major and general-strided small matrices | Exact ordered f32 and LLVM translation |

The chain includes real activation and weight quantizers, i8 matmul,
dequantization, bias and residual kernels. It reports zero external old-C
reads. The first and repeated-workspace rounds have identical output hashes.
Its synthetic stress inputs deliberately use widely varying channel scales;
the reported difference from ordered floating matmul is quantization error,
not an accuracy acceptance threshold.

Both original standalone matmuls and LayerNorm were rebuilt and ran
`make check` and `make simcheck`. The 196-row matmul checks all 42 programs
and 150528 outputs in normal, signed-wrap/tail-first and reverse rounds.
The 128-row matmul checks 32 programs and 131072 outputs. Integer results
remain exact. Standalone LayerNorm checks all 197 rows with maximum error
zero at its existing `rtol=atol=2e-5`, exact DMA ranges and guards. Fresh
linked-image stack bounds are 1216, 656 and 1168 bytes respectively, below
the unchanged 65536-byte stack.

## Remaining limits

37 of 40 cases compile through strict assembly processing and RISC-V object
generation. The three actual remaining compilation blockers are Attention
PV's dynamic K=197 tail (`V4`), GELU's `math.erf`, and Softmax's `math.exp`.
Five large floating reference grids and the full patch composition have not
completed numerical simulation in this checkpoint; those are validation gaps,
separate from compilation restrictions. Small floating-layout regressions
do not substitute for the five complete grids.

The complete ViT model and top-1/top-5 accuracy are unverified. The
[comparison tool](../../../examples/FPGA-BOSCAME/vit-base-patch16-224/tools/compare_accuracy.py)
prepares identical checkpoint/preprocessing/sample evidence and compares
logits, prediction agreement and labeled top-1/top-5. Missing model execution
or data produces an explicit unverified result. Quantization changes the
model's numbers; no model-level accuracy acceptance is inferred from exact
agreement with the quantized kernel oracle.

## Reproduction

Use the configured Python environment containing Triton and the built Buddy
tools. From the repository root:

```bash
ninja -C build bin/buddy-opt bin/buddy-translate -j2
python tests/Conversion/LowerLinalgToBOSCAME/PlanZeroAccumulator.py /tmp/zero-plan
python tests/Conversion/LowerLinalgToBOSCAME/PlanConstantBroadcasts.py /tmp/constant-plan
plan/tools/check_examples.sh
ninja -C build check-buddy -j2
```

The ViT [validation instructions](../../../examples/FPGA-BOSCAME/vit-base-patch16-224/VALIDATION.md)
describe case builds, strict simulation, the emitted classifier-chain target
and the reproducible accuracy protocol. Generated artifacts remain build
outputs; this directory retains compact reports and hashes.

The final [verification checkpoint](verification-checkpoint.json) records
239 passing example tests and 619 passing regression tests, one expected
failure, four unsupported tests and no unexpected failures. This run includes
the direct-constant-yield broadcast regression and the final target fixes.
All six design fixtures, 40 design checks and deterministic regeneration
pass. Task-scoped pre-commit and both working-tree/index whitespace checks
pass. No files were staged or committed by this work.
