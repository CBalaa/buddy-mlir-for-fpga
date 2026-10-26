# GELU and Softmax emitted-execution checkpoint

Both real ViT Triton sources pass neutral-contract lowering, contract
verification, planning, plan verification, emission, LLVM lowering, restricted
NR assembly and RISC-V object generation. The algorithms remain erf-form GELU
and max-shifted Softmax. Their explicit numerical authorizations permit the
backend polynomial recipes described in the separate
[coefficient generation and error analysis](../nonlinear-math/README.md).
Other erf/exp uses receive no implicit approximation authorization.

A fresh complete-inventory assembly/object build passes all 40 cases. SHA256
comparison of all 418 `kernel.*` artifacts belonging to the previously
supported 38 cases finds no differences. The two nonlinear emitted modules
and objects also match the artifacts used for this execution checkpoint.
The archived complete build report and unchanged-artifact comparison record
these compilation checks separately from the numerical evidence below.

## Complete strict simulation

[simcheck-report.json](simcheck-report.json) retains every selected program,
round, output hash, transfer count, source hash and per-shard numerical check.
It is compact JSON to retain the complete evidence without a large pretty-
printed file. No program or output metadata was removed.

| Case | Programs per complete grid | Outputs per grid | Three-round programs | Three-round output comparisons | Maximum finite absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| gelu_197x3072 | 4728 | 605184 | 14184 | 1815552 | 4.4334679838442526e-7 |
| softmax_12x197x197 | 2364 | 465708 | 7092 | 1397124 | 1.6072830041902186e-7 |

Both cases pass their existing `rtol=atol=2e-5` criteria against the original
ordered-f32 references. They additionally pass the same criteria against
independent binary64 mathematical references using stable erfc for GELU's
negative tail, and exp with math.fsum for Softmax. The table reports errors
against the latter reference. The maximum fractions of the permitted
`atol + rtol*|reference|` error are 0.004676 and 0.006466 respectively.

Each case uses 48 contiguous program shards. One strict Machine per shard is
poisoned once and retains its SPM through normal-forward, stress-reverse and
identical-stress-forward rounds. Every round's disjoint union must contain
the complete actual grid and output. The parent reconstructs results from
worker-returned actual output values and never fills them with oracle data.
Order is defined within each shard; no global inter-shard order or
single-Machine global-order result is claimed.

The original DDR/SPM/REG access and instruction checks remain enabled.
Additional per-program audits require exactly one read and write of every
authorized logical element, reject cross-program accesses and check
unchanged output bits outside a selected subset. DDR guards and unchanged
readonly X are checked. GELU reads and writes 2420736 bytes each per grid;
Softmax reads and writes 1862832 bytes each per grid. Reverse and forward
stress output hashes match for each case.

`nonlinear-outputs.npz` in each build directory contains all three actual
output arrays. The report records its SHA256. Collection requires the current
module, kernel and simulator source hashes, checks each stored array's hash,
recomputes both references and numerical metrics, and verifies exact per-shard
identity, program sequence and DDR coverage. An absent archive cannot be
replaced by a PASS label or reference hash. These generated NPZ files remain
build artifacts; reproducing collection after cleaning the build directory
requires rerunning the strict simulator.

## Independent high-precision checks

[numerical-output-check.json](numerical-output-check.json) checks the actual
stored outputs with mpmath at 80 decimal digits. Each round samples 6482 GELU
elements and 9456 Softmax elements. Selection includes a complete GELU stress
pattern period and stratified positions, or complete Softmax rows including
all sixteen stress patterns and stratified rows. The binary64 reference's
maximum differences from these samples are 1.7763568394002505e-15 for GELU
and 5.551115123125783e-17 for Softmax.

The GELU patterns cover [-12,12], every erf polynomial join and neighboring
f32 input, negative tails, NaN, infinities, signed zeros, tiny/subnormal and
maximal finite inputs. Softmax covers all-equal rows, positive/negative common
offsets, range [-200,0], dominant probabilities, normal/subnormal exp and ln(2)
reduction boundaries, extreme finite subtraction, mixed infinities, NaNs and
the actual 197-valid/59-masked lane shape.

Nonfinite classes follow the source expression: GELU(-Inf) is NaN and
GELU(+Inf) is +Inf; negative zero is retained. Any NaN or +Inf in a Softmax row,
or a row of all -Inf, yields an all-NaN row. With a finite maximum, -Inf
elements yield positive zero probabilities. The report counts these classes
separately; finite-error metrics do not hide failed nonfinite comparisons.

These are measured samples and full-grid observations. The independently
reproducible analytic bounds in the coefficient record include coefficient
quantization, range reduction and independent f32 multiply/add rounding;
they are separate evidence and do not establish correct rounding, one-ULP
accuracy, SFU accuracy or model accuracy.

[probe-report.json](probe-report.json) records the earlier selected-program
checks (15 GELU and 48 Softmax launches). They are partial checks and are not
used as full-grid evidence. Their durations measure interpreter cost only.

## Reproduction

[build-report.json](build-report.json) archives the fresh 40/40 assembly/object
build. [existing-artifact-comparison.json](existing-artifact-comparison.json)
checks all 418 artifacts of the previously passing 38 cases against the
[captured baseline](existing-artifact-baseline.json); every hash is unchanged.
This proves generated-code preservation for those cases without claiming a
new numerical simulation of all of them.

[artifact-hashes.json](artifact-hashes.json) binds this checkpoint to the
current implementation, original Triton sources and generated artifacts.
[production-object-audit.json](production-object-audit.json) independently
audits every executable object byte against the restricted NR ISA and memory
fence rules: GELU has 675 instructions and Softmax has 448, with no FMA or
host erf/exp call. The kernel/wrapper stack bounds are 736 and 208 bytes.
GELU's external DMA runtime, startup and trap stacks are outside this
relocatable-object audit and are explicitly excluded.

The final `ninja -C build check-buddy -j2` passes 239 example tests and 629
regression tests, with 2 unsupported tests in each suite and 1 expected
regression failure. The seven new regressions cover mathematical contracts,
forged plans, actual emitted arithmetic and restricted assembly, coefficient
certificates, max reduction, IEEE simulator arithmetic, encoder classification
and strict numerical evidence collection. Changed-file pre-commit and
`git diff --check` pass.

From the repository root, use a Python environment containing Triton,
triton-riscv, NumPy and mpmath:

```sh
VIT=examples/FPGA-BOSCAME/vit-base-patch16-224
PYTHONPATH=build/python_packages python -B "$VIT/tools/build.py" \
  --case gelu_197x3072 --case softmax_12x197x197 --stage asm
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B "$VIT/tools/simcheck.py" \
  --case gelu_197x3072 --case softmax_12x197x197 --nonlinear-workers 48 \
  --report "$VIT/build/nonlinear-simcheck-report.json"
python -B "$VIT/tools/simcheck.py" \
  --case gelu_197x3072 --case softmax_12x197x197 --collect
python -B "$VIT/tools/check_nonlinear_numerics.py"
python -B "$VIT/tools/build.py" --all --stage asm
python -B docs/validation/nonlinear-operators/capture.py \
  --baseline docs/validation/nonlinear-operators/existing-artifact-baseline.json \
  --object-audit docs/validation/nonlinear-operators/production-object-audit.json
python -B tests/Conversion/TritonToLinalgExperimental/vit-nonlinear-oracle.py
python -B tests/Conversion/TritonToLinalgExperimental/vit-nonlinear-collection.py
```

These checks execute emitted IR and audit target assembly/object generation.
They do not execute the assembly, run an FPGA or establish complete-model
inference or accuracy. FP32 reference full-grid and composed patch validation
remain separate gaps; they are not compilation blockers for these operators.
