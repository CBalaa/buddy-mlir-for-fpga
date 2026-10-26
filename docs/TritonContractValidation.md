# Triton Contract migration validation

The original paired evidence below records the frontend split before the
subsequent example ownership move. Its source digests describe that frozen
state; relocation verification is recorded separately in
[BoscameExampleRelocation.md](BoscameExampleRelocation.md).
The shared package was subsequently removed; current ownership and build
commands are described in [TritonContract.md](TritonContract.md). The package
test counts below remain a record of the earlier source state.

This record covers the frontend split described in [TritonContract.md](TritonContract.md).
The machine-readable [paired evidence](validation/triton-contract-migration/results.json)
contains source identities, test discovery counts and per-artifact SHA256 comparisons.
Generated logs and binaries are kept under `build-contract-migration/` locally;
that directory is not a source dependency.

## Baseline and comparison

| Source | Fixed baseline |
| --- | --- |
| Buddy | `cb75dfc5bc2bf5c709bb00d31435acd415b75a33` |
| triton-riscv | `a305fb6ddec2401183cf7d1af878a5d1b6ee3553` |
| Nested Triton | `58895270e6230491acc15fe7ba6d4c849e838db1` plus the frontend's patch stack |
| LLVM/MLIR | `f93f13e84b81d093b665f7a59e1ae6602956b7ac` |

Both baseline and migrated tools were built against the same LLVM 24 development
build with assertions. A stale frontend cache referring to a missing LLVM build
was excluded. LLVM source was not modified. Replaying all seven managed Triton
patches onto the fixed revision reproduced all 23 affected files byte for byte.
The evidence records the modified repositories separately; no commits or new
submodules are required to identify this paired source state.

The three independent examples compare seven products each: neutral Contract,
planned IR, emitted BOSCAME, LLVM dialect, LLVM IR, restricted assembly and object.
All 21 comparisons are byte-identical. The ViT inventory compiles all 40 cases
through restricted assembly and objects; all 240 comparisons across its six
recorded stages are byte-identical. No golden files were updated to hide code
changes. Final frontend manifests were refreshed after the last build and the
same comparisons repeated, binding simulator evidence to identical emitted IR.

## Build and interface checks

The fresh Buddy core build includes C API and Python registration and runs with
a Python environment without Triton. The frontend links its local neutral
package and standard Triton/LLVM/MLIR libraries, with no Buddy include, library
or tool dependency for AOT export. AOT tests disable active-driver construction
and give the frontend an unavailable Buddy tool directory.

| Suite | Discovered | Passed | Expected failure | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| Standalone shared package | 11 | 11 | 0 | 0 |
| Buddy shared package | 11 | 11 | 0 | 0 |
| Frontend shared package | 11 | 11 | 0 | 0 |
| Frontend lit | 334 | 323 | 9 | 2 |
| Buddy core lit | 555 | 553 | 0 | 2 |
| Buddy example lit | 241 | 239 | 0 | 2 |
| Explicit cross-repository integration | 16 | 16 | 0 | 0 |
| Existing CPU swap and IME Python tests | 2 | 2 | 0 | 0 |

The AOT API test covers sole-backend discovery, CPU dispatch, AST/TTIR
normalization, SymbolTable renaming, strict target/options/grid validation,
ABI/math conflicts, Gluon and later-stage rejection, ordinary JIT rejection,
incomplete graph limitations, hashes and actual build identities. Shared tests
exercise versions, graph/content histories, loops, aliasing, math authorization,
unknown operations and custom synthetic-copy policies. Buddy separately tests
that staging origin IDs cannot upgrade incomplete source history.

Integration tests retain exact integer oracles and strict simulator checks.
Legacy scratch fixtures now pass through two processes or receive explicit v1
boundary diagnostics. CPU Python execution checks CAS success/failure and FP32
conversion to/from the supported FP8 format, including exact stored bits and
signed zero. These ordinary CPU stages carry no Contract or BOSCAME attributes.
IME warmup emits a RISC-V ELF object with the expected `vmadot` instruction;
no target execution is claimed.

## Numerical acceptance

All three independent examples rerun `make`, `make check` and full `make simcheck`.
Their original stress inputs, reverse/tail-first launches, readonly inputs,
poisoned persistent SPM, guards and exact access ranges remain enabled. Integer
results are exact; LayerNorm retains its original `atol=rtol=2e-5` rule and reports
zero maximum absolute error for its 197-row run.

ViT reruns the 34 previously established operator checks. GELU and Softmax use
the full stress grids and all three rounds; PV checks 624 programs in each of
three rounds with exact ordered-f32 results and transfer-range audits. QK uses
its complete 2028-program grid. Every case's result and module hash is retained
in the evidence. All 40 launch/adapter checks and the W8A8 chain also run; the
chain retains exact integer results and ordered dequantization, bias and residual
operations. The full simulators retain persistent SPM within their documented
execution/shard scopes.

The six previously unestablished standalone compositions are excluded from the
numerical pass count: the five `linear_*` cases and `patch_embed_196x768`.
They still compile through restricted assembly and objects. This migration does
not establish full-model accuracy, board correctness or performance results.

Example acceptance covers only `matmul_i8_128x1024x128`,
`matmul_i8_196x768x768`, `layernorm_197x768` and `vit-base-patch16-224`.
The retired GEMM, Linalg demo and validations directories are removed. Their
example-specific board wrapper and saved LayerNorm wrapper are retired;
independent backend tests still cover ordered reductions, padded lanes, exact
transfer ranges, immutable inputs, guards and corrupt board readback. The AME
assembly-ordering fixture now lives in the backend test suite.

## Reproduction

Use separate Python environments: a frontend environment with the selected
Triton installation, Torch and NumPy, and a Buddy core environment without
Triton. Select fresh build directories and a single verified LLVM build.
See [the build and installation instructions](TritonContract.md#build-and-install).
For native changes first run `make` in the nested Triton source directory with
its configured `TRITON_BUILD_DIR` and frontend Python.

```sh
ninja -C "$BUDDY_BUILD" check-buddy
ninja -C "$FRONTEND_BUILD" check-triton-shared-lit-tests
TRITON_SHARED_OPT_PATH="$FRONTEND_OPT" \
TRITON_CONTRACT_PYTHON="$FRONTEND_PYTHON" \
  ninja -C "$FRONTEND_BUILD" check-triton-contract-integration

export BUDDY_MLIR_BINARY_DIR="$BUDDY_BUILD/bin"
export LLVM_BINARY_DIR="$LLVM_BUILD/bin"
export LLVM_BIN="$LLVM_BINARY_DIR"
export TRITON_SHARED_OPT_PATH="$FRONTEND_OPT"
export PATH="$BUDDY_MLIR_BINARY_DIR:$LLVM_BINARY_DIR:$PATH"
for example in matmul_i8_196x768x768 matmul_i8_128x1024x128 layernorm_197x768; do
  make -C "examples/FPGA-BOSCAME/$example" PYTHON="$BACKEND_PYTHON" all check simcheck
done

export BUDDY_ROOT="$PWD"
VIT="$FRONTEND_ROOT/examples/BOSCAMEFPGA/vit-base-patch16-224"
"$FRONTEND_PYTHON" "$VIT/tools/build.py" --all --stage asm
"$FRONTEND_PYTHON" "$VIT/tools/simcheck.py" --all
"$FRONTEND_PYTHON" "$VIT/tools/check_launch.py" --all
"$FRONTEND_PYTHON" "$VIT/tools/check_w8a8_chain.py"

PYTHONPATH="$FRONTEND_ROOT/triton/python" "$FRONTEND_PYTHON" -m pytest -s --tb=short \
  "$FRONTEND_ROOT/python/examples/test_swap.py" \
  "$FRONTEND_ROOT/python/examples/smtime/test_smtime.py"
```

`simcheck --all` reports the six unsupported compositions separately; inspect
case discovery and statuses rather than interpreting successful compilation as
numerical acceptance. To reproduce comparison evidence, preserve each baseline
and migrated output directory, hash the listed products without normalization,
and require equality:

```sh
python3 tools/compare-triton-contract-artifacts.py \
  --baseline-examples "$BASELINE_EXAMPLES" \
  --candidate-examples "$MIGRATED_EXAMPLES" \
  --baseline-vit "$BASELINE_VIT_KERNELS" \
  --candidate-vit "$MIGRATED_VIT_KERNELS" \
  --output artifact-comparison.json
```

The example roots contain `<example>/<example>/kernel.*`; the ViT roots contain
`<case>/kernel.*`. The saved JSON lists all comparisons and their digests.
Run both repositories' pre-commit hooks on existing changed files only, then
run `git diff --check` in each repository.
