# Triton AOT Contract handoff

Triton compilation belongs to the independent `triton-riscv` frontend. Buddy
starts with a verified, standard MLIR buffer Contract and owns BOSCAME planning,
plan verification, emission, LLVM lowering and firmware construction.

```text
Triton AST / TTIR
  triton-riscv: TTIR normalization -> bufferization -> Contract annotate/verify
  handoff: kernel.frontend.mlir + contract-manifest.json (+ kernel.ttir)
  Buddy: Contract verify -> plan -> plan verify -> emit -> LLVM / objects
```

The BOSCAME frontend target exports AOT artifacts only. It does not discover a
device or produce an object or a launchable Triton kernel. The ordinary CPU,
RISC-V and IME paths retain their existing defaults; `contract=false` remains
the default for `--triton-to-linalg-experimental`.

## Ownership and compatibility

Buddy owns the [Kernel Contract v1 specification](KernelContract.md). It defines
the neutral ABI, provenance, content versions, scopes, tasks and math
authorization. The boundary is standard MLIR text, with schema compatibility
checked by conformance tests in both compilers.

Buddy's implementation is part of `BuddyKernelContract`, under
`midend/include/Analysis` and `midend/lib/Analysis`. The frontend's strict
producer implementation belongs to its existing `TritonSharedAnalysis` library,
under `include/triton-shared/Analysis` and `lib/Analysis`. Each compiler builds
its own implementation against standard LLVM/MLIR. Neither core build needs
the other checkout, headers, libraries or tools. There is no separate shared
package, source mirror, synchronization script or package manifest.

The strict producer API has no planned-IR option. It checks existing module and
graph versions before normalization and rejects unknown versions. Existing i32
and i64 version attributes remain compatible. Buddy's adapter separately admits
`plan.*` attributes, skips producer normalization and retains logical memory
space zero. Only Buddy interprets `generated_by="planner-staging"`, through a
generic source-policy callback used identically by annotation and verification.
An incomplete source history cannot become complete merely because a generated
copy carries an origin ID.

The handoff contains no `tt`, `tts`, `tptr`, BOSCAME or other nonstandard
operations, target storage decisions, `plan.*` attributes, tensor types or
unrealized casts. A verified incomplete graph keeps its limitation. Verification
does not imply that the current BOSCAME planner supports every operation.

## Build and install

Use one verified LLVM/MLIR build for comparisons across the boundary. Check the
frontend's `triton-hash.txt`, apply its complete patch stack and build its Triton
Python installation using that LLVM. Do not reuse a CMake cache referring to a
missing or different LLVM build. In the frontend's Python environment:

```sh
export FRONTEND_ROOT=/path/to/triton-riscv
export LLVM_SYSPATH=/path/to/llvm-build
export TRITON_PLUGIN_DIRS="$FRONTEND_ROOT"
bash "$FRONTEND_ROOT/scripts/apply_patches.sh" "$FRONTEND_ROOT/triton"
python -m pip install --no-build-isolation -e "$FRONTEND_ROOT/triton"
```

Follow the frontend's build instructions for its Python, CMake, Ninja, compiler
and JSON dependencies. Its upstream patches include strict-properties text
compatibility and explicit symbol accessors for the selected MLIR. The existing
`0004-llvm-symbol-inline-asm-compat.patch` supplies those accessors for `tt.func`.
`triton-shared-opt` and the Python exporter must come from the same installation.
The tool can be selected with `TRITON_SHARED_OPT_PATH`, an installed bundled
tool, or PATH, in that order.
After rebuilding native Triton code, install the rebuilt library as well.
An editable Python installation can retain a copied `_C/libtriton.so`; verify
that its hash matches the selected build before validating AST exports.

Build Buddy normally, using a fresh build directory when first removing the
frontend. Its C API and Python dialect registration no longer register or link
Triton, TritonStructured or TPtr. The Tenstorrent TTIR and TorchInductor examples
are separate integrations and remain in place.

## Python AOT API

```python
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.contract import export_contract
from triton.compiler import ASTSource

artifact = export_contract(
    ASTSource(fn=kernel, signature=signature, constexprs=constexprs),
    target=GPUTarget("boscame", "generic", 1),
    launch_grid=(7, 6, 1),
    entry_name="my_kernel",             # optional SymbolTable rename
    caller_contract="args_aligned64_nonalias",  # explicit caller promise
)
artifact.write("handoff")
```

`source` also accepts a `.ttir` or TTIR `.mlir` file. Both paths run frontend
normalization. Gluon and later-stage IR are rejected. `launch_grid` is always
required and contains exactly three positive int32 extents. Only the exact
target `GPUTarget("boscame", "generic", 1)` is supported.

No alignment, disjointness or math authorization is inferred from the target.
Omitting `caller_contract` makes no new ABI promise. Existing explicit input
facts must agree with supplied grid, ABI and math promises. Renaming updates
symbol references through MLIR's SymbolTable API. Remaining uninlined calls
have an explicit v1 diagnostic.

Optional function-level approximation authorization uses the closed schema:

```python
math_accuracy = {
    "exp": {
        "domain": "all-f32",
        "special_values": "ieee",
        "max_abs_error": 1e-6,
        "max_rel_error": 1e-5,
    }
}
```

Only `erf` and `exp` are defined. Both error budgets must be finite and
nonnegative; authorization does not itself select an approximation. BOSCAME
recipes, coefficients, evaluation order and measured error evidence belong to
Buddy. The existing GELU/Softmax budgets and special-value behavior are unchanged.

`options` accepts boolean `debug`, `enable_make_gather_scatter`,
`enable_collapse_shape`, `enable_fp_fusion` and `allow_fp_reassoc`. Fusion and
reassociation must remain false. Unknown options are rejected. Ordinary AST
frontend defaults are explicit and CPU/IME environment configuration is ignored.
`triton.compile(..., target=...)` gives an AOT-only diagnostic for this target;
use `export_contract()` instead.

`ContractArtifact` contains `ttir`, `contract` and `manifest`. The manifest
records input/output hashes, actual entry ABI, grid, explicit/effective promises,
graph limitations, effective options, Contract schema version, frontend
source/build identity, Triton source and patch identity, loaded Python/compiler
binary hashes and LLVM/MLIR build hashes. It contains no developer absolute
paths. Every export revalidates serialized Contract IR in a fresh parser; there
is no persistent AOT cache or reuse of ordinary object caches.

## Command and example migration

The original frontend pass is now provided by external `triton-shared-opt`:

```sh
triton-shared-opt kernel.ttir \
  --triton-to-linalg-experimental='contract=true caller-contract=args_aligned64_nonalias launch-grid=my_kernel:7,6,1' \
  --kernel-contract-verify -o kernel.frontend.mlir
buddy-opt kernel.frontend.mlir --kernel-contract-verify \
  --bosc-ame-plan --bosc-ame-plan-verify -o kernel.planned.mlir
buddy-opt kernel.planned.mlir --bosc-ame-emit -o kernel.boscame.mlir
buddy-opt kernel.boscame.mlir --lower-bosc-ame -o kernel.llvm.mlir
buddy-translate kernel.llvm.mlir --buddy-to-llvmir -o kernel.ll
```

The pass CLI accepts normalized TTIR as before. The public AOT API additionally
normalizes AST/file input and produces the manifest. A TTIR-only command is:

```sh
python -m triton.backends.triton_shared.contract kernel.ttir \
  --launch-grid=7,6,1 --entry-name=my_kernel \
  --caller-contract=args_aligned64_nonalias --output handoff
```

Buddy no longer provides `--triton-to-linalg-experimental` or the retired
`--triton-to-boscame` scratch ABI. All Triton Python kernels and exporters,
including the complete ViT model,
live in `triton-riscv/examples/BOSCAMEFPGA`. Buddy retains three backend-only
examples with checked-in `kernel.frontend.mlir` inputs and starts at Contract
verification. Their ABI generators and simulators need no Triton package.
Use explicit tool variables, installed tools or PATH; no driver chooses a build
directory by sorting old build names. For the independent examples:

```sh
export BUDDY_MLIR_BINARY_DIR=/path/to/buddy-build/bin
export LLVM_BIN=/path/to/llvm-build/bin
make -C examples/FPGA-BOSCAME/matmul_i8_196x768x768 \
  PYTHON=/path/to/backend-python all check simcheck
# Optionally replace the checked-in input:
make -C examples/FPGA-BOSCAME/matmul_i8_196x768x768 \
  FRONTEND_IR=/path/to/handoff/kernel.frontend.mlir all check simcheck
```

In the frontend repository, the handoff identity invalidates frontend products
when the actual frontend, its Contract implementation or Python binary changes.
The compiled source identity includes the local Contract header and source.
Manifests accompany the Contract, and
board/image reports bind the frontend and handoff hashes to backend artifacts.
Existing descriptor/grid/alignment/nonalias checks, simulator guards and assembly
allowlists remain the acceptance criteria.

## Test boundaries

Contract conformance tests are part of the normal suites, under
`tests/KernelContract` in Buddy and `test/Contract` in the frontend:

```sh
ninja -C /path/to/frontend-build check-triton-shared-lit-tests
ninja -C /path/to/buddy-build check-buddy
```

Frontend lit discovers `.py` semantic tests and requires its configured Python
with NumPy. Pure tensor normalization/provenance tests belong there. Standard
buffer Contract, planner/emitter and simulator tests remain in Buddy's core
suite, which uses a Python environment without Triton. The cross-repository
suite lives in the frontend repository and is explicitly separate. Configure
that build with `BUDDY_ROOT` and `BUDDY_MLIR_BINARY_DIR` to enable it:

```sh
TRITON_SHARED_OPT_PATH=/path/to/triton-shared-opt \
TRITON_CONTRACT_PYTHON=/path/to/frontend-python \
  ninja -C /path/to/frontend-build check-triton-contract-integration
```

Its `contract-pipeline-opt` is a test-only process bridge, never a Buddy compiler
pass or linked frontend dependency. See the paired revision/source identities,
artifact comparisons and acceptance evidence in
[TritonContractValidation.md](TritonContractValidation.md). The subsequent
implementation layout and current test counts are recorded in
[KernelContractValidation.md](KernelContractValidation.md).
