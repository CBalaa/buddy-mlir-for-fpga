# Kernel Contract implementation layout validation

Both repositories' top-level `shared` directories are removed. Contract v1
remains the same standard MLIR boundary; [KernelContract.md](KernelContract.md)
is its normative specification. This record follows the original
[frontend split](TritonContractValidation.md) and
[example relocation](BoscameExampleRelocation.md), whose saved source identities
and results remain historical records.
The final section records the subsequent upstream branch and patch audit.

## Implementation ownership

The former package's annotation, verification and graph reconstruction code is
required by both compilers. Buddy builds that code from
`midend/lib/Analysis/KernelContractCore.cpp`, with its internal declarations in
`midend/include/Analysis/detail/KernelContractCore.h`. The existing
`KernelContract.cpp` owns the public passes and planner staging policy;
`MathAccuracy.h` contains the helpers used by the BOSCAME backend. They are part
of `BuddyKernelContract` and link only standard LLVM/MLIR libraries.

The frontend builds its own strict implementation from
`lib/Analysis/KernelContract.cpp` and
`include/triton-shared/Analysis/KernelContract.h`, as part of
`TritonSharedAnalysis`. It exposes no planned-IR option or consumer source-policy
callback. Those unused producer branches and the unused frontend math helper
are removed. The frontend still annotates and verifies every export, rejects
target decisions and unknown versions, and preserves incomplete graphs.

The separate package CMake target, standalone driver, revision file, package
manifest and synchronization script are retired. Both regenerated Ninja
dependency graphs contain no old package includes, source paths or libraries.
Frontend build identities now record the schema version and actual local
source hashes, including its Contract header and implementation. AOT manifests
retain input/output, ABI/grid, promises, tool, Triton and LLVM/MLIR identities.

## Verification

Both compilers were rebuilt against the same verified LLVM/MLIR build. Buddy's
C API and Python registrations were rebuilt and tested under Python without
Triton. The frontend was rebuilt with its existing pinned, patched Triton.
Neither LLVM nor nested Triton sources were changed by this layout adjustment.

| Suite | Discovered | Passed | Expected failure | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| Buddy core | 562 | 560 | 0 | 2 |
| Buddy examples | 241 | 239 | 0 | 2 |
| Frontend core | 345 | 334 | 9 | 2 |
| Optional frontend integration | 19 | 19 | 0 | 0 |

Buddy's 11 former package tests now run under `tests/KernelContract`. Ten
strict producer conformance tests now run under the frontend's `test/Contract`.
The synthetic transfer callback test runs only in Buddy, which owns the
consumer policy. No separate package suite is required.

All three standalone examples export without Buddy. Buddy consumes those fresh
Contracts and passes `make all check simcheck` for LayerNorm and both integer
Matmuls under Python without Triton. All 40 ViT cases compile again through
restricted assembly and objects.

The [machine-readable evidence](validation/kernel-contract-layout/results.json)
compares 261 products to the saved example-relocation baseline: three standalone
exports, 18 standalone backend products and 240 ViT products. Every comparison
is identical without normalization, as are all three final bin images. Integer
and floating-point results, evaluation order, math authorization and tolerances
are unchanged. The lengthy ViT numerical stress runs retain their earlier
evidence against identical emitted modules; they are not counted as reruns.

## Reproduction

Follow [TritonContract.md](TritonContract.md) for tool configuration and the
ordinary build, test and example commands. Contract tests are included in
`check-buddy` and `check-triton-shared-lit-tests`; the optional two-repository
suite remains `check-triton-contract-integration`.

For separate standalone export output, use the frontend driver's existing
`--export-contract --output <build-root>` arguments. It writes
`<build-root>/<example>/kernel.frontend.mlir`; pass that file to Buddy with
`FRONTEND_IR`. For ViT use `tools/build.py --all --stage asm --output <kernels>`.
Keep the baseline directories before rebuilding and compare the listed files
as raw bytes. Local run logs and artifacts for this adjustment are under
`build-contract-migration/shared-removal`; they are not build dependencies.

## Upstream branch and patch audit

An independent live query of
[`RuyiAI-Stack/triton-riscv` main](https://github.com/RuyiAI-Stack/triton-riscv/tree/main)
confirmed `a305fb6ddec2401183cf7d1af878a5d1b6ee3553`. At the time of the audit,
the local branch was `main`, tracked `origin/main` and had that same HEAD. Its
uncommitted changes were the frontend migration. The saved migration baseline
records a clean outer triton-riscv checkout at this revision; the nested Triton
already carried project patches. These are separate Git repositories.

The added `0007-llvm-symbol-traits.patch` and its applied `SymbolName,
SymbolVisibility` line are removed. Upstream main's existing `0004` already
supplies the `tt.func` name and visibility accessors, and
`FunctionOpInterface` retains the symbol interface without the added traits.
Starting from clean nested Triton
`58895270e6230491acc15fe7ba6d4c849e838db1`, applying only the six unchanged
patches tracked by upstream main reproduces all 23 modified source files byte
for byte. Applying the patch script again changes no source files. LLVM source
remains untouched.

Without `0007`, the Triton native library and frontend tool rebuild successfully.
The installed Python native library was synchronized and its hash verified
against the rebuilt library. The frontend core suite discovers 345 tests:
334 pass, 9 are expected failures and 2 are unsupported. All 19 optional
integration tests pass. CPU swap execution and RISC-V IME object generation both
pass with a new cache and explicit Triton Python package path.

All three standalone examples export again without Buddy and build through bin
with Buddy. All 40 ViT cases compile through restricted assembly and objects.
The same 261 artifact comparisons and three final image comparisons remain
byte-identical to the layout verification above. Existing numerical stress
evidence remains attached to identical emitted modules. The removed patch is
therefore unnecessary for this pinned Triton revision and LLVM/MLIR build.

[Patch-audit evidence](validation/triton-patch-audit/results.json) records the
live remote identity, original clean baseline, six-patch replay, rebuilt native
identity, test counts and raw artifact comparisons. Local logs are under
`build-contract-migration/symbol-traits-audit`.
