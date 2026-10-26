# BOSCAME example ownership and relocation verification

This record describes the example relocation before removal of the shared
Contract package. Its source identities and test counts remain frozen; see
[KernelContractValidation.md](KernelContractValidation.md) for the subsequent
implementation layout and current verification.

All Triton Python sources and frontend integration tests belong to
`triton-riscv`. Its `examples/BOSCAMEFPGA` directory contains the complete
`layernorm_197x768`, `matmul_i8_128x1024x128`, `matmul_i8_196x768x768` and
`vit-base-patch16-224` examples. The complete ViT directory, including local
ignored assets and generated artifacts, was moved without discarding files.
All 43 kernel source files retain their original SHA256 values.

Buddy retains the three standalone backend examples. Each consumes a checked-in
`kernel.frontend.mlir`, or a replacement selected with `FRONTEND_IR`. Their
Python helpers use fixed Contract ABI metadata to generate the same adapters
and launcher configuration, without importing Triton. The entry symbols,
descriptor/grid/alignment/nonalias checks, launchers, linker scripts, simulator
oracles and numerical tolerances are preserved. Frontend export targets and
their Python dependencies have been removed from these makefiles.

The frontend copies preserve complete image workflows. Export-only goals need
no Buddy checkout. Backend stages use explicitly selected tools; runtime,
assembly filters and simulators use `BUDDY_ROOT`. No path depends on the
frontend being nested inside Buddy. The complete ViT graph tooling remains
with the model, using installed Buddy Python packages and explicit backend
tool/LLVM paths.

TTIR integration fixtures, their two-process bridge and ViT model/oracle tests
are under `triton-riscv/test/Integration/BOSCAMEFPGA`. This optional suite is
excluded from the frontend core suite. Ordinary CPU execution tests also live
there because the existing CPU pipeline uses Buddy's generic lowering tools.
Buddy's simulator and standard Contract/backend tests remain in Buddy.

## Verification

The [machine-readable evidence](validation/boscame-example-relocation/results.json)
records source identities, artifact comparisons and validation counts. The
earlier [migration evidence](TritonContractValidation.md) remains a historical
record of its original paired source state.

- All three Buddy examples build from Contract to bin under Python without
  Triton, pass `make check` and pass full `make simcheck`.
- All three frontend examples export without Buddy and retain working complete
  `make all check` workflows with explicit external Buddy configuration.
- The three exported Contracts match the previous exports byte for byte.
  Checked-in input copies only remove one trailing empty line, as required by
  the repository's end-of-file hook.
- All 18 recorded standalone backend products and all 240 ViT products match
  the previous products byte for byte. Together with the three exports this
  preserves the original 261 artifact comparisons. All three final bin files
  also match the original images byte for byte.
- All 40 relocated ViT cases compile through restricted assembly and objects,
  and all 40 launcher ABI checks pass. The relocated LayerNorm, quantization,
  bias-add and W8A8 chain checks execute emitted IR with their existing rules.
  The prior complete GELU/Softmax/QK/PV stress evidence remains tied to
  identical emitted modules; those lengthy runs are not counted as reruns here.

| Suite | Discovered | Passed | Expected failure | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| Buddy core | 551 | 549 | 0 | 2 |
| Buddy examples | 241 | 239 | 0 | 2 |
| Frontend core | 335 | 324 | 9 | 2 |
| Optional frontend integration | 19 | 19 | 0 | 0 |
| Shared Contract, each repository | 11 | 11 | 0 | 0 |

The four ViT tests moved from Buddy core to the optional suite. The frontend
launch-grid test moved to frontend core, while CPU execution stays optional.
Test discovery counts therefore account for every relocated test.

## Reproduction

From the frontend checkout, select its installed Python backend and tool:

```sh
export TRITON_SHARED_OPT_PATH=/path/to/triton-shared-opt
make -C examples/BOSCAMEFPGA/matmul_i8_196x768x768 ttir PYTHON="$FRONTEND_PYTHON"
```

From Buddy, using ordinary Python with NumPy:

```sh
export BUDDY_MLIR_BINARY_DIR="$BUDDY_BUILD/bin"
export LLVM_BIN="$LLVM_BUILD/bin"
for name in layernorm_197x768 matmul_i8_128x1024x128 matmul_i8_196x768x768; do
  make -C "examples/FPGA-BOSCAME/$name" PYTHON="$BACKEND_PYTHON" all check simcheck
done
```

Add `FRONTEND_IR=/path/to/export/kernel.frontend.mlir` to use a fresh handoff.
The handoff must match the example's fixed ABI and dimensions.

Configure optional frontend tests with `-DBUDDY_ROOT=...` and
`-DBUDDY_MLIR_BINARY_DIR=...`, then run:

```sh
ninja -C "$BUDDY_BUILD" check-buddy
ninja -C "$FRONTEND_BUILD" check-triton-shared-lit-tests
TRITON_SHARED_OPT_PATH="$FRONTEND_OPT" TRITON_CONTRACT_PYTHON="$FRONTEND_PYTHON" \
  ninja -C "$FRONTEND_BUILD" check-triton-contract-integration
```

For relocated ViT workflows, set `BUDDY_ROOT` and put the selected Buddy/LLVM
binaries on PATH, then use `examples/BOSCAMEFPGA/vit-base-patch16-224` in the
frontend checkout. See its README for compilation and numerical commands.
