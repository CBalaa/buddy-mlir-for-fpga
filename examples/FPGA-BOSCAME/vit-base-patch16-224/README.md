# ViT-Base/Patch16-224 FPGA example

This directory contains an FPGA adaptation of
`google/vit-base-patch16-224`.  The model is fixed to batch 1, FP32 input
`[1,3,224,224]`, 196 patches plus the CLS token, and the official 12-layer
encoder.  See [`model/README.md`](model/README.md) for the graph contract and
reproducible commands.

The source of truth for operations is [`triton/kernels.py`](triton/kernels.py)
and the fixed shape inventory in [`triton/cases.py`](triton/cases.py).  Each
case is exported through Triton's CPU backend, lowered from TTIR to shared
linalg IR, then lowered by Buddy.  `tools/import_model.py` captures the real
HuggingFace graph; `tools/replace_triton.py` records and applies shape-checked
external calls; `tools/lower_model.py` emits the final MLIR hand-off.

The existing Buddy Python extension in `build-python/python_packages` was
compiled for Python 3.11. Use the matching `trex` environment for graph
capture and replacement, and `boscame` (Python 3.12) for Triton export and NR
kernel builds. Run commands from the repository root so this example's
`triton/` source directory does not shadow the installed Triton package.

```bash
source /home/chh/miniconda3/etc/profile.d/conda.sh
conda activate boscame
unset PYTHONPATH
source examples/FPGA-BOSCAME/common/triton/triton-env.sh
export PYTHONSAFEPATH=1
python3 -B examples/FPGA-BOSCAME/vit-base-patch16-224/tools/fetch_assets.py \
  --output examples/FPGA-BOSCAME/vit-base-patch16-224/assets
python3 -B examples/FPGA-BOSCAME/vit-base-patch16-224/triton/build.py \
  --output examples/FPGA-BOSCAME/vit-base-patch16-224/build/kernels-host --host --suite
conda activate trex
export PYTHONPATH="$PWD/build-python/python_packages${PYTHONPATH:+:$PYTHONPATH}"
python3 -B examples/FPGA-BOSCAME/vit-base-patch16-224/tools/import_model.py \
  --assets examples/FPGA-BOSCAME/vit-base-patch16-224/assets \
  --checkpoint examples/FPGA-BOSCAME/vit-base-patch16-224/assets/model.safetensors \
  --output examples/FPGA-BOSCAME/vit-base-patch16-224/build/import --save-mlir
```

The remaining graph and image steps are in [`model/README.md`](model/README.md).

The `--host` build is a numerical host check. To build the standalone NR
operator suite, reactivate `boscame`, run `triton/build.py` with `--suite`
and without `--host`, and review its `suite-all/manifest.json` before using
the shared FPGA runner. See the model README for the full graph and image
commands.

The graph stages (`replace`, `lower`, `host-lib`, and `nr-archive`) produce
the MLIR, host bridge, and kernel archive used by the model-image link. A
replacement report or successful link alone does not establish numerical
agreement with the original model.

For a W8A8 graph check, run the host bridge checker after building the
replacement. It reconstructs INT8 weights from the report's named source
tensors, allocates activation and accumulator workspaces from `input_layout`,
verifies the mixed-dtype `@vit` ABI, and reports maximum/mean error and top-1:

The default W8A8 acceptance band is `atol=0.5, rtol=0.1`; the report always
keeps the raw maximum and mean absolute errors, and stricter values can be
passed with `--atol` and `--rtol`.

```bash
PYTHONPATH="$PWD/build-python/python_packages" \
  /home/chh/miniconda3/envs/trex/bin/python -B \
  examples/FPGA-BOSCAME/vit-base-patch16-224/tools/check_w8a8_graph_host.py \
  --replacement examples/FPGA-BOSCAME/vit-base-patch16-224/build/replacement-w8a8-final2 \
  --host-library examples/FPGA-BOSCAME/vit-base-patch16-224/build/host-bridge-w8a8-current/libvit_triton_host.so
```

The immutable model revision is `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`.
The expected safetensors SHA256 is
`1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d`.
The image preprocessor uses resize/crop to 224 and mean/std `(0.5,0.5,0.5)`.
Board input is already normalized `pixel_values`; preprocessing is kept outside
the captured graph so the model ABI remains a fixed tensor contract.
