# ViT-Base/Patch16-224 on BOSCAME FPGA

This example adapts [`google/vit-base-patch16-224`](https://hf-mirror.com/google/vit-base-patch16-224) to the same pipeline used by `qwen3-0.6b`:

1. Triton kernels are exported for fixed ViT shapes and lowered through
   `triton-riscv` and Buddy to an NR FPGA image.
2. `../tools/import_model.py` captures the real HuggingFace
   `ViTForImageClassification` graph with Buddy's `DynamoCompiler`.
3. `../tools/replace_triton.py` matches graph nodes by shape and dataflow and
   replaces supported nodes with the corresponding Triton external call.
4. `../tools/lower_model.py` lowers the replaced graph to MLIR/LLVM. Host checks
   can run before making a board image.

The deployment is intentionally fixed to one image and one batch:

| tensor | shape |
| --- | --- |
| `pixel_values` | `[1, 3, 224, 224]` FP32 |
| patch tokens | `[1, 196, 768]` FP32 |
| encoder sequence | `[1, 197, 768]` FP32 |
| attention heads | `12`, head dimension `64` |
| classifier logits | `[1, 1000]` FP32 |

Each encoder block is pre-layernorm: Q/K/V projections, QK dot product (with the explicit `1/sqrt(64)` scale),
softmax, PV dot product, output projection and residual add, followed by the
3072-wide GELU MLP and a second residual add. LayerNorm includes the mean and
variance terms; it is not the Qwen RMSNorm kernel. Patch projection is a
16x16 stride-16 convolution expressed as a Triton kernel, so the graph does
not depend on a linalg-only convolution implementation.

## Build

Run from the repository root. The checked-in Buddy Python extension is built
for Python 3.11, so graph capture and the host graph check use the matching
`trex` environment. Triton export and NR compilation use `boscame`
(Python 3.12). The two kernel builds use separate output directories.

```bash
VIT="$PWD/examples/FPGA-BOSCAME/vit-base-patch16-224"
source /home/chh/miniconda3/etc/profile.d/conda.sh
conda activate boscame
unset PYTHONPATH
source examples/FPGA-BOSCAME/common/triton/triton-env.sh
export PYTHONSAFEPATH=1
python3 -B "$VIT/tools/fetch_assets.py" --output "$VIT/assets"
python3 -B "$VIT/triton/build.py" --output "$VIT/build/kernels-host" --host --suite
python3 -B "$VIT/triton/build.py" --output "$VIT/build/kernels-nr" --suite
```

The host suite checks all 20 fixed-shape cases independently. The NR build
produces a board operator suite at `build/kernels-nr/suite-all/suite.bin`.
Capture and replace the official model graph with Buddy:

```bash
conda activate trex
export PYTHONPATH="$PWD/build-python/python_packages${PYTHONPATH:+:$PYTHONPATH}"
python3 -B "$VIT/tools/import_model.py" --assets "$VIT/assets" \
  --checkpoint "$VIT/assets/model.safetensors" --output "$VIT/build/import" --save-mlir
python3 -B "$VIT/tools/replace_triton.py" --import-dir "$VIT/build/import" \
  --triton-build "$VIT/build/kernels-host" --output "$VIT/build/replacement" \
  --require-complete
TRITON_BUILD="$VIT/build/kernels-host" \
ADAPTERS="$VIT/build/replacement/vit_triton_adapters.c" \
OUT="$VIT/build/host-bridge" bash "$VIT/tools/build_host_bridge.sh"
python3 -B "$VIT/tools/check_graph_host.py" --replacement "$VIT/build/replacement" \
  --import-dir "$VIT/build/import" \
  --host-library "$VIT/build/host-bridge/libvit_triton_host.so" \
  --output "$VIT/build/host-graph-check.json"
```

`host-graph-check.json` compares all 1000 logits with the same seeded input in
PyTorch, verifies the 200 parameter tensors and checks that the subgraph has
201 input descriptors. Only after it reports `PASS`, lower the main graph and
subgraph and build the NR image:

```bash
python3 -B "$VIT/tools/lower_model.py" --replacement "$VIT/build/replacement" \
  --output "$VIT/build/lowered" --buddy-opt "$PWD/build-python/bin/buddy-opt" \
  --buddy-translate "$PWD/build-python/bin/buddy-translate"
conda activate boscame
unset PYTHONPATH
RISCV_CC="$PWD/llvm/build-2d26/bin/clang --target=riscv64-unknown-elf" \
LLVM_AR="$PWD/llvm/build-2d26/bin/llvm-ar" \
TRITON_BUILD="$VIT/build/kernels-nr" \
ADAPTERS="$VIT/build/replacement/vit_triton_adapters.c" \
OUT="$VIT/build/nr-archive" bash "$VIT/tools/build_nr_archive.sh"
python3 -B "$VIT/tools/build_model_image.py" --repo-root "$PWD" \
  --params "$VIT/build/import/params.json" \
  --checkpoint "$VIT/assets/model.safetensors" \
  --graph-ir "$VIT/build/lowered/forward.ll" \
  --main-graph-ir "$VIT/build/lowered/main.ll" \
  --subgraph-mlir "$VIT/build/replacement/subgraph0.triton.mlir" \
  --archive "$VIT/build/nr-archive/libvit_triton_nr.a" \
  --adapters "$VIT/build/replacement/vit_triton_adapters.c" \
  --input-seed 0 --expected-top1 767 --output "$VIT/build/image"
```

`triton-call-replacement.json` records every replaced case and any uncovered
nodes. `model-image.json` reports a static link and ELF audit; `PASS_STATIC`
does not indicate that the FPGA has executed the image. The DDR load plan
holds the small boot image at `0x80000000` and the 346 MB FP32 weight segment
at `0xb8000000`.

When a board is available, use the shared loader and inspect both the
`[vit] top1 PASS` line and the NR completion marker in its UART log:

```bash
"$VIT/tools/run_model.sh" "$VIT/build/image" --fpga=5 --capture-seconds=1800
```

## Operator progress trace

For a board run that stops before the classifier result, build a separate
diagnostic image.  The graph remains the same; only the generated C wrappers
around its 245 Triton calls print UART markers.  This follows the Qwen3
progress-probe approach without adding a completion fence to each call.

Activate `trex` and set `PYTHONPATH` as in the graph-capture commands above,
then regenerate the wrappers from the same captured graph:

```bash
python3 -B "$VIT/tools/replace_triton.py" --import-dir "$VIT/build/import" \
  --triton-build "$VIT/build/kernels-host" --output "$VIT/build/replacement-trace" \
  --require-complete
```

Activate `boscame` and unset `PYTHONPATH` as in the NR build commands above.
Reuse the already lowered graph and kernel archive to link the diagnostic
image:

```bash
python3 -B "$VIT/tools/build_model_image.py" --repo-root "$PWD" \
  --params "$VIT/build/import/params.json" \
  --checkpoint "$VIT/assets/model.safetensors" \
  --graph-ir "$VIT/build/lowered/forward.ll" \
  --main-graph-ir "$VIT/build/lowered/main.ll" \
  --subgraph-mlir "$VIT/build/replacement-trace/subgraph0.triton.mlir" \
  --archive "$VIT/build/nr-archive/libvit_triton_nr.a" \
  --adapters "$VIT/build/replacement-trace/vit_triton_adapters.c" \
  --input-seed 0 --expected-top1 767 --trace-ops \
  --hang-diagnostics \
  --output "$VIT/build/image-trace"
```

After the UVHS daemon is healthy and a board is available, run the trace
image with `tools/run_model.sh`:

```bash
"$VIT/tools/run_model.sh" "$VIT/build/image-trace" --fpga=4 \
  --capture-seconds=1800 --startup-timeout=900
python3 -B "$VIT/tools/summarize_op_trace.py" \
  --uart-log examples/FPGA-BOSCAME/build/fpga-runs/RUN_ID/uart.raw.log \
  --replacement-report "$VIT/build/replacement-trace/triton-call-replacement.json"
```

Replace `RUN_ID` with the run directory printed by the loader.  Each UART
line has a stable graph symbol, Triton case, and a `BEGIN` or `END` marker.
A final `BEGIN` without `END` shows the wrapper entered but its return
marker was not observed.  It does not by itself prove that the kernel hung,
especially when UVHS or UART stops externally.  `END` records adapter
return; it does not synchronize asynchronous AME work.  If there is no
operator `BEGIN`, execution stopped between `[vit] forward start` and the
first Triton wrapper.  Normal images do not print these markers unless
`--trace-ops` is set at image build time.

With `--hang-diagnostics`, the same image also enables the shared NH hang-watch
mailbox.  NH periodically reports the last adapter ordinal and up to three
memref descriptors, while `[vit] graph return` confirms that the complete graph
returned.  These observations help separate a long-running kernel from a UART
or UVHS interruption; stale mailbox samples are not proof of the RA program
counter or of a cache/AME root cause.

`build.py` uses the same `kernel.linalg.mlir -> buddy-opt -> LLVM` stages as
Qwen. No handwritten linalg operator implementation is needed for this model.
