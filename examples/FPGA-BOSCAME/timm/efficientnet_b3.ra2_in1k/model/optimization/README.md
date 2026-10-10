# AME / dequant optimization checks (EfficientNet)

Same **evidence pattern** as `qwen3-0.6b/model/optimization/`: independent C oracles,
A/B linked images, strict UART archivers. Vision models use **smaller** dequant and
int8 linear shapes (classifier `1×1536→1000`, 1×1 conv matmuls) instead of
151936-wide LM heads.

## Dequantization A/B

After Triton builds baseline and RVV-optimized `dequantize_*` cases for EfficientNet
shapes, link them with the Qwen-style `build_dequant_ab.py` flow (copy that script
here when running a board campaign). Shapes come from `model/build/.../triton-build/`,
not from LM vocabulary width.

Example shape family (replace paths after a local build):

```bash
MODEL=examples/FPGA-BOSCAME/timm/efficientnet_b3.ra2_in1k/model
python3 "$MODEL/optimization/build_dequant_ab.py" \
  --baseline "$MODEL/build/ame-v05/baseline/dequantize_1x1000" \
  --optimized "$MODEL/build/ame-v05/rvv/dequantize_1x1000"
```

Board archive: `archive_dequant_ab.py` → `model/validation/board/ame-v05/...`

## AME int8 matmul A/B

Use `build_ame_ab.py` on EfficientNet `matmul_i8` cases (1×1 conv and classifier
tiles). The int64 oracle and fence timing semantics match Qwen; only case names
and matrix sizes differ.

## Model variant relink

Qwen's `build_model_variant.py` swaps kernels in a **1-layer LLM** stage with
tokenizer and KV segments. EfficientNet has **no tokenizer/KV**; when testing an
optimized kernel on a partial graph, relink with `kernel_build_overlay.py` +
`archive_kernels.py` and `build_model_image.py` (FP32) or the W8A8 segment +
`build_nr_w8a8_image.py` path once the full graph is quantised.

Do not treat a single-kernel PASS as full-model acceptance.
