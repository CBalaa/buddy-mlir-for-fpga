#!/usr/bin/env python3
# ===- quant_model_reference.py - W8A8 vision reference --------*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
"""W8A8 host reference scaffolding for EfficientNet-B3 (vision).

The Qwen tree implements a full prefill/decode graph walk with RoPE, attention,
and KV. Vision deployment quantizes only selected ``linear`` / 1×1 conv matmuls;
everything else stays FP32 until the W8A8 rewrite covers the full graph.

This tool documents the CLI surface and implements a **classifier head**
``linear_w8a8`` check using ``quant_reference.py`` and checkpoint weights.
Full-graph W8A8 forward will reuse the same quant contract once
``rewrite_linear_to_w8a8`` is enabled end-to-end in ``triton_call_replace.py``.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_reference import error_metrics, linear_w8a8, quantize_rows  # noqa: E402


def load_classifier_weight(checkpoint: Path):
    from safetensors.torch import load_file

    state = load_file(str(checkpoint / "model.safetensors"))
    # timm EfficientNet-B3 head: classifier.weight [1000, 1536], bias [1000]
    weight = state["classifier.weight"].float().numpy()
    bias = state["classifier.bias"].float().numpy()
    return weight, bias


def check_classifier_linear(checkpoint: Path, activation_rows: int, output: Path):
    weight, bias = load_classifier_weight(checkpoint)
    rng = np.random.default_rng(0)
    activation = rng.standard_normal((activation_rows, weight.shape[1]), dtype=np.float32)
    wq, wscale = quantize_rows(weight)
    quantized_out = linear_w8a8(activation, wq, wscale)
    fp32_out = activation @ weight.T + bias[None, :]
    # Compare only the linear matmul slice (bias stays FP32 in W8A8 path).
    fp32_linear = activation @ weight.T
    metrics = error_metrics(quantized_out, fp32_linear)
    report = {
        "model": "timm/efficientnet_b3.ra2_in1k",
        "check": "classifier linear_w8a8 vs FP32 matmul",
        "activation_shape": list(activation.shape),
        "weight_shape": list(weight.shape),
        "metrics_vs_fp32_linear": metrics,
        "quantization": "per-output-channel int8 weights; per-row activation int8 (quant_reference)",
        "note": "Full-graph W8A8 forward is not implemented here; use host_reference.py for FP32.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="directory containing model.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-rows", type=int, default=4,
                        help="synthetic batch rows for classifier linear check")
    # LLM-only flags retained for path parity; vision does not use them.
    parser.add_argument("--prefill-len", type=int, default=None,
                        help="N/A for vision (ignored)")
    parser.add_argument("--prompt-ids", default=None, help="N/A for vision (ignored)")
    parser.add_argument("--decode-steps", type=int, default=None, help="N/A for vision (ignored)")
    args = parser.parse_args()
    check_classifier_linear(args.checkpoint, args.activation_rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
