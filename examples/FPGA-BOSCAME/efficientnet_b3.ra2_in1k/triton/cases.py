# ===- cases.py - EfficientNet Triton specialisation inventory --*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Read-only inventory. Imports no Triton package so `export.py --list` works
# before the compiler environment exists. Full-model shapes live in the
# committed ``../model-cases/`` tree (static closed set). Extra roots can
# still be appended via EFFNET_CASE_ROOTS, the same idea as Qwen's
# QWEN_CASE_ROOTS.
#
# ===----------------------------------------------------------------------===//
"""Static specializations of EfficientNet-B3 Triton kernels."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
MODEL_CASES = PARENT / "model-cases"
EXTRA_CASE_ROOTS = tuple(
    Path(entry) for entry in os.environ.get("EFFNET_CASE_ROOTS", "").split(os.pathsep)
    if entry
)

SMOKE = (
    "silu_1x256", "sigmoid_1x32", "add_1x256", "mul_1x32",
    "gap_9x9x32", "se_scale_9x9x32", "linear_1x16x32",
    "conv2d_stem_tiny", "conv2d_1x1_9x9_32x16",
    "dwconv2d_9x9x32_k3", "dwconv2d_10x10x16_k5",
    "matmul_3x19x70", "quantize_1x16", "dequantize_1x16",
)


def case_search_roots():
    roots = [PARENT]
    if MODEL_CASES.is_dir():
        roots.append(MODEL_CASES)
    roots.extend(EXTRA_CASE_ROOTS)
    return tuple(roots)


def case_directory(name):
    for root in case_search_roots():
        candidate = root / name
        if (candidate / "metadata.json").is_file():
            return candidate
    raise ValueError("unknown case: " + name)


def power_of_two(value):
    return 1 << (value - 1).bit_length()


def cdiv(value, divisor):
    return (value + divisor - 1) // divisor


def product(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def describe(name):
    directory = case_directory(name)
    metadata = json.loads((directory / "metadata.json").read_text())
    kind = metadata["kind"]
    constants = {}
    grid = (1, 1, 1)
    arguments = []
    kernel = kind
    kernel_module = "kernels"

    def arg(label, rank, dtype="f32"):
        arguments.append({"name": label, "rank": rank, "dtype": dtype})

    if kind in ("matmul_i8", "matmul_f32"):
        m, n, k = metadata["shape"]
        integer = kind == "matmul_i8"
        kernel = "matmul"
        arg("A", 2, "i8" if integer else "f32")
        arg("B", 2, "i8" if integer else "f32")
        arg("C", 2, "i32" if integer else "f32")
        bm = power_of_two(min(m, 16))
        bn = int(os.environ.get("EFFNET_TRITON_AME_N",
                                os.environ.get("QWEN_TRITON_AME_N", "16")))
        bk = min(64, power_of_two(max(k, 1)))
        constants = dict(M=m, N=n, K=k, BM=bm, BN=bn, BK=bk, INTEGER=integer)
        grid = (cdiv(m, bm), cdiv(n, bn), 1)
    elif kind in ("per_token_quantization", "dequantization"):
        shape = metadata["shape"]
        if kind == "per_token_quantization":
            rows, width = shape
            kernel = "quantize"
            arg("X", 2)
            arg("Q", 2, "i8")
            arg("Scale", 1)
            # Must cover the full row: BLOCK >= WIDTH (masked). Capping at 1024
            # silently drops the tail (e.g. quantize_1x1536 → BLOCK=1024 FAIL).
            block = power_of_two(max(width, 1))
            constants = dict(WIDTH=width, BLOCK=block)
            grid = (rows, 1, 1)
        else:
            rows, cols = shape
            dequant_mode = os.environ.get(
                "EFFNET_TRITON_DEQUANT",
                os.environ.get("QWEN_TRITON_DEQUANT", "baseline"))
            if dequant_mode == "rvv":
                kernel = "dequantize_rows"
                kernel_module = "kernels_dequant"
                arg("X", 2, "i32")
                arg("Row", 1)
                arg("Column", 1)
                arg("Out", 2)
                block = power_of_two(min(max(cols, 1), 128))
                constants = dict(ROWS=rows, COLS=cols, BLOCK=block)
                grid = (cdiv(cols, block), rows, 1)
            else:
                kernel = "dequantize"
                arg("X", 2, "i32")
                arg("Row", 1)
                arg("Column", 1)
                arg("Out", 2)
                block = 128
                constants = dict(ROWS=rows, COLS=cols, BLOCK=block)
                grid = (cdiv(rows * cols, block), 1, 1)
    elif kind == "linear_f32":
        m, n, k = metadata["shape"]
        kernel = "linear"
        arg("A", 2)
        arg("W", 2)
        arg("Bias", 1)
        arg("Out", 2)
        bm, bn, bk = power_of_two(min(m, 16)), 16, min(64, power_of_two(max(k, 1)))
        constants = dict(M=m, N=n, K=k, BM=bm, BN=bn, BK=bk)
        grid = (cdiv(m, bm), cdiv(n, bn), 1)
    elif kind == "conv2d_f32":
        cv = metadata["conv"]
        kernel = "conv2d_nchw"
        arg("X", 4)
        arg("Weight", 4)
        arg("Bias", 1)
        arg("Out", 4)
        spat, oc = cv["OH"] * cv["OW"], cv["OC"]
        bm = min(16, power_of_two(max(spat, 1)))
        bn, bk = 16, 64
        constants = dict(H=cv["H"], W=cv["W"], C=cv["C"], OC=oc,
                         KH=cv["KH"], KW=cv["KW"], SH=cv["SH"], SW=cv["SW"],
                         PH=cv["PH"], PW=cv["PW"], OH=cv["OH"], OW=cv["OW"],
                         BM=bm, BN=bn, BK=bk)
        grid = (cdiv(spat, bm), cdiv(oc, bn), 1)
    elif kind == "dwconv2d_f32":
        cv = metadata["conv"]
        kernel = "dwconv2d_nchw"
        arg("X", 4)
        arg("Weight", 4)
        arg("Bias", 1)
        arg("Out", 4)
        constants = dict(H=cv["H"], W=cv["W"], C=cv["C"],
                         KH=cv["KH"], KW=cv["KW"], SH=cv["SH"], SW=cv["SW"],
                         PH=cv["PH"], PW=cv["PW"], OH=cv["OH"], OW=cv["OW"],
                         BLOCK=16)
        grid = (cv["OH"] * cv["OW"], cdiv(cv["C"], 16), 1)
    elif kind == "gap":
        _, channels, height, width = metadata["buffers"][0]["shape"]
        kernel = "gap"
        arg("X", 4)
        arg("Out", 2)
        block = power_of_two(min(channels, 128))
        constants = dict(SPATIAL=height * width, C=channels, BLOCK=block)
        grid = (cdiv(channels, block), 1, 1)
    elif kind == "se_scale":
        _, channels, height, width = metadata["buffers"][0]["shape"]
        kernel = "se_scale"
        arg("X", 4)
        arg("Scale", 2)
        arg("Out", 4)
        constants = dict(SPATIAL=height * width, C=channels, BLOCK=16)
        grid = (height * width, cdiv(channels, 16), 1)
    else:
        shape = metadata["buffers"][0]["shape"]
        count = product(shape)
        arg("X", len(shape))
        if kind in ("add", "mul"):
            arg("Y", len(shape))
            kernel = "binary"
            constants = dict(COUNT=count, MULTIPLY=kind == "mul", BLOCK=128)
        else:
            kernel = kind
            constants = dict(COUNT=count, BLOCK=128)
        arg("Out", len(metadata["buffers"][-1]["shape"]))
        grid = (cdiv(count, 128), 1, 1)

    pointer_type = {"f32": "*fp32", "i8": "*i8", "i32": "*i32", "i64": "*i64"}
    return {
        "name": name,
        "family": kind,
        "kernel": kernel,
        "kernel_module": kernel_module,
        "symbol": "triton_" + name,
        "arguments": arguments,
        "signature": {argument["name"]: pointer_type[argument["dtype"]]
                      for argument in arguments},
        "constexprs": constants,
        "grid": list(grid),
        "directory": str(directory),
        "launch_source": str(directory / "launch.c"),
        "validation": "Unmodified launch.c and its independent numerical oracle",
    }


def inventory():
    found = {}
    for root in case_search_roots():
        for path in sorted(root.glob("*/metadata.json")):
            found.setdefault(path.parent.name, path)
    return [describe(name) for name in sorted(found)]


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
