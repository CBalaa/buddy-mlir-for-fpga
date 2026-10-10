#!/usr/bin/env python3
# ===- emit_model_cases.py - graph ops to Triton case dirs ------*- python -*-===//
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
# Read raw-ops.json from import_model.py and emit operator case directories
# at the *real* model shapes (kernel.mlir + launch.c). The committed closed
# set is the example root next to tools/; run_full.sh uses that by default
# and only re-emits when REEMIT=1. Temporary trees under model/build/
# remain gitignored.
#
# PYTHONSAFEPATH omits this script's directory; generate_cases is loaded
# from the same directory via an explicit sys.path insert.
#
# ===----------------------------------------------------------------------===//
"""Emit Triton case directories from a captured Buddy op dump."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_cases import conv2d, dwconv2d, gap, linear, pointwise, se_scale


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def emit_from_ops(ops, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    created = []

    def once(name, builder):
        if name in seen:
            return
        seen.add(name)
        created.append(builder())

    by_name = {op["name"]: op for op in ops}
    for op in ops:
        kind = op.get("op_type")
        shape = as_list(op.get("shape"))
        args = op.get("arguments") or []
        parents = op.get("parent_shapes") or []

        if kind == "Conv2dOp" and shape and len(shape) == 4 and len(parents) >= 2:
            inp, weight = as_list(parents[0]), as_list(parents[1])
            if not inp or not weight or len(inp) != 4 or len(weight) != 4:
                continue
            _, cin, height, width = inp
            oc, ic, kh, kw = weight
            stride = as_list(args[3]) if len(args) > 3 else [1, 1]
            padding = as_list(args[4]) if len(args) > 4 else [0, 0]
            groups = int(args[8]) if len(args) > 8 else 1
            sh, sw = int(stride[0]), int(stride[1] if len(stride) > 1 else stride[0])
            ph = int(padding[0])
            pw = int(padding[1] if len(padding) > 1 else padding[0])
            if groups == cin and ic == 1:
                name = f"dwconv2d_{height}x{width}x{cin}_k{kh}_s{sh}"
                once(name, lambda h=height, w=width, c=cin, k=kh, s=sh, p=ph, q=pw:
                     dwconv2d(output, h, w, c, k, k, s, s, p, q))
            else:
                name = f"conv2d_{height}x{width}_{cin}x{oc}_k{kh}_s{sh}"
                once(name, lambda h=height, w=width, c=cin, o=oc, k=kh, s=sh, p=ph, q=pw:
                     conv2d(output, h, w, c, o, k, k, s, s, p, q))

        elif kind in ("AdaptiveAvgPool2dOp", "AvgPool2dOp", "MeanOp") and parents:
            inp = as_list(parents[0])
            if inp and len(inp) == 4 and shape and (len(shape) == 2 or
                                                    (len(shape) == 4 and shape[2] == 1 and shape[3] == 1)):
                _, channels, height, width = inp
                name = f"gap_{height}x{width}x{channels}"
                once(name, lambda h=height, w=width, c=channels: gap(output, h, w, c))

        elif kind in ("AddMMOp", "MatmulOp") and shape and len(shape) == 2:
            # A[M,K] x W^T[K,N] -> [M,N]; weight is usually [N,K]
            m, n = shape
            k = None
            for parent in parents:
                parent = as_list(parent)
                if parent and len(parent) == 2 and parent[0] == n:
                    k = parent[1]
                    break
                if parent and len(parent) == 2 and parent[0] == m:
                    k = parent[1]
            if k:
                name = f"linear_{m}x{n}x{k}"
                once(name, lambda a=m, b=n, c=k: linear(output, a, b, c))

        elif kind == "MulOp" and shape:
            parent_types = op.get("parent_types") or []
            if "SigmoidOp" in parent_types:
                name = "silu_" + "x".join(map(str, shape))
                once(name, lambda s=tuple(shape): pointwise(output, "silu", s))
            elif len(parents) == 2:
                left, right = as_list(parents[0]), as_list(parents[1])
                if left and right and len(left) == 4 and len(right) in (2, 4):
                    _, channels, height, width = left
                    name = f"se_scale_{height}x{width}x{channels}"
                    once(name, lambda h=height, w=width, c=channels: se_scale(output, h, w, c))
                else:
                    name = "mul_" + "x".join(map(str, shape))
                    once(name, lambda s=tuple(shape): pointwise(output, "mul", s))

        elif kind == "SigmoidOp" and shape:
            children_need_silu = any(
                other.get("op_type") == "MulOp" and op["name"] in (other.get("parents") or [])
                for other in ops)
            if not children_need_silu:
                name = "sigmoid_" + "x".join(map(str, shape))
                once(name, lambda s=tuple(shape): pointwise(output, "sigmoid", s))

        elif kind == "AddOp" and shape:
            name = "add_" + "x".join(map(str, shape))
            once(name, lambda s=tuple(shape): pointwise(output, "add", s))

    index = {"count": len(created), "cases": [item["name"] for item in created]}
    (output / "case-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, required=True, help="raw-ops.json from import_model.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ops = json.loads(args.ops.read_text())
    emit_from_ops(ops, args.output)


if __name__ == "__main__":
    main()
