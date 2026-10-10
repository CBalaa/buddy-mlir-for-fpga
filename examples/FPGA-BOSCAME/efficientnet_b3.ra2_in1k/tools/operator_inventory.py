#!/usr/bin/env python3
# ===- operator_inventory.py - EfficientNet graph coverage --------*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Walk raw-ops.json from import_model.py and report which ops map to known
# Triton / linalg case kinds. Unmatched compute ops stay uncovered.
#
# ===----------------------------------------------------------------------===//
"""Inventory EfficientNet Buddy graph ops against known FPGA case kinds."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VIEW_OPS = {
    "ViewOp", "PermuteOp", "UnsqueezeOp", "SliceOp", "ExpandOp",
    "AliasOp", "CloneOp", "ReshapeOp", "IdentityOp", "DetachOp",
}


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def classify(op, ops_by_name):
    kind = op.get("op_type")
    shape = as_list(op.get("shape"))
    parents = op.get("parent_shapes") or []
    parent_types = op.get("parent_types") or []
    args = op.get("arguments") or []

    if kind in VIEW_OPS or kind in ("PlaceholderOp", "OutputOp", "GetAttrOp"):
        return "view_or_io", None

    if kind == "Conv2dOp" and shape and len(shape) == 4 and len(parents) >= 2:
        inp, weight = as_list(parents[0]), as_list(parents[1])
        if inp and weight and len(inp) == 4 and len(weight) == 4:
            _, cin, height, width = inp
            oc, ic, kh, _kw = weight
            groups = int(args[8]) if len(args) > 8 else 1
            if groups == cin and ic == 1:
                return "dwconv2d_f32", f"dwconv2d_{height}x{width}x{cin}_k{kh}"
            return "conv2d_f32", f"conv2d_{height}x{width}_{cin}x{oc}_k{kh}"

    if kind in ("AdaptiveAvgPool2dOp", "AvgPool2dOp", "MeanOp") and parents:
        inp = as_list(parents[0])
        if inp and len(inp) == 4:
            _, channels, height, width = inp
            return "gap", f"gap_{height}x{width}x{channels}"

    if kind in ("AddMMOp", "MatmulOp") and shape and len(shape) == 2:
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
            return "linear_f32", f"linear_{m}x{n}x{k}"

    if kind == "MulOp" and shape:
        if "SigmoidOp" in parent_types:
            return "silu", "silu_" + "x".join(map(str, shape))
        if len(parents) == 2:
            left, right = as_list(parents[0]), as_list(parents[1])
            if left and right and len(left) == 4 and len(right) in (2, 4):
                _, channels, height, width = left
                return "se_scale", f"se_scale_{height}x{width}x{channels}"
            return "mul", "mul_" + "x".join(map(str, shape))

    if kind == "SigmoidOp" and shape:
        children_need_silu = any(
            other.get("op_type") == "MulOp" and op["name"] in (other.get("parents") or [])
            for other in ops_by_name.values())
        if children_need_silu:
            return "silu_interior", None
        return "sigmoid", "sigmoid_" + "x".join(map(str, shape))

    if kind == "AddOp" and shape:
        return "add", "add_" + "x".join(map(str, shape))

    return "unmatched", None


def inventory(ops):
    by_name = {op["name"]: op for op in ops}
    histogram = Counter()
    cases = Counter()
    unmatched = []
    for op in ops:
        family, case = classify(op, by_name)
        histogram[family] += 1
        if case:
            cases[case] += 1
        elif family == "unmatched":
            unmatched.append({
                "name": op.get("name"),
                "op_type": op.get("op_type"),
                "shape": op.get("shape"),
            })
    return {
        "op_count": len(ops),
        "family_histogram": dict(histogram.most_common()),
        "distinct_case_shapes": len(cases),
        "case_histogram": dict(cases.most_common()),
        "unmatched_compute": unmatched,
        "status": "PASS" if not unmatched else "HAS_UNMATCHED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, required=True,
                        help="raw-ops.json from import_model.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ops = json.loads(args.ops.read_text())
    report = inventory(ops)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "op_count": report["op_count"],
        "distinct_case_shapes": report["distinct_case_shapes"],
        "unmatched": len(report["unmatched_compute"]),
        "output": str(args.output),
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
