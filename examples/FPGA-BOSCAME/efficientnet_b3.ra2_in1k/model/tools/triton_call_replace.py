#!/usr/bin/env python3
# ===- triton_call_replace.py - graph nodes to Triton calls -----*- python -*-===//
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
# Structural replacement only. A node is rewritten when:
#   1. dataflow matches a known pattern (conv, silu = Mul(Sigmoid), gap, add);
#   2. the exact shape exists as a built Triton case (frontend.json);
#   3. ranks and dtypes agree with that kernel's ABI;
#   4. the replacement is single-output.
#
# Failures stay in the graph and are listed as uncovered. Nothing falls
# back to a scalar C implementation. Graph-facing symbols are
# _mlir_ciface_effnet_graph_*; adapters call _mlir_ciface_kernel_<case>.
#
# PYTHONSAFEPATH / python -P omits this script's directory, so
# import_model is loaded via an explicit sys.path insert.
#
# ===----------------------------------------------------------------------===//
"""Replace matched EfficientNet subgraphs with compiled Triton kernel calls."""
import argparse
import json
import re
import sys
from pathlib import Path

# Same directory as import_model.py; required when PYTHONSAFEPATH=1.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_kernel_index(triton_build):
    index = {}
    for frontend in sorted(Path(triton_build).glob("*/frontend.json")):
        data = json.loads(frontend.read_text())
        index[data["name"]] = {
            "name": data["name"],
            "symbol": data["symbol"],
            "adapter_entry": "_mlir_ciface_kernel_" + data["name"],
            "arguments": data["arguments"],
            "constexprs": data["constexprs"],
            "grid": data["grid"],
            "frontend": str(frontend),
        }
    return index


def shape_of(op):
    meta = getattr(op, "tensor_meta", None)
    if not isinstance(meta, dict) or meta.get("shape") is None:
        return None
    try:
        return [int(v) for v in meta["shape"]]
    except (TypeError, ValueError):
        return None


def dtype_of(op):
    meta = getattr(op, "tensor_meta", None)
    if not isinstance(meta, dict):
        return None
    return str(meta.get("dtype"))


def is_float32(op):
    return "Float32" in (dtype_of(op) or "")


def parents_of(graph, op):
    return [graph.node_table[name] for name in (op._parents or [])
            if name in graph.node_table]


def children_of(graph, op):
    return [graph.node_table[name] for name in (op._children or [])
            if name in graph.node_table]


def returning_symbol(graph, op, kernel_symbol):
    site = f"{graph._func_name}_{op.name}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", site):
        raise ValueError(f"cannot form a stable C symbol for graph call site {site!r}")
    return kernel_symbol + "__" + site


def retarget_call(graph, call_op, operands):
    new_names = [operand.name for operand in operands]
    old_parents = list(call_op._parents or [])
    call_op._parents = []
    for name in new_names:
        call_op.add_parent(name)
        producer = graph.node_table[name]
        if call_op.name not in producer._children:
            producer.add_children(call_op.name)
    for name in old_parents:
        if name in new_names:
            continue
        producer = graph.node_table.get(name)
        if producer is not None and call_op.name in producer._children:
            producer._children.remove(call_op.name)
    call_op._arguments = new_names
    call_op._args_index = [0] * len(new_names)


def _row_major_stride(shape, axis):
    stride = 1
    for dim in shape[axis + 1:]:
        stride *= dim
    return stride


def match_silu(graph, op, index):
    if type(op).__name__ != "MulOp" or not is_float32(op):
        return None
    parents = parents_of(graph, op)
    if len(parents) != 2:
        return None
    first, second = parents
    sigmoid = second if type(second).__name__ == "SigmoidOp" else first
    if type(sigmoid).__name__ != "SigmoidOp":
        return None
    source = parents_of(graph, sigmoid)
    if len(source) != 1:
        return None
    source = source[0]
    if source.name not in (first.name, second.name):
        return None
    shape = shape_of(op)
    if not shape or shape_of(source) != shape:
        return None
    name = "silu_" + "x".join(map(str, shape))
    case = index.get(name)
    if case is None:
        return {"reason": f"no built silu case {name}", "shape": shape}
    return {
        "case": case, "symbol": "effnet_graph_" + name, "kind": "silu",
        "shape": shape, "operands": [source], "result": op,
        "interior": [sigmoid], "entry_args": ["a1", "a0"],
        "adapter": "reorder (result, X) -> (X, result)",
    }


def match_sigmoid(graph, op, index):
    if type(op).__name__ != "SigmoidOp" or not is_float32(op):
        return None
    for child in children_of(graph, op):
        if type(child).__name__ == "MulOp":
            return None
    parents = parents_of(graph, op)
    if len(parents) != 1:
        return None
    shape = shape_of(op)
    if not shape:
        return None
    name = "sigmoid_" + "x".join(map(str, shape))
    case = index.get(name)
    if case is None:
        return {"reason": f"no built sigmoid case {name}", "shape": shape}
    return {
        "case": case, "symbol": "effnet_graph_" + name, "kind": "sigmoid",
        "shape": shape, "operands": parents, "result": op, "interior": [],
        "entry_args": ["a1", "a0"],
        "adapter": "reorder (result, X) -> (X, result)",
    }


def match_add(graph, op, index):
    if type(op).__name__ != "AddOp" or not is_float32(op):
        return None
    parents = parents_of(graph, op)
    if len(parents) != 2:
        return None
    shape = shape_of(op)
    if not shape or shape_of(parents[0]) != shape or shape_of(parents[1]) != shape:
        return None
    name = "add_" + "x".join(map(str, shape))
    case = index.get(name)
    if case is None:
        return {"reason": f"no built add case {name}", "shape": shape}
    return {
        "case": case, "symbol": "effnet_graph_" + name, "kind": "add",
        "shape": shape, "operands": parents, "result": op, "interior": [],
        "entry_args": ["a1", "a2", "a0"],
        "adapter": "reorder (result, X, Y) -> (X, Y, result)",
    }


def match_se_scale(graph, op, index):
    if type(op).__name__ != "MulOp" or not is_float32(op):
        return None
    parents = parents_of(graph, op)
    if len(parents) != 2:
        return None
    left, right = parents
    left_shape, right_shape = shape_of(left), shape_of(right)
    if not left_shape or not right_shape:
        return None
    if type(left).__name__ == "SigmoidOp" or type(right).__name__ == "SigmoidOp":
        return None
    activation, scale = left, right
    if len(left_shape) != 4:
        activation, scale = right, left
        left_shape, right_shape = shape_of(activation), shape_of(scale)
    if not left_shape or len(left_shape) != 4:
        return None
    if len(right_shape) not in (2, 4):
        return None
    _, channels, height, width = left_shape
    name = f"se_scale_{height}x{width}x{channels}"
    case = index.get(name)
    if case is None:
        return {"reason": f"no built se_scale case {name}", "shape": left_shape}
    return {
        "case": case, "symbol": "effnet_graph_" + name, "kind": "se_scale",
        "shape": left_shape, "operands": [activation, scale], "result": op,
        "interior": [], "entry_args": ["a1", "a2", "a0"],
        "adapter": "reorder (result, X, Scale) -> (X, Scale, result)",
    }


def match_gap(graph, op, index):
    if type(op).__name__ not in ("AdaptiveAvgPool2dOp", "AvgPool2dOp", "MeanOp"):
        return None
    parents = parents_of(graph, op)
    if len(parents) != 1:
        return None
    inp = shape_of(parents[0])
    out = shape_of(op)
    if not inp or len(inp) != 4:
        return None
    if not out or not (len(out) == 2 or (len(out) == 4 and out[2] == 1 and out[3] == 1)):
        return None
    _, channels, height, width = inp
    name = f"gap_{height}x{width}x{channels}"
    case = index.get(name)
    if case is None:
        return {"reason": f"no built gap case {name}", "shape": inp}
    return {
        "case": case, "symbol": "effnet_graph_" + name, "kind": "gap",
        "shape": inp, "operands": parents, "result": op, "interior": [],
        "entry_args": ["a1", "a0"],
        "adapter": "reorder (result, X) -> (X, result)",
    }


def match_conv(graph, op, index):
    if type(op).__name__ != "Conv2dOp" or not is_float32(op):
        return None
    args = list(op.args)
    if len(args) < 9:
        return None
    parents = parents_of(graph, op)
    if len(parents) < 2:
        return None
    activation, weight = parents[0], parents[1]
    inp, wshape, out = shape_of(activation), shape_of(weight), shape_of(op)
    if not inp or not wshape or not out or len(inp) != 4 or len(wshape) != 4:
        return None
    _, cin, height, width = inp
    oc, ic, kh, kw = wshape
    stride, padding, groups = args[3], args[4], int(args[8])
    sh = int(stride[0]) if stride else 1
    sw = int(stride[1]) if stride and len(stride) > 1 else sh
    ph = int(padding[0]) if padding else 0
    pw = int(padding[1]) if padding and len(padding) > 1 else ph
    operands = [activation, weight]
    if len(parents) >= 3:
        operands.append(parents[2])
    if groups == cin and ic == 1:
        name = f"dwconv2d_{height}x{width}x{cin}_k{kh}_s{sh}"
        kind = "dwconv"
    else:
        name = f"conv2d_{height}x{width}_{cin}x{oc}_k{kh}_s{sh}"
        kind = "conv"
    case = index.get(name)
    if case is None:
        return {"reason": f"no built {kind} case {name}", "shape": out}
    if len(operands) < 3:
        return {"reason": f"{name} needs a fused bias parent", "shape": out}
    return {
        "case": case, "symbol": "effnet_graph_" + name, "kind": kind,
        "shape": [height, width, cin, oc, kh, sh], "operands": operands,
        "result": op, "interior": [],
        "entry_args": ["a1", "a2", "a3", "a0"],
        "adapter": "reorder (result, X, W, B) -> (X, W, B, result)",
        "conv": {"H": height, "W": width, "C": cin, "OC": oc,
                 "KH": kh, "KW": kw, "SH": sh, "SW": sw, "PH": ph, "PW": pw},
    }


def match_linear(graph, op, index):
    typename = type(op).__name__
    if typename not in ("MatmulOp", "AddMMOp") or not is_float32(op):
        return None
    parents = parents_of(graph, op)
    out = shape_of(op)
    if not out or len(out) != 2:
        return None
    m, n = out
    weight = None
    activation = None
    bias = None
    for parent in parents:
        pshape = shape_of(parent)
        if pshape is None:
            continue
        if len(pshape) == 1 and pshape[0] == n:
            bias = parent
        elif len(pshape) == 2 and pshape[0] == n:
            weight = parent
        elif len(pshape) == 2 and pshape[0] == m:
            activation = parent
    if weight is None or activation is None:
        return None
    k = shape_of(weight)[1]
    name = f"linear_{m}x{n}x{k}"
    case = index.get(name)
    if case is None:
        return {"reason": f"no built linear case {name}", "shape": [m, n, k]}
    if bias is None:
        return {"reason": f"{name} needs a bias parent", "shape": [m, n, k]}
    return {
        "case": case, "symbol": "effnet_graph_" + name, "kind": "linear",
        "shape": [m, n, k], "operands": [activation, weight, bias],
        "result": op, "interior": [],
        "entry_args": ["a1", "a2", "a3", "a0"],
        "adapter": "reorder (result, A, W, B) -> (A, W, B, result)",
    }


def replace(graph, matcher, index, call_types, report):
    from buddy.compiler.graph.operation import CallExternalOp, OpType

    matched = []
    for op in list(graph.body):
        found = matcher(graph, op, index)
        if found is None:
            continue
        matched.append((op, found))

    replaced = 0
    for op, found in matched:
        if "reason" in found:
            report["uncovered"].append({
                "kind": matcher.__name__.replace("match_", ""),
                "node": op.name,
                "shape": found["shape"],
                "reason": found["reason"],
            })
            continue
        found["symbol"] = returning_symbol(graph, op, found["symbol"])
        call_op = CallExternalOp(
            call_func_name=found["symbol"],
            args=[operand.name for operand in found["operands"]],
            args_index=[0] * len(found["operands"]),
            tensor_meta=dict(op.tensor_meta) if isinstance(op.tensor_meta, dict)
            else {"shape": shape_of(op), "dtype": dtype_of(op)},
            name=op.name,
        )
        graph.displace_node(op, call_op)
        call_op._op_type = OpType.Unfusable
        retarget_call(graph, call_op, found["operands"])
        for interior in found.get("interior", []):
            if interior.name not in graph.node_table:
                continue
            remaining = [child for child in children_of(graph, interior)
                         if child.name in graph.node_table]
            if remaining:
                report["kept_interior"].append(
                    {"node": interior.name,
                     "still_used_by": [child.name for child in remaining]})
                continue
            graph.delete_node(interior, parents_of(graph, interior))
        found["graph_ranks"] = [len(shape_of(o)) for o in found["operands"]]
        found["result_rank"] = len(shape_of(op))
        found["result_shape"] = shape_of(op)
        call_types.setdefault(found["symbol"], found)
        report["replaced"].append({
            "node": op.name, "kind": found["kind"], "shape": found["shape"],
            "symbol": found["symbol"],
            "c_symbol": "_mlir_ciface_" + found["symbol"],
            "archive_entry": found["case"]["adapter_entry"],
            "operands": [o.name for o in found["operands"]],
        })
        replaced += 1
    return replaced


def generate_adapters(call_types, output):
    lines = [
        "/* Generated by model/tools/triton_call_replace.py -- do not edit. */",
        '#include "support.h"',
        "",
    ]
    total = 0
    for symbol, found in sorted(call_types.items()):
        if found.get("void"):
            continue
        shape = found["result_shape"]
        count = 1
        for dim in shape:
            count *= dim
        lines.append(f"static float effnet_out_{symbol}[{count}];")
        total += count
    if total:
        lines.append("")
    declared = set()
    for found in call_types.values():
        entry = found["case"]["adapter_entry"]
        if entry in declared:
            continue
        declared.add(entry)
        ranks = [a["rank"] for a in found["case"]["arguments"]]
        lines.append(f"extern void {entry}({', '.join(f'MemRef{r} *' for r in ranks)});")
    if declared:
        lines.append("")
    for symbol, found in sorted(call_types.items()):
        entry = found["case"]["adapter_entry"]
        case_ranks = [a["rank"] for a in found["case"]["arguments"]]
        graph_ranks = found["graph_ranks"]
        result_rank = found["result_rank"]
        shape = found["result_shape"]
        if found.get("void"):
            params = ", ".join(f"MemRef{r} *a{i}"
                               for i, r in enumerate(graph_ranks))
            lines.append(f"void _mlir_ciface_{symbol}({params}) {{")
            lines.append("  " + entry + "(" + ", ".join(
                f"(MemRef{case_ranks[i]} *)a{i}"
                for i in range(len(case_ranks))) + ");")
            lines.append("}")
            lines.append("")
            continue
        params = ", ".join(f"MemRef{r} *a{i}"
                           for i, r in enumerate([result_rank] + list(graph_ranks)))
        lines.append(f"void _mlir_ciface_{symbol}({params}) {{")
        lines.append(f"  float *out = effnet_out_{symbol};")
        lines.append("  a0->allocated = (void *)0;")
        lines.append("  a0->aligned = out;")
        lines.append("  a0->offset = 0;")
        for i, dim in enumerate(shape):
            lines.append(f"  a0->sizes[{i}] = {dim};")
            lines.append(f"  a0->strides[{i}] = {_row_major_stride(shape, i)};")
        call_args = [
            f"(MemRef{case_ranks[i]} *){name}"
            for i, name in enumerate(found["entry_args"])
        ]
        lines.append(f"  {entry}({', '.join(call_args)});")
        lines.append("}")
        lines.append("")
    output.write_text("\n".join(lines) + "\n")
    return total


def workspace_record(node):
    return {
        "name": node.name,
        "shape": list(shape_of(node)),
        "dtype": str(node.tensor_meta.get("dtype")),
        "role": getattr(node, "workspace_role", "kernel_output"),
        "initialization": ("zero, every graph invocation"
                           if getattr(node, "workspace_role", "") == "accumulator"
                           else "kernel_output"),
    }


def _reindex(graph, params, inputs):
    index_of = {id(node): i for i, node in enumerate(graph.body)}
    graph._fake_params = sorted(index_of[id(node)] for node in params
                                if id(node) in index_of)
    graph._inputs = sorted(index_of[id(node)] for node in inputs
                           if id(node) in index_of)


def _placeholder_anchor(graph, parameters):
    known = {id(node) for node in parameters}
    for node in graph.body:
        if id(node) not in known:
            return node
    return None


def _input_anchor(graph, inputs):
    for node in graph.body:
        if type(node).__name__ != "PlaceholderOp":
            return node
    return None


def _new_node(graph, cls, name, arguments, shape, dtype, before=None):
    import torch
    node = cls()
    node.name = name
    node._arguments = list(arguments)
    node.tensor_meta = {"shape": torch.Size(shape), "dtype": dtype}
    graph.node_table[name] = node
    if before is None:
        graph.body.append(node)
    else:
        graph.body.insert(graph.body.index(before), node)
    return node


def match_linear_for_w8a8(graph, op):
    """Locate MatmulOp / AddMMOp linears for W8A8 rewrite (no FP32 case required)."""
    typename = type(op).__name__
    if typename not in ("MatmulOp", "AddMMOp") or not is_float32(op):
        return None
    parents = parents_of(graph, op)
    out = shape_of(op)
    if not out or len(out) != 2:
        return None
    m, n = out
    weight = activation = bias = None
    for parent in parents:
        pshape = shape_of(parent)
        if pshape is None:
            continue
        if len(pshape) == 1 and pshape[0] == n:
            bias = parent
        elif len(pshape) == 2 and pshape[0] == n:
            weight = parent
        elif len(pshape) == 2 and pshape[0] == m:
            activation = parent
    if weight is None or activation is None:
        return None
    k = shape_of(weight)[1]
    return {
        "shape": [m, n, k],
        "operands": [activation, weight],
        "bias": bias,
        "kind": "linear-w8a8",
    }


def rewrite_linear_to_w8a8(graph, index, call_types, report, tied, only=None):
    """quantize -> matmul_i8 -> dequantize (+ optional FP32 bias AddOp).

    Same contract as qwen3-0.6b; graph symbols use effnet_graph_.
    """
    import os
    from buddy.compiler.graph import TensorDType
    from buddy.compiler.graph.operation import AddOp, CallExternalOp, PlaceholderOp

    _debug = os.environ.get("W8A8_DEBUG") == "1"
    params_before = list(graph.params)
    inputs_before = list(graph.inputs)
    added_inputs = []
    if "original_parameters" not in report:
        report["original_parameters"] = [p.name for p in params_before]
    added_params = []
    matched = 0
    for op in list(graph.body):
        if type(op).__name__ not in ("MatmulOp", "AddMMOp"):
            continue
        if only is not None and op.name not in only:
            continue
        found = match_linear_for_w8a8(graph, op)
        if found is None:
            continue
        activation, weight = found["operands"]
        bias = found.get("bias")
        rows, k = shape_of(activation)
        n = shape_of(weight)[0]

        quantise_case = index.get(f"quantize_{rows}x{k}")
        matmul_case = index.get(f"matmul_{rows}x{n}x{k}")
        dequant_case = index.get(f"dequantize_{rows}x{n}")
        missing = [name for name, case in (
            (f"quantize_{rows}x{k}", quantise_case),
            (f"matmul_{rows}x{n}x{k}", matmul_case),
            (f"dequantize_{rows}x{n}", dequant_case)) if case is None]
        if missing:
            report["uncovered"].append({
                "kind": "linear-w8a8", "node": op.name,
                "shape": found["shape"],
                "reason": "no built case(s): " + ", ".join(missing)})
            continue

        prefix = f"w8a8_{op.name}"
        shared = tied.get(weight.name)
        if shared is not None:
            w8 = graph.node_table[shared[0]]
            ws = graph.node_table[shared[1]]
        else:
            anchor = _placeholder_anchor(graph, params_before)
            w8 = _new_node(graph, PlaceholderOp, prefix + "_weight_i8", [],
                           shape_of(weight), TensorDType.Int8, before=anchor)
            ws = _new_node(graph, PlaceholderOp, prefix + "_weight_scale", [], [n],
                           TensorDType.Float32, before=anchor)
            added_params.extend([w8, ws])
            tied[weight.name] = (w8.name, ws.name)

        anchor = _input_anchor(graph, inputs_before)
        workspace = []
        for suffix, shape, dtype in (
                ("_q", [rows, k], TensorDType.Int8),
                ("_a_scale", [rows], TensorDType.Float32),
                ("_acc", [rows, n], TensorDType.Int32),
                ("_out", [rows, n], TensorDType.Float32)):
            workspace.append(_new_node(graph, PlaceholderOp, prefix + suffix, [],
                                       shape, dtype, before=anchor))
        quantised, act_scale, accumulator, result = workspace
        accumulator.workspace_role = "accumulator"
        added_inputs.extend(workspace)

        def call(case, operands):
            name = f"{prefix}_{case['name']}"
            node = CallExternalOp(
                call_func_name="effnet_graph_" + name,
                args=[operand.name for operand in operands],
                args_index=[0] * len(operands),
                tensor_meta={},
                name=name)
            graph.node_table[name] = node
            graph.body.insert(graph.body.index(op), node)
            for operand in operands:
                node.add_parent(operand.name)
                operand.add_children(name)
            call_types["effnet_graph_" + name] = {
                "case": case,
                "kind": "w8a8-" + case["name"].split("_")[0],
                "shape": [len(operands)],
                "operands": list(operands),
                "graph_ranks": [len(shape_of(o)) for o in operands],
                "result_shape": None,
                "result_rank": 0,
                "void": True,
                "entry_args": [f"a{i}" for i in range(len(operands))],
                "adapter": "void call: every destination is an argument",
            }
            report.setdefault("w8a8_calls", []).append({
                "node": name, "case": case["name"],
                "operands": [o.name for o in operands],
            })
            return node

        call(quantise_case, [activation, quantised, act_scale])
        call(matmul_case, [quantised, w8, accumulator])
        call(dequant_case, [accumulator, act_scale, ws, result])

        consumer = result
        if bias is not None:
            add_name = prefix + "_bias_add"
            add_op = AddOp()
            add_op.name = add_name
            add_op._arguments = [result.name, bias.name]
            add_op.tensor_meta = dict(op.tensor_meta) if isinstance(op.tensor_meta, dict) \
                else {"shape": shape_of(op), "dtype": dtype_of(op)}
            graph.node_table[add_name] = add_op
            graph.body.insert(graph.body.index(op), add_op)
            add_op.add_parent(result.name)
            add_op.add_parent(bias.name)
            result.add_children(add_name)
            bias.add_children(add_name)
            consumer = add_op
            report.setdefault("w8a8_bias_adds", []).append({
                "node": add_name, "bias": bias.name, "shape": [rows, n]})

        for child in children_of(graph, op):
            child._arguments = [consumer.name if a == op.name else a
                                for a in child.args]
            child._parents[:] = [consumer.name if p == op.name else p
                                 for p in child._parents]
            consumer.add_children(child.name)
        if op.name in graph.node_table:
            graph.delete_node(op, parents_of(graph, op))
        matched += 1
        report.setdefault("w8a8_linears", []).append({
            "node": op.name, "rows": rows, "n": n, "k": k,
            "cases": [quantise_case["name"], matmul_case["name"],
                      dequant_case["name"]],
            "weight_param": weight.name,
            "weight_param_name": w8.name,
            "scale_param_name": ws.name,
            "bias": None if bias is None else bias.name,
        })
        if _debug:
            print(f"[w8a8] replaced {op.name} {rows}x{n}x{k}", file=sys.stderr)

    _reindex(graph, params_before + added_params, inputs_before + added_inputs)
    if added_inputs:
        report.setdefault("w8a8_workspace_inputs", []).extend(
            workspace_record(node) for node in added_inputs)
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--triton-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--w8a8", action="store_true",
                        help="Rewrite Matmul/AddMM to quantize/matmul_i8/dequantize")
    parser.add_argument("--w8a8-only", action="append", default=None,
                        help="Only rewrite these node names (repeatable)")
    args = parser.parse_args()

    import torch
    import import_model as im
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.operation import PlaceholderOp
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as decomp

    index = load_kernel_index(args.triton_build)
    model = im.build_model(args.checkpoint, args.stages)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decomp,
        func_name="forward",
        enable_external_calls=True,
    )
    data = torch.from_numpy(im.board_input(args.input_size))
    with torch.no_grad():
        graph = compiler.importer(model, data)[0]
    graph._enable_external_calls = True

    matchers = [
        ("conv", match_conv),
        ("linear", match_linear),
        ("silu", match_silu),
        ("se_scale", match_se_scale),
        ("sigmoid", match_sigmoid),
        ("gap", match_gap),
        ("add", match_add),
    ]
    report = {
        "stage": "graph -> external Triton call replacement",
        "triton_build": str(args.triton_build),
        "kernel_index_size": len(index),
        "w8a8": bool(args.w8a8),
        "replaced": [],
        "uncovered": [],
        "kept_interior": [],
        "pattern_matches": {},
    }
    call_types = {}
    before = len(graph.body)
    if args.w8a8:
        report["pattern_matches"]["w8a8"] = rewrite_linear_to_w8a8(
            graph, index, call_types, report, {}, only=args.w8a8_only)
    for name, matcher in matchers:
        if args.w8a8 and name == "linear":
            continue
        report["pattern_matches"][name] = replace(
            graph, matcher, index, call_types, report)

    graph.op_groups = {"sg": [op for op in graph.body if not isinstance(op, PlaceholderOp)]}
    graph.group_map_device = {"sg": DeviceType.CPU}
    graph.lower_to_top_level_ir()
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "subgraph0.triton.mlir").write_text(
        str(driver.subgraphs[0]._imported_module) + "\n")
    (args.output / "forward.triton.mlir").write_text(
        str(driver.construct_main_graph(True)) + "\n")
    generate_adapters(call_types, args.output / "effnet_triton_adapters.c")
    report.update({
        "ops_before": before,
        "ops_after": len(graph.body),
        "external_calls": sum(1 for op in graph.body if type(op).__name__ == "CallExternalOp"),
        "distinct_symbols": sorted(call_types),
        "adapter_source": str(args.output / "effnet_triton_adapters.c"),
        "status": ("PARTIAL" if report["uncovered"]
                   else "PASS" if call_types else "NO_MATCHES"),
    })
    (args.output / "triton-call-replacement.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("replaced", "uncovered", "kept_interior",
                                   "w8a8_calls", "w8a8_linears",
                                   "w8a8_workspace_inputs", "w8a8_bias_adds")},
                     indent=2))
    print(f"replaced {len(report['replaced'])} nodes, "
          f"w8a8_linears={len(report.get('w8a8_linears', []))}, "
          f"{len(report['distinct_symbols'])} distinct symbols")
    if report["uncovered"]:
        print(f"uncovered {len(report['uncovered'])} (first 8):")
        for entry in report["uncovered"][:8]:
            print("   ", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
