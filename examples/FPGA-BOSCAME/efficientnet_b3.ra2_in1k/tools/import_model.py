#!/usr/bin/env python3
# ===- import_model.py - timm EfficientNet to Buddy graph -------*- python -*-===//
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
# Fuse Conv+BN before DynamoCompiler so the graph is conv-with-bias, not
# the training-time BN chain. Tensors stay NCHW (Conv2dOp NCHW_FCHW).
# SiLU is imported as Mul(x, Sigmoid(x)). Writes raw-ops.json for
# emit_model_cases.py. board_input() is the deterministic NCHW fill the
# NR launch and host_reference.py --board-input must share.
#
# Requires this-repo buddy.compiler on PYTHONPATH (build-migrate
# python_packages) and the same CPython ABI as the LLVM bindings.
#
# ===----------------------------------------------------------------------===//
"""Import timm/efficientnet_b3.ra2_in1k into a Buddy graph."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def fuse_conv_bn(model):
    import torch
    from torch.nn.utils import fusion
    changed = True
    while changed:
        changed = False
        for name, module in list(model.named_modules()):
            kids = list(module.named_children())
            for index, (child_name, child) in enumerate(kids[:-1]):
                nxt_name, nxt = kids[index + 1]
                if isinstance(child, torch.nn.Conv2d) and isinstance(nxt, torch.nn.BatchNorm2d):
                    setattr(module, child_name, fusion.fuse_conv_bn_eval(child, nxt))
                    setattr(module, nxt_name, torch.nn.Identity())
                    changed = True
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, "num_batches_tracked"):
            del module.num_batches_tracked


def describe_graph(graph):
    ops = []
    by_name = {op.name: op for op in graph.body}
    for index, op in enumerate(graph.body):
        entry = {
            "index": index,
            "name": op.name,
            "op_type": type(op).__name__,
            "arguments": list(getattr(op, "args", [])),
        }
        meta = getattr(op, "_tensor_meta", None)
        if isinstance(meta, dict):
            shape = meta.get("shape")
            try:
                entry["shape"] = list(shape) if shape is not None else None
            except TypeError:
                entry["shape"] = str(shape)
            entry["dtype"] = str(meta.get("dtype"))
        if getattr(op, "_parents", None):
            entry["parents"] = list(op._parents)
            entry["parent_types"] = [type(by_name[p]).__name__
                                     for p in op._parents if p in by_name]
            entry["parent_shapes"] = []
            for parent in op._parents:
                if parent not in by_name:
                    continue
                parent_meta = getattr(by_name[parent], "_tensor_meta", None)
                if isinstance(parent_meta, dict) and parent_meta.get("shape") is not None:
                    try:
                        entry["parent_shapes"].append(list(parent_meta["shape"]))
                    except TypeError:
                        entry["parent_shapes"].append(str(parent_meta["shape"]))
        ops.append(entry)
    return ops


def histogram(graph):
    counts = {}
    for op in graph.body:
        name = type(op).__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def load_state(checkpoint: Path):
    import torch
    safetensors = checkpoint / "model.safetensors"
    if safetensors.is_file():
        from safetensors.torch import load_file
        return load_file(str(safetensors))
    binary = checkpoint / "pytorch_model.bin"
    if binary.is_file():
        return torch.load(binary, map_location="cpu")
    raise FileNotFoundError(
        f"no model.safetensors or pytorch_model.bin under {checkpoint}")


def board_input(input_size=288):
    """Deterministic NCHW image shared by host_reference and the NR launch.

    Formula: ``((i * 13) % 129 - 64) * 0.125`` over the flattened
    ``1×3×H×W`` buffer. The board launch must use the same fill so
    argmax can be checked without uploading an image tensor.
    """
    import numpy as np
    n = 3 * input_size * input_size
    index = np.arange(n, dtype=np.int64)
    values = ((index * 13) % 129 - 64).astype(np.float32) * np.float32(0.125)
    return values.reshape(1, 3, input_size, input_size)


def build_model(checkpoint: Path, stages=None):
    import timm
    model = timm.create_model("efficientnet_b3.ra2_in1k", pretrained=False)
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    fuse_conv_bn(model)
    if stages is not None:
        model.blocks = model.blocks[:stages]
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--stages", type=int, default=None,
                        help="keep stem + first N MBConv stages (debug)")
    parser.add_argument("--no-fuse", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.transform import simply_fuse
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as inductor_decomp

    if args.no_fuse:
        import timm
        model = timm.create_model("efficientnet_b3.ra2_in1k", pretrained=False)
        model.load_state_dict(load_state(args.checkpoint), strict=True)
        model.eval().requires_grad_(False)
        if args.stages is not None:
            model.blocks = model.blocks[:args.stages]
    else:
        model = build_model(args.checkpoint, args.stages)

    data = torch.randn(1, 3, args.input_size, args.input_size)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward",
    )
    started = time.time()
    with torch.no_grad():
        graphs = compiler.importer(model, data)
    graph = graphs[0]
    params = compiler.imported_params[graph]

    (args.output / "raw-ops.json").write_text(
        json.dumps(describe_graph(graph), indent=2, default=str) + "\n")
    report = {
        "import_seconds": round(time.time() - started, 3),
        "op_count": len(graph.body),
        "histogram": histogram(graph),
        "param_count": len(params),
        "param_shapes": [list(p.shape) for p in params],
        "input_shape": [1, 3, args.input_size, args.input_size],
        "stages": args.stages,
        "fused_conv_bn": not args.no_fuse,
    }
    (args.output / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")

    entries = []
    blobs = []
    for index, param in enumerate(params):
        array = param.detach().cpu().contiguous().numpy()
        raw = array.tobytes()
        entries.append({
            "index": index, "shape": list(array.shape), "dtype": str(array.dtype),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        })
        if array.dtype == np.float32:
            blobs.append(array.reshape(-1))
    (args.output / "params.json").write_text(json.dumps(entries, indent=2) + "\n")
    if blobs:
        np.concatenate(blobs).tofile(args.output / "arg0.data")

    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    (args.output / "subgraph0.mlir").write_text(str(driver.subgraphs[0]._imported_module) + "\n")
    (args.output / "forward.mlir").write_text(str(driver.construct_main_graph(True)) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
