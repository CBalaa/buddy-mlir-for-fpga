#!/usr/bin/env python3
# ===- lower_model_nr.py - replaced graph to RISC-V object ------*- python -*-===//
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
# Re-import, apply the same replacements as triton_call_replace.py, dump
# weights.bin + weight-layout.json, buddy-translate to LLVM IR, llc to a
# RISC-V object. llvm-nm then classifies remaining undefined symbols
# (Triton adapters, malloc/expf). Linking the bare-metal image is
# build_model_image.py. Uses this-repo llvm/build-2d26 (needs llvm-nm).
#
# ===----------------------------------------------------------------------===//
"""Lower the (optionally replaced) EfficientNet graph toward an NR object."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

OPENMP_PREFIX = "__kmpc_"
HEAP = {"malloc", "free", "calloc", "realloc", "aligned_alloc", "posix_memalign"}
LIBC_LIBM = {"memcpy", "memset", "memmove", "expf", "logf", "sinf", "cosf",
             "sqrtf", "tanhf", "memrefCopy", "fmodf", "powf", "floorf"}


def run(command, **kwargs):
    return subprocess.run([str(c) for c in command], capture_output=True,
                          text=True, **kwargs)


def classify(symbols):
    groups = {"openmp_runtime": [], "heap": [], "libc_libm": [],
              "effnet_graph_kernels": [], "other": []}
    for name in sorted(symbols):
        if name.startswith(OPENMP_PREFIX):
            groups["openmp_runtime"].append(name)
        elif name in HEAP:
            groups["heap"].append(name)
        elif name.startswith("_mlir_ciface_effnet_graph_"):
            groups["effnet_graph_kernels"].append(name)
        elif name in LIBC_LIBM:
            groups["libc_libm"].append(name)
        else:
            groups["other"].append(name)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--triton-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--no-replace", action="store_true")
    parser.add_argument("--w8a8", action="store_true",
                        help="Rewrite Matmul/AddMM to quantize/matmul_i8/dequantize")
    parser.add_argument("--openmp", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import import_model as im
    import triton_call_replace as tcr
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph.operation import PlaceholderOp
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as decomp

    repo = args.repo_root or Path(__file__).resolve().parents[4]
    llvm_bin = repo / "llvm/build-2d26/bin"
    buddy_bin = repo / "build-migrate/bin"
    args.output.mkdir(parents=True, exist_ok=True)

    model = im.build_model(args.checkpoint, args.stages)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decomp,
        func_name="forward",
        enable_external_calls=not args.no_replace,
    )
    data = torch.from_numpy(im.board_input(args.input_size))
    with torch.no_grad():
        graph = compiler.importer(model, data)[0]
    params = list(compiler.imported_params.get(graph, []))

    replacement = None
    if not args.no_replace:
        graph._enable_external_calls = True
        index = tcr.load_kernel_index(args.triton_build)
        local = {"replaced": [], "uncovered": [], "kept_interior": []}
        call_types = {}
        counts = {}
        if args.w8a8:
            counts["w8a8"] = tcr.rewrite_linear_to_w8a8(
                graph, index, call_types, local, {})
        for name, matcher in (("conv", tcr.match_conv),
                              ("linear", tcr.match_linear),
                              ("silu", tcr.match_silu),
                              ("se_scale", tcr.match_se_scale),
                              ("sigmoid", tcr.match_sigmoid),
                              ("gap", tcr.match_gap),
                              ("add", tcr.match_add)):
            if args.w8a8 and name == "linear":
                continue
            counts[name] = tcr.replace(graph, matcher, index, call_types, local)
        if args.w8a8:
            params_now, inputs_now = list(graph.params), list(graph.inputs)
            tcr._reindex(graph, params_now, inputs_now)
            local["workspace_inputs"] = local.get("w8a8_workspace_inputs") or []
        replacement = {
            "pattern_matches": counts,
            "uncovered": local["uncovered"],
            "w8a8": bool(args.w8a8),
            "w8a8_linears": local.get("w8a8_linears", []),
            "external_calls": sum(1 for op in graph.body
                                  if type(op).__name__ == "CallExternalOp"),
        }
        tcr.generate_adapters(call_types, args.output / "effnet_triton_adapters.c")

    graph.op_groups = {"sg": [op for op in graph.body
                              if not isinstance(op, PlaceholderOp)]}
    graph.group_map_device = {"sg": DeviceType.CPU}
    graph.lower_to_top_level_ir()
    graph.lower_to_llvm_ir(enable_openmp=args.openmp)

    sections, blobs, offset = [], [], 0
    for index, tensor in enumerate(params):
        array = tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        raw = array.tobytes()
        sections.append({
            "index": index, "shape": list(array.shape), "dtype": "f32",
            "offset_bytes": offset, "offset_elements": offset // 4,
            "bytes": len(raw),
        })
        blobs.append(array.reshape(-1))
        offset += len(raw)
    offset = (offset + 63) & ~63
    weights_path = args.output / "weights.bin"
    packed = np.concatenate(blobs) if blobs else np.zeros(0, dtype=np.float32)
    padded = np.zeros(offset // 4, dtype=np.float32)
    padded[:packed.size] = packed
    padded.tofile(weights_path)
    import hashlib
    layout = {
        "bytes": offset,
        "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "param_count": len(sections),
        "sections": sections,
        "convention": "result_then_params_then_inputs",
        "inputs": [{"name": "image", "rank": 4,
                    "shape": [1, 3, args.input_size, args.input_size]}],
        "outputs": [{"name": "logits", "rank": 2, "shape": [1, 1000]}],
    }
    (args.output / "weight-layout.json").write_text(json.dumps(layout, indent=2) + "\n")

    module_path = args.output / "forward.llvm.mlir"
    module_path.write_text(str(graph._imported_module))
    ll_path = args.output / "forward.ll"
    steps = {}
    result = run([buddy_bin / "buddy-translate", "--buddy-to-llvmir",
                  module_path, "-o", ll_path])
    steps["buddy_translate"] = {"returncode": result.returncode,
                                "stderr": result.stderr[-2000:]}
    if result.returncode != 0 or not ll_path.is_file():
        report = {"status": "FAILED at buddy-translate", "steps": steps,
                  "replacement": replacement}
        (args.output / "nr-lower.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    object_path = args.output / "forward.o"
    result = run([llvm_bin / "llc", ll_path, "-O2", "-filetype=obj",
                  "-mtriple=riscv64", "-target-abi=lp64d",
                  "-mattr=+m,+a,+f,+d,+c,+v,+zvl512b,+xboscame",
                  "-riscv-v-vector-bits-min=512", "-code-model=medium",
                  "-o", object_path])
    steps["llc"] = {"returncode": result.returncode, "stderr": result.stderr[-2000:]}
    if result.returncode != 0 or not object_path.is_file():
        report = {"status": "FAILED at llc", "steps": steps,
                  "replacement": replacement}
        (args.output / "nr-lower.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    undefined = run([llvm_bin / "llvm-nm", "--undefined-only", object_path])
    undef = sorted({line.split()[-1] for line in undefined.stdout.splitlines()
                    if line.strip()})
    groups = classify(undef)
    report = {
        "stage": "model graph -> RISC-V object (stops before linking a board image)",
        "replacement": replacement,
        "openmp_enabled": args.openmp,
        "artifacts": {
            "llvm_dialect_mlir": str(module_path),
            "llvm_ir": str(ll_path),
            "riscv_object": str(object_path),
            "weights": str(weights_path),
            "weight_layout": str(args.output / "weight-layout.json"),
        },
        "steps": steps,
        "symbols": {"undefined_total": len(undef), "undefined_by_provider": groups},
        "status": "object built; not linkable as a bare-metal image yet",
    }
    (args.output / "nr-lower.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
