#!/usr/bin/env python3
# ===- run_graph_host.py - host EE for EfficientNet graph ---------*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Execute the Buddy-compiled EfficientNet forward through MLIR's
# ExecutionEngine and score logits against host_reference.py. With
# --replace, load libeffnet_triton_host.so from build_host_bridge.sh so
# the same Triton call sites the board uses run on the host.
#
# ===----------------------------------------------------------------------===//
"""Host-execute the Buddy EfficientNet graph and score against FP32 reference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def prepare_llvm_libs(repo_root, work_dir):
    import glob
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    needed = ["libmlir_runner_utils.so", "libmlir_c_runner_utils.so", "libomp.so"]
    found = {}
    search = [repo_root / "llvm/build-2d26/lib", repo_root / "llvm/build/lib"]
    for name in needed:
        for directory in search:
            candidate = directory / name
            if candidate.is_file():
                found[name] = candidate
                break
        if name not in found:
            for candidate in glob.glob(str(repo_root / f"llvm/**/{name}"),
                                       recursive=True):
                found[name] = Path(candidate)
                break
    for candidate in sorted(glob.glob(str(
            repo_root / "llvm/build*/runtimes/runtimes-bins/openmp/runtime/src/libomp.so"))):
        found.setdefault("libomp.so", Path(candidate))
    for name, source in found.items():
        link = work_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source)
    missing = [n for n in needed if n not in found]
    return str(work_dir), {n: str(p) for n, p in found.items()}, missing


def error_stats(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [list(reference.shape), list(candidate.shape)]}
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"nonfinite": True}
    diff = np.abs(reference - candidate)
    return {
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "elements": int(diff.size),
    }


class GraphRunner:
    def __init__(self, graph, llvm_libs_dir=None, opt_level=1, threads=8,
                 external_libs=()):
        from buddy_mlir.execution_engine import ExecutionEngine
        from buddy_mlir import runtime as rt
        import ctypes
        import platform

        self._rt = rt
        self._ctypes = ctypes
        if llvm_libs_dir:
            os.environ["LLVM_LIBS_DIR"] = llvm_libs_dir
        if threads:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["OMP_DYNAMIC"] = "FALSE"

        graph.op_groups = {"sg": list(graph.body)}
        from buddy.compiler.graph.type import DeviceType
        graph.group_map_device = {"sg": DeviceType.CPU}
        graph.lower_to_top_level_ir()
        graph.lower_to_llvm_ir()

        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        base = Path(llvm_libs_dir) if llvm_libs_dir else (
            Path(__file__).resolve().parents[5] / "llvm/build-2d26/lib")
        shared_libs = [
            str(base / ("libmlir_runner_utils" + extension)),
            str(base / ("libmlir_c_runner_utils" + extension)),
        ]
        openmp = base / ("libomp" + extension)
        if openmp.is_file():
            shared_libs.append(str(openmp))
        shared_libs.extend(str(lib) for lib in external_libs)
        self._engine = ExecutionEngine(
            graph._imported_module, opt_level=opt_level,
            shared_libs=shared_libs,
            enable_pic=platform.machine().startswith("riscv"),
        )
        self._graph = graph
        self.func_name = graph._func_name

    def __call__(self, *arrays):
        rt = self._rt
        ctypes = self._ctypes
        input_slots = []
        self._keepalive = [np.ascontiguousarray(a) for a in arrays]
        for array in self._keepalive:
            descriptor = rt.get_ranked_memref_descriptor(array)
            input_slots.append(ctypes.pointer(ctypes.pointer(descriptor)))
        output_struct = self._graph._output_descriptor()
        output_slot = ctypes.pointer(ctypes.pointer(output_struct))
        self._engine.invoke(self.func_name, output_slot, *input_slots)
        outputs = []
        for index in range(len(self._graph._output_memref)):
            descriptor = getattr(output_struct, str(index))
            value = rt.ranked_memref_to_numpy(ctypes.pointer(descriptor))
            outputs.append(np.array(value))
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="directory with logits.npy from host_reference.py")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--w8a8", action="store_true",
                        help="With --replace: rewrite linears to W8A8 kernels")
    parser.add_argument("--triton-build", type=Path, default=None)
    parser.add_argument("--external-lib", type=Path, action="append", default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--opt-level", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-abs-error", type=float, default=1e-3)
    args = parser.parse_args()

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import import_model as im
    import triton_call_replace as tcr
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph.operation import PlaceholderOp
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as decomp

    repo = args.repo_root or Path(__file__).resolve().parents[5]
    args.output.mkdir(parents=True, exist_ok=True)
    llvm_libs, lib_sources, missing = prepare_llvm_libs(
        repo, args.output / "llvm-libs")
    if missing:
        raise SystemExit("missing MLIR runtime libs: " + ", ".join(missing))

    model = im.build_model(args.checkpoint, args.stages)
    data = torch.from_numpy(im.board_input(args.input_size))
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decomp,
        func_name="forward",
        enable_external_calls=bool(args.replace),
    )
    with torch.no_grad():
        graph = compiler.importer(model, data)[0]
    params = list(compiler.imported_params.get(graph, []))

    replacement = None
    if args.replace:
        if not args.triton_build:
            raise SystemExit("--replace needs --triton-build")
        if args.w8a8 and not args.replace:
            raise SystemExit("--w8a8 requires --replace")
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
        if local["uncovered"]:
            raise SystemExit("replacement still has uncovered ops: "
                             + json.dumps(local["uncovered"][:4]))
        tcr.generate_adapters(call_types, args.output / "effnet_triton_adapters.c")
        replacement = {
            "pattern_matches": counts,
            "w8a8": bool(args.w8a8),
            "w8a8_linears": local.get("w8a8_linears", []),
            "external_calls": sum(1 for op in graph.body
                                  if type(op).__name__ == "CallExternalOp"),
            "status": "PASS",
        }

    # Keep only non-placeholder ops in the group used by the EE path.
    graph.op_groups = {"sg": [op for op in graph.body
                              if not isinstance(op, PlaceholderOp)]}

    external_libs = list(args.external_lib or [])
    started = time.time()
    runner = GraphRunner(graph, llvm_libs_dir=llvm_libs, opt_level=args.opt_level,
                         threads=args.threads, external_libs=external_libs)
    arrays = [np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))
              for t in params]
    arrays.append(np.ascontiguousarray(data.cpu().numpy().astype(np.float32)))
    outputs = runner(*arrays)
    elapsed = time.time() - started
    logits = outputs[0]
    np.save(args.output / "logits.npy", logits)

    reference = np.load(args.reference_dir / "logits.npy")
    stats = error_stats(reference, logits)
    predicted = int(np.argmax(logits.reshape(-1)))
    ref_class = int(np.argmax(reference.reshape(-1)))
    report = {
        "stage": "host execution of the Buddy-compiled EfficientNet graph",
        "replace": bool(args.replace),
        "replacement": replacement,
        "external_libs": [str(p) for p in external_libs],
        "jit_opt_level": args.opt_level,
        "elapsed_seconds": elapsed,
        "predicted_class": predicted,
        "reference_class": ref_class,
        "argmax_match": predicted == ref_class,
        "logits": stats,
        "mlir_runtime_libs": lib_sources,
        "status": "PASS" if (
            predicted == ref_class
            and stats.get("max_abs_error", 1e9) <= args.max_abs_error
            and "nonfinite" not in stats
            and "shape_mismatch" not in stats
        ) else "FAIL",
    }
    (args.output / "host-run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
