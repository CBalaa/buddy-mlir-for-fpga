#!/usr/bin/env python3
"""Lower a replaced ViT graph through Buddy to MLIR/LLVM.

This is the model-level counterpart of ``triton/build.py``.  It intentionally
never inserts arithmetic: all numerical operations are already in the Triton
kernel objects and this stage only runs Buddy's bufferization/lowering passes.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path

def run(command, output=None):
    result = subprocess.run([str(x) for x in command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if output is not None: output.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("command failed: " + " ".join(map(str, command)) + "\n" + result.stdout)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buddy-opt", default=os.environ.get("BUDDY_OPT", "buddy-opt"))
    parser.add_argument("--buddy-translate", default=os.environ.get("BUDDY_TRANSLATE", "buddy-translate"))
    parser.add_argument("--host", action="store_true", help="omit target-specific NR lowering")
    parser.add_argument("--allow-unlowered", action="store_true",
                        help="write a pass-through artifact if Buddy tools are unavailable")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # GraphDriver writes the numerical subgraph and a main graph that calls
    # it.  Lower the former first: that is where all Triton external calls
    # live.  The main graph can be linked once the subgraph object exists.
    source = args.replacement / "subgraph0.triton.mlir"
    if not source.is_file():
        source = args.replacement / "forward.triton.mlir"
    if not source.is_file():
        source = args.replacement / "forward.mlir"
    if not source.is_file():
        raise SystemExit("replacement graph is missing subgraph0.triton.mlir")
    report_path = args.replacement / "triton-call-replacement.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    lowered = args.output / "forward.lowered.mlir"
    llvm = args.output / "forward.ll"
    main_source = args.replacement / "forward.triton.mlir"
    if not main_source.is_file():
        raise SystemExit("replacement graph is missing forward.triton.mlir")
    main_lowered = args.output / "main.lowered.mlir"
    main_llvm = args.output / "main.ll"
    tosa_passes = "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))"
    passes = ["--canonicalize", "--cse",
              "--empty-tensor-to-alloc-tensor",
              "--one-shot-bufferize=bufferize-function-boundaries allow-return-allocs-from-loops=true buffer-alignment=64",
              "--convert-linalg-to-loops", "--expand-strided-metadata", "--lower-affine",
              "--convert-scf-to-cf", "--convert-cf-to-llvm", "--convert-arith-to-llvm", "--convert-index-to-llvm",
              "--memref-expand", "--finalize-memref-to-llvm", "--convert-func-to-llvm",
              "--reconcile-unrealized-casts"]
    if not args.host:
        # All dense ViT operations have become external Triton calls before
        # this stage.  The remaining graph plumbing uses ordinary memref/SCF
        # operations, so there is no linalg matmul for the BOSCAME AME pass
        # to lower.  That pass is applied to the separate Triton kernel IR.
        pass
    try:
        tosa_lowered = args.output / "forward.tosa-lowered.mlir"
        run([args.buddy_opt, source, "--pass-pipeline=" + tosa_passes, "-o", tosa_lowered],
            args.output / "tosa-lowering.log")
        run([args.buddy_opt, tosa_lowered, *passes, "-o", lowered], args.output / "buddy-opt.log")
        run([args.buddy_translate, "--buddy-to-llvmir", lowered, "-o", llvm],
            args.output / "buddy-translate.log")
        # The Buddy main graph takes one flat parameter buffer and the image.
        # Its private @vit declaration must keep the ordinary lowered ABI of
        # the separately compiled subgraph. Request a C wrapper only for
        # @forward, the entry called by the NR main.
        main_text = main_source.read_text()
        old = "-> memref<1x1000xf32> {"
        if main_text.count(old) != 1:
            raise RuntimeError("expected one fixed-shape @forward entry")
        main_input = args.output / "main.ciface.mlir"
        main_input.write_text(main_text.replace(
            old, "-> memref<1x1000xf32> attributes {llvm.emit_c_interface} {"))
        main_tosa = args.output / "main.tosa-lowered.mlir"
        run([args.buddy_opt, main_input, "--pass-pipeline=" + tosa_passes,
             "-o", main_tosa], args.output / "main-tosa-lowering.log")
        run([args.buddy_opt, main_tosa, *passes, "-o", main_lowered],
            args.output / "main-buddy-opt.log")
        run([args.buddy_translate, "--buddy-to-llvmir", main_lowered,
             "-o", main_llvm], args.output / "main-buddy-translate.log")
        if "define void @_mlir_ciface_forward(" not in main_llvm.read_text():
            raise RuntimeError("main graph did not export _mlir_ciface_forward")
        status = "PASS"
    except (OSError, RuntimeError) as error:
        if not args.allow_unlowered:
            raise
        shutil.copyfile(source, lowered)
        (args.output / "lowering-error.txt").write_text(str(error) + "\n")
        status = "UNLOWERED"
    result = {"stage": "Buddy graph lowering", "status": status,
              "source": str(source), "lowered": str(lowered),
              "llvm": str(llvm) if llvm.is_file() else None,
              "main_source": str(main_source) if main_source.is_file() else None,
              "main_llvm": str(main_llvm) if main_llvm.is_file() else None,
              "external_calls": report.get("graphs", {}).get("forward", {}).get("external_calls", 0),
              "target": "host" if args.host else "nr-fpga"}
    (args.output / "lower.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
