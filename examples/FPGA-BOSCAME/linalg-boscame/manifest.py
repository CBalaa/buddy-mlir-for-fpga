#!/usr/bin/env python3
"""Bind a board image to its exact compiler, source files, and LLVM input."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCE_DIRS = (
    "midend/include/Conversion/BOSCAMEHelpers",
    "midend/lib/Conversion/BOSCAMEHelpers",
    "midend/include/Conversion/LowerLinalgToBOSCAME",
    "midend/lib/Conversion/LowerLinalgToBOSCAME",
    "midend/include/Conversion/TritonToLinalgExperimental",
    "midend/lib/Conversion/TritonToLinalgExperimental",
    "midend/include/Conversion/StructuredToMemref",
    "midend/lib/Conversion/StructuredToMemref",
    "midend/include/Dialect/BOSCAME",
    "midend/lib/Dialect/BOSCAME",
    "examples/FPGA-BOSCAME/linalg-boscame",
    "examples/FPGA-BOSCAME/tools",
    "examples/FPGA-BOSCAME/layernorm_197x768",
    "runtime/boscame",
)
SOURCE_SUFFIXES = {
    ".c",
    ".cpp",
    ".h",
    ".td",
    ".py",
    ".mlir",
    ".ld",
    ".S",
    ".mk",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tool_path(path):
    if path.is_file():
        return path
    found = shutil.which(str(path))
    if not found:
        raise FileNotFoundError("missing build tool: " + str(path))
    return Path(found)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inputs(args):
    sources = {}
    for directory in SOURCE_DIRS:
        for path in sorted((REPO / directory).rglob("*")):
            if path.is_file() and (
                path.suffix in SOURCE_SUFFIXES
                or path.name in ("CMakeLists.txt", "makefile")
            ):
                sources[path.relative_to(REPO).as_posix()] = sha256(path)
    return {
        "kernel": args.kernel,
        "variant": args.variant,
        "compiler_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "compiler_sha256": sha256(tool_path(args.buddy_opt)),
        "compiler_translate_sha256": sha256(tool_path(args.buddy_translate)),
        "llc_sha256": sha256(tool_path(args.llc)),
        "source_sha256": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "finish"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--buddy-opt", type=Path, required=True)
    parser.add_argument("--buddy-translate", type=Path, required=True)
    parser.add_argument("--llc", type=Path, required=True)
    parser.add_argument(
        "--kernel", choices=("gemm", "layernorm"), required=True
    )
    parser.add_argument(
        "--variant", choices=("buffer", "frontend"), required=True
    )
    parser.add_argument("--llvm-ir", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / "build-inputs.json"
    manifest = args.out / "build-manifest.json"
    current = inputs(args)
    if args.stage == "capture":
        # A failed rebuild must not leave an older image looking current.
        manifest.unlink(missing_ok=True)
        write_json(snapshot, current)
        return
    if current != json.loads(snapshot.read_text()):
        raise SystemExit(
            "compiler or image sources changed during build; rebuild image"
        )
    if not args.llvm_ir:
        parser.error("finish requires --llvm-ir")
    saved_ir = args.out / "kernel.ll"
    if args.llvm_ir.resolve() != saved_ir.resolve():
        shutil.copyfile(args.llvm_ir, saved_ir)
    current.update(
        llvm_ir_sha256=sha256(saved_ir),
        elf_sha256=sha256(args.out / (args.kernel + ".elf")),
        bin_sha256=sha256(args.out / (args.kernel + ".bin")),
    )
    final_inputs = inputs(args)
    if any(current[name] != value for name, value in final_inputs.items()):
        raise SystemExit(
            "compiler or image sources changed during manifest generation"
        )
    write_json(manifest, current)


if __name__ == "__main__":
    main()
