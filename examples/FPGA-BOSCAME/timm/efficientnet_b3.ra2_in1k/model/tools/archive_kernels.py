#!/usr/bin/env python3
# ===- archive_kernels.py - Triton NR objects to static library -*- python -*-===//
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
# Pack already-audited RISC-V kernel.o + adapter.o into libeffnet_triton.a.
# Requires this-repo llvm-ar / llvm-nm. Rejects objects that pull in
# malloc/printf/main so the bare-metal image stays self-contained aside
# from the NR runtime.
#
# ===----------------------------------------------------------------------===//
"""Snapshot already-built EfficientNet Triton NR objects into a static archive."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(tool, *args):
    return subprocess.check_output([*shlex.split(tool), *map(str, args)], text=True)


def symbols(nm, path, defined):
    flags = "--defined-only" if defined else "--undefined-only"
    output = run(nm, "--format=posix", "--extern-only", flags, path)
    names = {line.split()[0] for line in output.splitlines() if len(line.split()) >= 2}
    return names, output


def archive(root, cases, output, ar, nm):
    if output.exists():
        raise ValueError("output exists; use a fresh directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-", dir=output.parent) as directory:
        staging = Path(directory)
        objects, records, definitions = [], [], set()
        for name in sorted(set(cases)):
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError("invalid case name: " + name)
            source = root / name
            nr = source / "nr"
            manifest = json.loads((nr / "manifest.json").read_text())
            front = json.loads((source / "frontend.json").read_text())
            if manifest["case"]["name"] != name or front["name"] != name:
                raise ValueError("manifest case mismatch: " + name)
            if manifest.get("elf_audit", {}).get("status") != "PASS":
                raise ValueError("case has no successful ELF audit: " + name)
            expected = {front["symbol"], "_mlir_ciface_kernel_" + name}
            case_defs = set()
            for basename in ("kernel.o", "adapter.o"):
                path = nr / basename
                data = path.read_bytes()
                if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 243:
                    raise ValueError("not a RISC-V ELF64 object: " + str(path))
                copied = staging / (name + "." + basename)
                copied.write_bytes(data)
                defs, _ = symbols(nm, copied, True)
                undefs, _ = symbols(nm, copied, False)
                if not defs <= expected or defs & definitions:
                    raise ValueError("unexpected/duplicate symbols: " + str(defs))
                if undefs & {"malloc", "calloc", "realloc", "free", "printf", "main", "launch"}:
                    raise ValueError("unexpected heap/test dependency: " + str(undefs))
                definitions.update(defs)
                case_defs.update(defs)
                objects.append(copied)
            if case_defs != expected:
                raise ValueError("missing kernel/adapter entry: " + name)
            records.append({"case": name, "symbols": sorted(case_defs)})
        lib = staging / "libeffnet_triton.a"
        run(ar, "rcsD", lib, *objects)
        report = {
            "format": 1,
            "archive": "libeffnet_triton.a",
            "sha256": sha(lib.read_bytes()),
            "members": run(ar, "t", lib).splitlines(),
            "case_count": len(records),
            "cases": records,
        }
        (staging / "archive.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-build", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ar", default="llvm-ar")
    parser.add_argument("--nm", default="llvm-nm")
    args = parser.parse_args()
    result = archive(args.triton_build, args.case, args.output, args.ar, args.nm)
    print(f"{result['case_count']} kernels; {args.output / 'libeffnet_triton.a'}")
