#!/usr/bin/env python3
# ===- archive_model_run.py - archive EfficientNet FPGA run -------*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Validate a completed FPGA run (DDR readback + UART acceptance) and copy
# the evidence into validation/. Classification-focused; no tokenizer.
#
# ===----------------------------------------------------------------------===//
"""Archive a completed EfficientNet FPGA model run with numeric UART checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

from check_board_trace import check
from prepare_model_run import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def artifact(path):
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def safe_file(directory, name):
    require(isinstance(name, str) and Path(name).name == name and name not in (".", ".."),
            "segment filename must be a single safe component")
    path = directory / name
    require(not path.is_symlink() and path.is_file(),
            "missing/symlink segment: " + str(path))
    return path


def verify_run_identity(run, prepared, uart):
    result, manifest = read(run / "result.json"), read(run / "run-manifest.json")
    deployment = read(prepared / "deployment.json")
    plan_path = prepared / "ddr-load.plan"
    plan = tomllib.loads(plan_path.read_text())
    require(result.get("status") == "OK" and result.get("ddr_readback_matches") is True
            and result.get("completion_marker_seen") is True,
            "run incomplete or DDR readback failed")
    require(result.get("uart_bytes") == len(uart),
            "UART byte count differs from completed worker result")
    require(manifest.get("plan_sha256") == digest(plan_path),
            "uploaded plan hash mismatch")
    require(plan.get("version") == 1 and plan.get("ddr_base") == 0x80000000,
            "unexpected DDR plan header")
    segments = plan.get("segments", [])
    require(segments and segments == [
        {k: v for k, v in s.items() if k != "arena_end"}
        for s in deployment.get("segments", [])
    ], "deployment segments differ from actual plan")
    names = [s["file"] for s in segments]
    require(len(set(names)) == len(names), "duplicate DDR segment filename")
    require(manifest.get("segments") == [
        {k: s[k] for k in ("file", "size", "sha256")} for s in segments
    ], "uploaded segments differ from plan")
    require(result.get("segment_readbacks") == {n: True for n in names},
            "worker did not verify every exact segment readback")
    for s in segments:
        path = safe_file(prepared, s["file"])
        require(path.stat().st_size == s["size"] and digest(path) == s["sha256"],
                "prepared segment bytes differ from uploaded manifest: " + s["file"])
    return {
        "run_id": manifest.get("run_id") or result.get("run_id"),
        "elf_sha256": deployment.get("elf_sha256"),
        "segments": names,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="fpga_run.sh output directory (result.json + uart)")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="host-reference.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh archive directory under model/validation/")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("output exists; use a fresh directory: " + str(args.output))

    uart_path = args.run / "uart.raw.log"
    if not uart_path.is_file():
        # Some runners store UART as uart.log
        uart_path = args.run / "uart.log"
    require(uart_path.is_file(), "missing uart.raw.log / uart.log in run dir")
    uart = uart_path.read_bytes()
    identity = verify_run_identity(args.run, args.prepared, uart)
    reference = read(args.reference)
    trace = check(uart.decode("utf-8", errors="replace"), reference)

    args.output.mkdir(parents=True)
    for name in ("result.json", "run-manifest.json"):
        shutil.copy2(args.run / name, args.output / name)
    shutil.copy2(uart_path, args.output / "uart.raw.log")
    shutil.copy2(args.prepared / "deployment.json", args.output / "deployment.json")
    shutil.copy2(args.prepared / "ddr-load.plan", args.output / "ddr-load.plan")
    shutil.copy2(args.reference, args.output / "host-reference.json")
    (args.output / "board-trace.json").write_text(json.dumps(trace, indent=2) + "\n")

    verification = {
        "model": "efficientnet_b3.ra2_in1k",
        "identity": identity,
        "board_trace": trace,
        "prepared": artifact(args.prepared / "image.bin"),
        "weights": artifact(args.prepared / "weights.bin"),
        "status": "PASS" if trace["status"] == "PASS" else "FAIL",
        "markers": {
            "MODEL_RUN_NUMERIC_PASS": trace["status"] == "PASS",
            "ARGMAX_PASS": bool(trace.get("argmax_match")),
            "RUNTIME_PASS": bool(trace.get("runtime_pass")),
        },
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n")
    print(json.dumps({
        "status": verification["status"],
        "output": str(args.output),
        "run_id": identity.get("run_id"),
        "argmax_match": trace.get("argmax_match"),
    }, indent=2))
    return 0 if verification["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
