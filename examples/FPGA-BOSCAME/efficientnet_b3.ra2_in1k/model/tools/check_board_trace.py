#!/usr/bin/env python3
# ===- check_board_trace.py - UART acceptance for EfficientNet ----*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
"""Validate an EfficientNet board UART log against host-reference.json.

The predicted class must match exactly. The top logit passes when its
float32 value is within a relative tolerance of the host reference.
On-board ``verify`` / ``RA returned`` strings are recorded, but a FAIL
from the older bit-exact image does not override a class-correct logit.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

# Full-model top logit. Single operators use 2e-4; 200 layers accumulate
# a little past that. 1e-3 still rejects a wrong class-scale result.
TOP_LOGIT_RTOL = 1e-3


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def logit_close(got_bits: int, ref_bits: int, rtol: float = TOP_LOGIT_RTOL) -> bool:
    got = f32_from_bits(got_bits)
    ref = f32_from_bits(ref_bits)
    if got == ref:
        return True
    if got != got or ref != ref:
        return False
    return abs(got - ref) <= rtol * abs(ref)


def logit_rel_error(got_bits: int, ref_bits: int):
    got = f32_from_bits(got_bits)
    ref = f32_from_bits(ref_bits)
    if got != got or ref != ref or ref == 0.0:
        return None
    return abs(got - ref) / abs(ref)


def runtime_passed(log: str) -> bool:
    return (len(re.findall(r"^\[nr\] RA returned: PASS\r?$", log, re.M)) == 1
            and not re.search(
                r"\[nr\][^\r\n]*(?:FAIL|TRAP)|TRAP mcause=|"
                r"verify[^\r\n]*:\s*FAIL", log))


def parse_model_line(log: str):
    match = re.search(
        r"\[model\] argmax=([0-9A-Fa-f]+) logit_bits=([0-9A-Fa-f]+)", log)
    if not match:
        return None
    return {
        "argmax": int(match.group(1), 16),
        "logit_bits": int(match.group(2), 16),
    }


def check(log: str, reference: dict) -> dict:
    model = parse_model_line(log)
    verify_pass = bool(re.search(
        r"^verify efficientnet_b3\.ra2_in1k: PASS\r?$", log, re.M))
    expected_class = reference.get("predicted_class")
    expected_bits = reference.get("logits_max_bits")
    argmax_match = (model is not None and expected_class is not None
                    and model["argmax"] == int(expected_class))
    bits_match = (model is not None and expected_bits is not None
                  and logit_close(model["logit_bits"], int(expected_bits)))
    rel_error = (logit_rel_error(model["logit_bits"], int(expected_bits))
                 if model is not None and expected_bits is not None else None)
    no_trap = re.search(r"TRAP mcause=", log) is None
    complete = (model is not None and argmax_match and bits_match and no_trap)
    return {
        "scope": "argmax exact + top logit rtol 1e-3; no TRAP",
        "trace_complete": complete,
        "verify_pass": verify_pass,
        "runtime_pass": runtime_passed(log),
        "board": model,
        "reference_class": expected_class,
        "reference_logit_bits": expected_bits,
        "argmax_match": argmax_match,
        "logit_bits_match": bits_match,
        "logit_rel_error": rel_error,
        "top_logit_rtol": TOP_LOGIT_RTOL,
        "status": "PASS" if complete else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    log = args.uart.read_text(errors="replace")
    reference = json.loads(args.reference.read_text())
    report = check(log, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
