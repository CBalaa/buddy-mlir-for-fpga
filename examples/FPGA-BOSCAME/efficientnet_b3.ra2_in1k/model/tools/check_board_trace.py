#!/usr/bin/env python3
# ===- check_board_trace.py - UART acceptance for EfficientNet ----*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
"""Validate an EfficientNet board UART log against host-reference.json."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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
                  and model["logit_bits"] == int(expected_bits))
    complete = (runtime_passed(log) and verify_pass and model is not None
                and argmax_match and bits_match)
    return {
        "scope": "argmax + top logit bits + NR PASS markers",
        "trace_complete": complete,
        "verify_pass": verify_pass,
        "runtime_pass": runtime_passed(log),
        "board": model,
        "reference_class": expected_class,
        "reference_logit_bits": expected_bits,
        "argmax_match": argmax_match,
        "logit_bits_match": bits_match,
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
