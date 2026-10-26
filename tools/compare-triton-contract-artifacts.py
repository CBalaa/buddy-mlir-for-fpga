#!/usr/bin/env python3
"""Compare the unified-toolchain migration baseline without normalizing IR."""

import argparse
import hashlib
import json
from pathlib import Path

EXAMPLES = (
    "matmul_i8_196x768x768",
    "matmul_i8_128x1024x128",
    "layernorm_197x768",
)
STAGES = (
    "kernel.frontend.mlir",
    "kernel.planned.mlir",
    "kernel.boscame.mlir",
    "kernel.ll",
    "kernel.nr.s",
    "kernel.o",
)


def compare(baseline, candidate, stages):
    result = {}
    for stage in stages:
        before = (baseline / stage).read_bytes()
        after = (candidate / stage).read_bytes()
        result[stage] = {
            "baseline_sha256": hashlib.sha256(before).hexdigest(),
            "candidate_sha256": hashlib.sha256(after).hexdigest(),
            "identical": before == after,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-examples", type=Path, required=True)
    parser.add_argument("--candidate-examples", type=Path, required=True)
    parser.add_argument("--baseline-vit", type=Path, required=True)
    parser.add_argument("--candidate-vit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = {
        name: compare(
            args.baseline_examples / name / name,
            args.candidate_examples / name / name,
            (*STAGES, "kernel.llvm.mlir"),
        )
        for name in EXAMPLES
    }
    cases = {
        path.parent.name
        for path in args.baseline_vit.glob("*/kernel.frontend.mlir")
    }
    candidate_cases = {
        path.parent.name
        for path in args.candidate_vit.glob("*/kernel.frontend.mlir")
    }
    if len(cases) != 40 or cases != candidate_cases:
        parser.error("expected the same complete inventory of 40 ViT cases")
    vit = {
        name: compare(
            args.baseline_vit / name, args.candidate_vit / name, STAGES
        )
        for name in sorted(cases)
    }
    records = [
        record
        for group in (examples, vit)
        for case in group.values()
        for record in case.values()
    ]
    differences = sum(not record["identical"] for record in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": 1,
                "comparison": "raw bytes; no normalization",
                "total": len(records),
                "differences": differences,
                "examples": examples,
                "vit": vit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Compared {len(records)} artifacts; {differences} differences")
    return bool(differences)


if __name__ == "__main__":
    raise SystemExit(main())
