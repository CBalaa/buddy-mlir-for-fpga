#!/usr/bin/env python3
"""Check current compiler artifacts and archive portable checkpoint evidence."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
VIT = ROOT / "examples/FPGA-BOSCAME/vit-base-patch16-224"
CASES = ("gelu_197x3072", "softmax_12x197x197")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--object-audit", type=Path)
    args = parser.parse_args()
    report = json.loads((VIT / "build/build-report.json").read_text())
    assert report["status"] == "PASS" and len(report["cases"]) == 40
    records = []
    for case in report["cases"]:
        assert case["status"] == "PASS" and case["stage"] == "asm"
        directory = VIT / "build/kernels" / case["name"]
        for name, expected in case["artifacts"].items():
            assert digest(directory / name) == expected, (case["name"], name)
        records.append(
            {key: value for key, value in case.items() if key != "artifact_dir"}
        )
    write("build-report.json", {"status": "PASS", "cases": records})

    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        assert len(baseline) == 38
        checked = 0
        for case in baseline:
            assert case["name"] not in CASES and case["status"] == "PASS"
            for name, expected in case["artifacts"].items():
                assert (
                    digest(VIT / "build/kernels" / case["name"] / name)
                    == expected
                )
                checked += 1
        write("existing-artifact-baseline.json", baseline)
        write(
            "existing-artifact-comparison.json",
            {
                "status": "PASS",
                "existing_cases": len(baseline),
                "compared_artifacts": checked,
                "differences": [],
                "scope": "all pre-existing kernel.* artifacts, including emitted IR and objects; no new numerical run is inferred",
            },
        )

    if args.object_audit:
        audit = json.loads(args.object_audit.read_text())
        assert audit["status"] == "PASS"
        for name in CASES:
            assert audit["cases"][name]["object_sha256"] == digest(
                VIT / "build/kernels" / name / "kernel.o"
            )
        write("production-object-audit.json", audit)

    inputs = [
        "midend/include/Analysis/MathAccuracy.h",
        "midend/include/Conversion/LowerLinalgToBOSCAME/BOSCAMEMathApprox.h",
        "midend/include/Conversion/LowerLinalgToBOSCAME/BOSCAMEMathApproxCoefficients.h",
        "midend/include/Conversion/LowerLinalgToBOSCAME/BOSCAMEPointwise.h",
        "midend/lib/Analysis/KernelContract.cpp",
        *[
            f"midend/lib/Conversion/LowerLinalgToBOSCAME/BOSCAMEPlan{suffix}.cpp"
            for suffix in ("", "Verify", "Emit")
        ],
        "examples/FPGA-BOSCAME/tools/ame_to_word.py",
        "examples/FPGA-BOSCAME/tools/boscame_buffer_sim.py",
        "examples/FPGA-BOSCAME/vit-base-patch16-224/cases.py",
        *[
            f"examples/FPGA-BOSCAME/vit-base-patch16-224/tools/{name}.py"
            for name in (
                "build",
                "oracles",
                "simcheck",
                "nonlinear_simcheck",
                "check_nonlinear_numerics",
            )
        ],
        "docs/validation/nonlinear-math/generate.py",
        "docs/validation/nonlinear-math/error-certificate.json",
        "docs/validation/nonlinear-math/numerical-evidence.json",
        "docs/validation/nonlinear-operators/capture.py",
        *[
            f"tests/Conversion/LowerLinalgToBOSCAME/{name}.py"
            for name in (
                "PlanMathApprox",
                "math-approx-numerics",
                "maxnum-reduction",
                "buffer-ieee-arithmetic",
                "ame-vector-mask-classification",
            )
        ],
        *[
            f"tests/Conversion/TritonToLinalgExperimental/{name}.py"
            for name in ("vit-nonlinear-oracle", "vit-nonlinear-collection")
        ],
    ]
    snapshots = {}
    for name in CASES:
        directory = VIT / "build/kernels" / name
        snapshots[name] = {
            p.name: digest(p) for p in directory.glob("kernel.*")
        }
        snapshots[name]["kernel_source"] = digest(VIT / name / "kernel.py")
        saved = json.loads((directory / "simcheck-result.json").read_text())
        assert saved["status"] == "PASS" and saved["module_sha256"] == digest(
            directory / "kernel.boscame.mlir"
        )
        assert saved["output_archive_sha256"] == digest(
            directory / "nonlinear-outputs.npz"
        )
    write(
        "artifact-hashes.json",
        {
            "repository_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": {name: digest(ROOT / name) for name in inputs},
            "cases": snapshots,
        },
    )
    print("PASS: 40 builds, current source/artifact/output-archive hashes")


if __name__ == "__main__":
    main()
