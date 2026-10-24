"""Check every retained launch-metadata case independently."""

import subprocess
import sys
from pathlib import Path

from buddy_mlir import ir
from buddy_mlir.dialects import func  # noqa: F401

source, work, tool = sys.argv[1:]
work = Path(work)
work.mkdir(parents=True, exist_ok=True)
invalid = {"missing_axis", "missing_axis_with_contract", "unrecognized_role"}
with ir.Context() as context:
    module = ir.Module.parse(Path(source).read_text())
    for operation in module.body.operations:
        name = ir.StringAttr(operation.attributes["sym_name"]).value
        path = work / (name + ".mlir")
        path.write_text("module {\n" + str(operation) + "\n}")
        results = [
            subprocess.run(
                [tool, str(path), "--" + entry], text=True, capture_output=True
            )
            for entry in ("linalg-to-boscame", "lower-linalg-to-boscame")
        ]
        if name in invalid:
            for result in results:
                assert result.returncode != 0, name
                assert (
                    "invalid legacy program-axis metadata" in result.stderr
                ), result.stderr
            continue
        for result in results:
            assert result.returncode == 0, (name, result.stderr)
        assert results[0].stdout == results[1].stdout
        output = results[0].stdout
        assert "bosc_ame.program_" not in output
        if name in {
            "known",
            "unknown",
            "partial_shuffled",
            "program_id_only",
            "declaration",
        }:
            assert "contract.program =" in output, (name, output)
        else:
            assert "contract.program =" not in output, (name, output)
        if name in {
            "known",
            "partial_shuffled",
            "contract_only",
            "declaration",
        }:
            assert "contract.grid =" in output, (name, output)
        if name == "partial_shuffled":
            assert 'axis = 2 : i32, role = "num_programs"' in output
            assert 'axis = 0 : i32, role = "num_programs"' in output
        if name == "known":
            assert "array<i32: 197, 1, 1>" in output
print(
    "PASS valid launch roles, shuffled axes, declarations and invalid metadata"
)
