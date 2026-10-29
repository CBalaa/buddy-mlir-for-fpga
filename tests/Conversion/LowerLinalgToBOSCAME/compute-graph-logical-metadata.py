# RUN: %PYTHON %s buddy-opt %S/buffer-placement-ame-metadata.mlir
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Rebuild legacy logical metadata; never let stale reverse users override IR."""

import re
import subprocess
import sys
from pathlib import Path

OPTIONS = (
    "--linalg-to-boscame=plan-version=1 buffer-placement=true enable-spm-dma=true "
    "enable-ame=true dma-abi=args_aligned64_nonalias "
)
ARGUMENT = (
    "bosc_ame.logical_value_id = 10 : i64,\n"
    "        bosc_ame.logical_value_users = array<i64: 1>"
)
DESCRIPTOR = (
    "bosc_ame.logical_value_ids = array<i64: 13>,\n"
    "      bosc_ame.logical_value_users = [array<i64>]"
)


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def argument(source, value="10 : i64", users="array<i64: 1>"):
    return replace_once(
        source,
        ARGUMENT,
        f"bosc_ame.logical_value_id = {value},\n"
        f"        bosc_ame.logical_value_users = {users}",
    )


def descriptor(source, values="array<i64: 13>", users="[array<i64>]"):
    return replace_once(
        source,
        DESCRIPTOR,
        f"bosc_ame.logical_value_ids = {values},\n"
        f"      bosc_ame.logical_value_users = {users}",
    )


def check(binary, label, source):
    result = subprocess.run(
        [binary, "-", OPTIONS, "-o", "-"],
        input=source,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (label, result.stderr)
    assert "bosc_ame.logical_value" not in result.stdout, (label, result.stdout)
    assert "bosc_ame.compute_id" not in result.stdout, (label, result.stdout)
    assert "contract.compute_graph = {complete = true" in result.stdout, label
    return result.stdout


def main():
    binary = sys.argv[1]
    source = "\n".join(
        line
        for line in Path(sys.argv[2]).read_text().splitlines()
        if not line.lstrip().startswith("//")
    )
    baseline = check(binary, "valid legacy descriptors", source)
    omitted = re.sub(
        r",\s*bosc_ame.logical_value_users = array<i64[^>]*>", "", source
    )
    omitted = re.sub(
        r"\s*bosc_ame.logical_value_users = \[array<i64[^>]*>\],", "", omitted
    )
    assert "bosc_ame.logical_value_users" not in omitted
    cases = [
        ("omitted reverse users", omitted),
        (
            "unknown input value",
            replace_once(
                source,
                "bosc_ame.compute_input_values = array<i64: 10, 11, 12>",
                "bosc_ame.compute_input_values = array<i64: 10, 11, 12, 999>",
            ),
        ),
        ("duplicate reverse users", argument(source, users="array<i64: 1, 1>")),
        (
            "incomplete graph claim",
            replace_once(
                source,
                "bosc_ame.compute_graph_complete = true",
                "bosc_ame.compute_graph_complete = false",
            ),
        ),
    ]
    for users in (
        "array<i64: 999>",
        "array<i64: 0>",
        "array<i64>",
        "[array<i64: 1>]",
    ):
        cases.append(
            (
                "stale/malformed argument users " + users,
                argument(source, users=users),
            )
        )
    for value in ('"invalid"', "-1 : i64"):
        cases.append(
            ("malformed legacy value " + value, argument(source, value=value))
        )
    for values, users in (
        ("array<i64: 13>", '"invalid"'),
        ("array<i64: 13>", "array<i64>"),
        ("array<i64: 13, 15>", "[array<i64>]"),
        ("array<i64: 13>", "[0 : i64]"),
        ("array<i64: 13>", "[array<i32: 0>]"),
        ('"invalid"', "[array<i64>]"),
        ("array<i64: -1>", "[array<i64>]"),
        ("array<i64: 13>", "[array<i64: 999>]"),
    ):
        cases.append(
            (
                "stale/malformed operation descriptor " + values + users,
                descriptor(source, values, users),
            )
        )
    for field, before, after in (
        ("input", "array<i64: 10, 11, 12>", "array<i64: 10, 11, -1>"),
        ("output", "array<i64: 13>", "array<i64: -1>"),
    ):
        key = f"bosc_ame.compute_{field}_values"
        cases.append(
            (
                "negative task " + field,
                replace_once(source, key + " = " + before, key + " = " + after),
            )
        )
    # Legacy task/value annotations are derived facts. Even a complete-graph
    # claim cannot make stale IDs override actual standard-IR reads and writes.
    # Neutral contract corruption is rejected by compute-graph-fusion.py.
    for label, variant in cases:
        assert check(binary, label, variant) == baseline, label

    other = """
  func.func @other_graph() attributes {
      bosc_ame.compute_graph_complete = true,
      bosc_ame.compute_graph_node_count = 1 : i64} {
    %out = memref.alloc() alignment = 64 : memref<16xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill {bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_output_values = array<i64: 20>}
      ins(%zero : f32) outs(%out : memref<16xf32>)
    return
  }
"""
    two = (
        argument(source, users="array<i64: 2>").rsplit("}", 1)[0] + other + "}"
    )
    rebuilt = check(binary, "cross-function legacy reverse users", two)
    assert rebuilt.count("contract.task =") == 3, rebuilt
    print(
        "PASS legacy logical descriptors and reverse users are rebuilt from actual IR"
    )


if __name__ == "__main__":
    main()
