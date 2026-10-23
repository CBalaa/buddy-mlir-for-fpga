# RUN: %PYTHON %s buddy-opt %S/buffer-placement-ame-metadata.mlir
"""Validate logical value descriptors only for explicitly complete graphs."""

import subprocess
import sys
from pathlib import Path

OPTIONS = (
    "--linalg-to-boscame=buffer-placement=true enable-spm-dma=true "
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


def check(binary, label, source, diagnostic=None):
    result = subprocess.run(
        [binary, "-", OPTIONS, "-o", "/dev/null"],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if diagnostic is None:
        assert result.returncode == 0, (label, result.stderr)
    else:
        assert result.returncode != 0, (label, "unexpected success")
        assert diagnostic in result.stderr, (label, diagnostic, result.stderr)


def main():
    binary = sys.argv[1]
    source = "\n".join(
        line
        for line in Path(sys.argv[2]).read_text().splitlines()
        if not line.lstrip().startswith("//")
    )

    # Optional reverse lists and inputs without descriptors remain supported.
    check(binary, "valid descriptors", source)
    omitted_users = source.replace(
        ",\n        bosc_ame.logical_value_users = array<i64: 1>", ""
    ).replace("      bosc_ame.logical_value_users = [array<i64>],\n", "")
    assert "bosc_ame.logical_value_users" not in omitted_users
    check(binary, "omitted reverse lists", omitted_users)
    check(
        binary,
        "input without a producer or descriptor",
        replace_once(
            source,
            "bosc_ame.compute_input_values = array<i64: 10, 11, 12>",
            "bosc_ame.compute_input_values = array<i64: 10, 11, 12, 999>",
        ),
    )
    check(
        binary,
        "reverse lists are sets",
        argument(source, users="array<i64: 1, 1>"),
    )
    check(
        binary,
        "incomplete functions skip logical validation",
        descriptor(
            argument(
                replace_once(
                    source,
                    "bosc_ame.compute_graph_complete = true",
                    "bosc_ame.compute_graph_complete = false",
                ),
                users="array<i64: 999>",
            ),
            values='"invalid"',
        ),
    )

    for label, users, diagnostic in [
        (
            "unresolved argument user",
            "array<i64: 999>",
            "unresolved BOSCAME logical value user 999 for value 10",
        ),
        (
            "argument user does not read value",
            "array<i64: 0>",
            "BOSCAME logical value user 0 does not read value 10",
        ),
        (
            "argument reverse list omits reader",
            "array<i64>",
            "BOSCAME logical value users mismatch for value 10",
        ),
        (
            "invalid argument users type",
            "[array<i64: 1>]",
            "invalid BOSCAME function argument logical value users",
        ),
    ]:
        check(binary, label, argument(source, users=users), diagnostic)

    for value in ['"invalid"', "-1 : i64"]:
        check(
            binary,
            f"invalid argument value ID {value}",
            argument(source, value=value),
            "invalid BOSCAME function argument logical value ID",
        )

    for values, users in [
        ("array<i64: 13>", '"invalid"'),
        ("array<i64: 13>", "array<i64>"),
        ("array<i64: 13, 15>", "[array<i64>]"),
    ]:
        check(
            binary,
            f"invalid operation users shape {values} {users}",
            descriptor(source, values=values, users=users),
            "BOSCAME logical_value_users must match logical_value_ids",
        )
    for users in ["[0 : i64]", "[array<i32: 0>]"]:
        check(
            binary,
            f"invalid operation users entry {users}",
            descriptor(source, users=users),
            "BOSCAME logical_value_users entry must be an i64 array",
        )
    for values in ['"invalid"', "array<i64: -1>"]:
        check(
            binary,
            f"invalid operation value IDs {values}",
            descriptor(source, values=values),
            "invalid BOSCAME logical value IDs",
        )
    for field, before, after in [
        ("input", "array<i64: 10, 11, 12>", "array<i64: 10, 11, -1>"),
        ("output", "array<i64: 13>", "array<i64: -1>"),
    ]:
        attribute = f"bosc_ame.compute_{field}_values"
        check(
            binary,
            f"negative task {field} value",
            replace_once(
                source, f"{attribute} = {before}", f"{attribute} = {after}"
            ),
            f"negative BOSCAME logical value ID in {attribute}",
        )
    check(
        binary,
        "unresolved operation descriptor user",
        descriptor(source, users="[array<i64: 999>]"),
        "unresolved BOSCAME logical value user 999 for value 13",
    )

    # Task IDs are module-wide, but reverse references resolve in one function.
    other_function = """
  func.func @other_graph(%out: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 1 : i64} {
    %zero = arith.constant 0.0 : f32
    linalg.fill {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "fill",
      bosc_ame.compute_output_values = array<i64: 20>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true}
      ins(%zero : f32) outs(%out : memref<16xf32>)
    return
  }
"""
    check(
        binary,
        "cross-function logical user",
        replace_once(
            argument(source, users="array<i64: 2>"),
            "\n  }\n}",
            "\n  }\n" + other_function + "}",
        ),
        "unresolved BOSCAME logical value user 2 for value 10",
    )
    print("PASS complete graph logical value descriptors and reverse users")


if __name__ == "__main__":
    main()
