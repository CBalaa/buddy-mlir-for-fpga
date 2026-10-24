# RUN: %PYTHON %s %t buddy-opt
"""Preserve SPM fill versions, aliases and effect barriers through replanning."""

import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402

TYPE = "memref<32xf32, #bosc_ame.spm<0>>"


def attributes(number):
    return (
        f"bosc_ame.compute_id = {number} : i64, "
        'bosc_ame.compute_kind = "fill", '
        "bosc_ame.compute_stage = 0 : i64, "
        "bosc_ame.compute_input_values = array<i64>, "
        f"bosc_ame.compute_output_values = array<i64: {number}>, "
        "bosc_ame.compute_predecessors = array<i64>, "
        "bosc_ame.compute_barrier_after = false, "
        "bosc_ame.rematerializable = true"
    )


def generic(left, right, output, operation):
    return f"""
    linalg.generic {{indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}}
      ins({left}, {right} : {TYPE}, {TYPE}) outs({output} : {TYPE}) {{
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %value = arith.{operation} %lhs, %rhs : f32
      linalg.yield %value : f32
    }}
"""


def source(case):
    declarations = ""
    between = ""
    after = ""
    reader = "%mean"
    if case == "alias":
        declarations = f"%alias = memref.cast %mean : {TYPE} to {TYPE}"
        reader = "%alias"
    elif case == "overlap":
        declarations = (
            "%overlap = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}"
            f" : {TYPE}"
        )
    elif case == "unknown_effect":
        between = "func.call @unknown() : () -> ()"
    elif case == "late_unknown_effect":
        after = "func.call @unknown() : () -> ()"
    elif case == "nested_late_unknown_effect":
        after = "scf.if %choose { func.call @unknown() : () -> () }"
    elif case == "read_before_fill":
        declarations = generic("%x", "%mean", "%y", "subf")
    elif case == "output_write":
        between = generic("%x", "%x", "%mean", "addf")
    elif case == "dma_reader":
        between = f"memref.copy %mean, %out : {TYPE} to memref<32xf32>"
    return f"""
#id = affine_map<(i) -> (i)>
module {{
  func.func private @unknown()
  func.func @broadcast(%input: memref<32xf32>, %out: memref<32xf32>, %choose: i1) {{
    %two = arith.constant 2.0 : f32
    %three = arith.constant 3.0 : f32
    %mean = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : {TYPE}
    %x = bosc_ame.spm.view {{bank = 0 : i32, offset = 128 : i64}} : {TYPE}
    %y = bosc_ame.spm.view {{bank = 0 : i32, offset = 256 : i64}} : {TYPE}
    memref.copy %input, %x : memref<32xf32> to {TYPE}
    {declarations}
    linalg.fill {{{attributes(11)}}} ins(%two : f32) outs(%mean : {TYPE})
    {between}
    {generic("%x", reader, "%y", "subf")}
    linalg.fill {{{attributes(12)}}} ins(%three : f32) outs(%mean : {TYPE})
    {generic("%y", reader, "%y", "mulf")}
    {after}
    memref.copy %y, %out : {TYPE} to memref<32xf32>
    return
  }}
}}
"""


def main():
    directory, binary = Path(sys.argv[1]), sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    for case in (
        "repeated_fill",
        "alias",
        "overlap",
        "unknown_effect",
        "late_unknown_effect",
        "nested_late_unknown_effect",
        "read_before_fill",
        "output_write",
        "dma_reader",
    ):
        result = subprocess.run(
            [
                binary,
                "-",
                "--lower-linalg-to-boscame=buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias",
            ],
            input=source(case),
            text=True,
            capture_output=True,
            check=False,
        )
        rejected = {
            "overlap": "overlapping SPM input windows require a shared root",
            "unknown_effect": "call may clobber live SPM workspace",
            "late_unknown_effect": "call may clobber live SPM workspace",
            "nested_late_unknown_effect": "if requires explicit then and else scopes",
            "read_before_fill": "no initialized content version",
        }
        if case in rejected:
            assert result.returncode != 0, case
            assert rejected[case] in result.stderr, result.stderr
            if case == "nested_late_unknown_effect":
                with_else = source(case).replace(
                    "scf.if %choose { func.call @unknown() : () -> () }",
                    "scf.if %choose { func.call @unknown() : () -> () } else {}",
                )
                nested = subprocess.run(
                    [
                        binary,
                        "-",
                        "--lower-linalg-to-boscame=dma-abi=args_aligned64_nonalias",
                    ],
                    input=with_else,
                    text=True,
                    capture_output=True,
                )
                assert nested.returncode != 0
                assert "call may clobber live SPM workspace" in nested.stderr, (
                    nested.stderr
                )
            continue
        assert result.returncode == 0, (case, result.stderr)
        text = result.stdout
        # Legacy graph IDs are derived information. The shared analysis rebuilds
        # tasks and keeps the two distinct fill versions in their original order.
        assert "bosc_ame.compute_id" not in text, text
        assert 'bosc_ame.task_map = [{action = "retained"' in text, text
        assert "offset = 0 : i64" in text, text
        root = re.search(
            r"(%[\w.]+) = bosc_ame\.spm\.view[^\n]*offset = 0 : i64", text
        )[1]
        assert re.search(rf"vector\.store[^\n]*, {re.escape(root)}\[", text), (
            text
        )
        path = directory / f"broadcast-{case}.mlir"
        path.write_text(text)
        values = [i * 0.25 - 4 for i in range(32)]
        original = struct.pack("<32f", *values)
        guard = b"\xb7" * 64
        inputs = bytearray(original + guard)
        output = bytearray(128 + 64)
        output[128:] = guard
        machine = Machine(path, "broadcast")
        machine.spm[0][:] = b"\x69" * len(machine.spm[0])
        machine.launch({"%arg0": inputs, "%arg1": output})
        expected = tuple(
            -3 * value if case == "output_write" else (value - 2) * 3
            for value in values
        )
        assert struct.unpack("<32f", output[:128]) == expected
        assert inputs == original + guard
        assert output[128:] == guard
    print(
        "PASS broadcast versions: exact arithmetic and guards; alias, overlap, calls and early reads checked"
    )


if __name__ == "__main__":
    main()
