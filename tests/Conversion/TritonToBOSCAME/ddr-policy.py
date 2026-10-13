# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Catch missing DDR policies after DMA and conflicting runtime declarations."""

import os
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, os.fspath(TOOLS))
from boscame_check import buddy_opt_path, run_pass  # noqa: E402
from boscame_sim import SimError, simulate  # noqa: E402


def run(module):
    return simulate(
        module,
        "padded_k",
        {
            "%arg0": bytearray(64),
            "%arg1": bytearray(128),
            "%arg2": bytearray(32),
            "%arg4": bytearray([0xA5] * 256),
        },
        {"%arg3": 31},
    )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).parent / "Inputs/padded-k-initializer.ttir"
    output = work / "correct.mlir"
    code, stderr = run_pass(buddy_opt_path(), fixture, output)
    assert code == 0, stderr
    run(output)
    text = output.read_text()
    for policy in ("read", "write"):
        missing = work / f"missing-{policy}.mlir"
        changed, count = re.subn(
            rf"^\s*(?:func\.)?call @bosc_ame_ddr_{policy}_begin\(\).*\n",
            "",
            text,
            flags=re.M,
        )
        assert count > 0
        missing.write_text(changed)
        try:
            run(missing)
        except SimError as error:
            assert f"CPU DDR {policy} requires the {policy} policy" in str(
                error
            )
        else:
            raise AssertionError(f"missing {policy} policy was not detected")
    # CHECK: PASS: missing DDR read/write transitions are rejected
    print("PASS: missing DDR read/write transitions are rejected")

    for signature, valid in (("()", True), ("(i32)", False)):
        source = work / f"declaration-{valid}.ttir"
        source.write_text(
            fixture.read_text().replace(
                "module {",
                "module {\n  func.func private "
                f"@bosc_ame_ddr_read_begin{signature}",
                1,
            )
        )
        code, stderr = run_pass(buddy_opt_path(), source, output)
        if valid:
            assert code == 0, stderr
            assert (
                output.read_text().count(
                    "func.func private @bosc_ame_ddr_read_begin()"
                )
                == 1
            )
            run(output)
        else:
            assert code != 0
            assert "reserved BOSCAME DDR policy helper" in stderr, stderr
    # CHECK: PASS: DDR helper declarations are reused or diagnosed
    print("PASS: DDR helper declarations are reused or diagnosed")


if __name__ == "__main__":
    main()
