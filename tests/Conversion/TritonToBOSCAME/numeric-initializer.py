# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Nonzero other/initializer, K padding, masked stores and scratch reuse."""

import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, os.fspath(TOOLS))
from boscame_check import buddy_opt_path, run_pass  # noqa: E402
from boscame_sim import simulate  # noqa: E402


def signed(value):
    return value - 256 if value >= 128 else value


def wrap32(value):
    return ((value + (1 << 31)) % (1 << 32)) - (1 << 31)


def main():
    fixture = Path(__file__).parent / "Inputs/padded-k-initializer.ttir"
    original = fixture.read_text()
    for previous in (True, False):
        source = original
        if not previous:
            source = "\n".join(
                line
                for line in source.splitlines()
                if "%previous =" not in line and "%sum =" not in line
            ).replace("tt.store %cp2, %sum,", "tt.store %cp2, %dot,")
        input_path = Path(sys.argv[1] + f".{previous}.ttir")
        output_path = Path(sys.argv[1] + f".{previous}.mlir")
        input_path.write_text(source)
        code, stderr = run_pass(buddy_opt_path(), input_path, output_path)
        assert code == 0, stderr
        # This exceeds the 128-byte ACC tile: the 4x64 B packed B slab is
        # the larger scratch view. No original allocation includes K padding.
        scratch = bytearray([0xA5] * 256)
        a = bytearray((i * 7 + 9) % 256 for i in range(2 * 32))
        b = bytearray((i * 5 + 3) % 256 for i in range(4 * 32))
        c = bytearray(b"\xfe\xff\xff\x7f" * 8)
        for valid in (31, 0, 32, -1, 40):
            expected = bytearray(c)
            for j in range(2):
                total = 17
                if previous:
                    total += int.from_bytes(
                        c[j * 4 : j * 4 + 4], "little", signed=True
                    )
                for k in range(32):
                    lhs = signed(a[k]) if k < valid else 3
                    rhs = signed(b[j * 32 + k]) if k < valid else -2
                    total += lhs * rhs
                expected[j * 4 : j * 4 + 4] = wrap32(total).to_bytes(
                    4, "little", signed=True
                )
            machine, _ = simulate(
                output_path,
                "padded_k",
                {"%arg0": a, "%arg1": b, "%arg2": c, "%arg4": scratch},
                {"%arg3": valid},
            )
            assert c == expected, (previous, valid, c, expected)
            for access in machine.reads():
                if access["buffer"] in ("%arg0", "%arg1"):
                    assert access["start"] % 32 < valid, access
                if access["buffer"] == "%arg2":
                    assert previous and access["start"] < 8, access
    # CHECK: PASS: nonzero initializer/other, padded K, masks and repeated scratch reuse
    print(
        "PASS: nonzero initializer/other, padded K, masks and repeated scratch reuse"
    )


if __name__ == "__main__":
    main()
