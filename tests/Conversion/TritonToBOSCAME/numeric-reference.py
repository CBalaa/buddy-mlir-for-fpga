# RUN: %PYTHON %s %t.boscame.mlir 2>&1 | FileCheck %s
"""The reference i8 kernel (196x768x768, 16x16x256 dots) end to end.

Two programs are simulated: the first M tile and the last, partial one, so the
boundary mask, the `previous` load/add/store epilogue and the masked store that
must leave the elements it excludes untouched are all exercised.  The DDR is
pre-filled with sentinels outside the kernel's valid region: reading or writing
them would be caught.
"""

import os
import random
import sys

TOOLS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "examples",
    "FPGA-BOSCAME",
    "tools",
)
sys.path.insert(0, os.path.abspath(TOOLS))

from boscame_check import buddy_opt_path, run_pass  # noqa: E402
from boscame_sim import simulate  # noqa: E402

M, N, K = 196, 768, 768
TILE = 16
BLOCK = 256
C_ROWS = 208  # the buffer carries sentinel rows past M
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Inputs",
    "reference-matmul-i8-196x768x768.ttir",
)


def signed(byte):
    return byte - 256 if byte >= 128 else byte


def wrap32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def oracle(a, b, c, pid_x, pid_y):
    """Independent model of one program: three 256-wide K blocks, in place."""
    for i in range(TILE):
        row = pid_x * TILE + i
        if row >= M:
            continue
        for j in range(TILE):
            column = pid_y * TILE + j
            if column >= N:
                continue
            offset = (row * N + column) * 4
            total = int.from_bytes(
                c[offset : offset + 4], "little", signed=True
            )
            for k in range(K):
                total += signed(a[row * K + k]) * signed(b[column * K + k])
            c[offset : offset + 4] = wrap32(total).to_bytes(
                4, "little", signed=True
            )


def main():
    output = sys.argv[1]
    code, stderr = run_pass(buddy_opt_path(), FIXTURE, output)
    if code != 0:
        print("FAIL: the pass failed:", stderr)
        return 1

    random.seed(20261015)
    a = bytearray(random.randrange(256) for _ in range(M * K))
    b = bytearray(random.randrange(256) for _ in range(N * K))
    c = bytearray(random.randrange(256) for _ in range(C_ROWS * N * 4))
    grid = {"%arg4": (M + TILE - 1) // TILE, "%arg5": N // TILE, "%arg6": 1}

    for pid_x in (0, (M + TILE - 1) // TILE - 1):
        for pid_y in (0, 1):
            state = {"a": bytearray(a), "b": bytearray(b), "c": bytearray(c)}
            expected = bytearray(c)
            oracle(state["a"], state["b"], expected, pid_x, pid_y)
            scalars = dict(grid)
            scalars.update({"%arg7": pid_x, "%arg8": pid_y, "%arg9": 0})
            machine, _ = simulate(
                output,
                "triton_matmul_i8_196x768x768",
                {
                    "%arg0": state["a"],
                    "%arg1": state["b"],
                    "%arg2": state["c"],
                    "%arg3": bytearray([0xA5] * 4096),
                },
                scalars,
            )
            mismatches = sum(
                1
                for offset in range(M * N * 4)
                if state["c"][offset] != expected[offset]
            )
            if mismatches:
                print(
                    f"FAIL oracle: program ({pid_x}, {pid_y}) has {mismatches} differing bytes"
                )
                return 1
            sentinel = bytearray(c[M * N * 4 :])
            if state["c"][M * N * 4 :] != sentinel:
                print(
                    f"FAIL sentinel: program ({pid_x}, {pid_y}) wrote outside its M tile"
                )
                return 1
            invalid_reads = [
                entry
                for entry in machine.reads()
                if entry["buffer"] == "%arg0" and entry["start"] >= M * K
            ]
            invalid_reads += [
                entry
                for entry in machine.reads()
                if entry["buffer"] == "%arg2" and entry["start"] >= M * N * 4
            ]
            if invalid_reads:
                print(
                    f"FAIL mask: program ({pid_x}, {pid_y}) read {len(invalid_reads)} excluded DDR bytes"
                )
                return 1
    # CHECK: PASS oracle: 4 programs match the in-place `previous + A x B` model byte for byte
    print(
        "PASS oracle: 4 programs match the in-place `previous + A x B` model "
        "byte for byte"
    )
    # CHECK: PASS sentinel: the rows past M keep their DDR values
    print("PASS sentinel: the rows past M keep their DDR values")
    # CHECK: PASS mask: no DDR byte outside the valid region is ever read
    print("PASS mask: no DDR byte outside the valid region is ever read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
