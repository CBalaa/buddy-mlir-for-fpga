#!/usr/bin/env python3
"""Check standalone BOSCAME GEMM output using the buffer functional model.

This executes both 64-row programs with persistent SPM and compares every i32
result with a NumPy int64 product followed by i32 wrapping. It is not a board
or performance measurement. Requires NumPy and the in-tree buddy-opt.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "tools"))
from boscame_buffer_sim import Buffer, Machine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        type=Path,
        default=REPO / "build/examples/linalg-boscame/gemm.boscame.mlir",
    )
    parser.add_argument(
        "--buddy-opt", type=Path, default=REPO / "build/bin/buddy-opt"
    )
    args = parser.parse_args()

    # All signed i8 values appear. B is deliberately asymmetric and physically
    # [N,K], so treating it as ordinary row-major logical [K,N] fails.
    rows = np.arange(128, dtype=np.int64)[:, None]
    depth = np.arange(1024, dtype=np.int64)[None, :]
    a = ((rows * 13 + depth * 7) % 256 - 128).astype(np.int8)
    b = ((rows * 29 + depth * 11 + 3) % 256 - 128).astype(np.int8)
    c0 = (
        (np.arange(128 * 128, dtype=np.int64) * 123456789 + 0x7FFF0000)
        .astype("<u4")
        .view("<i4")
        .reshape(128, 128)
    )
    guard = bytes([0xA7]) * 64
    data = [bytearray(guard + x.tobytes() + guard) for x in (a, b, c0)]
    with tempfile.TemporaryDirectory() as temp:
        loops = Path(temp) / "gemm.loops.mlir"
        subprocess.run(
            [
                str(args.buddy_opt),
                str(args.module),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            check=True,
        )
        machine = Machine(loops, "gemm_program")

    signature = machine.function.text.split("(", 1)[1].split(")", 1)[0]
    names = re.findall(r"(%[\w.\-]+)\s*:", signature)
    if len(names) != 3:
        raise RuntimeError("expected exactly three ranked memref arguments")
    access_counts = []
    # Run in reverse order to exercise caller slicing independently of pid order.
    for m0 in (64, 0):
        starts = (64 + m0 * 1024, 64, 64 + m0 * 128 * 4)
        lengths = (65536, 131072, 32768)
        machine.values = {
            names[0]: Buffer(
                data[0], "i8", [64, 1024], base=starts[0], name="A"
            ),
            names[1]: Buffer(
                data[1], "i8", [1024, 128], [1, 1024], base=starts[1], name="B"
            ),
            names[2]: Buffer(
                data[2], "i32", [64, 128], base=starts[2], name="C"
            ),
        }
        first_access = len(machine.accesses)
        machine.block(machine.function.blocks[0])
        if machine.pending or machine.live_matrix:
            raise RuntimeError("unfinished DMA or matrix values at return")
        allowed = dict(zip(("A", "B", "C"), zip(starts, lengths)))
        accesses = machine.accesses[first_access:]
        for name, start, end, kind in accesses:
            begin, length = allowed[name]
            if not begin <= start < end <= begin + length:
                raise RuntimeError(
                    f"program m0={m0} touched excluded {name} bytes"
                )
            if name != "C" and kind != "read":
                raise RuntimeError(f"program m0={m0} wrote input {name}")
        expected_accesses = [
            ("A", starts[0], starts[0] + lengths[0], "read"),
            ("B", starts[1], starts[1] + lengths[1], "read"),
            ("C", starts[2], starts[2] + lengths[2], "read"),
            ("C", starts[2], starts[2] + lengths[2], "write"),
        ]
        if sorted(accesses) != sorted(expected_accesses):
            raise RuntimeError(
                f"program m0={m0} expected four complete DDR transfers, "
                f"observed {len(accesses)} DDR accesses"
            )
        access_counts.append(len(accesses))

    want = (
        c0.astype(np.int64) + a.astype(np.int64) @ b.astype(np.int64).T
    ).astype("<i4")
    got = np.frombuffer(
        data[2], dtype="<i4", count=128 * 128, offset=64
    ).reshape(128, 128)
    if data[0] != guard + a.tobytes() + guard:
        raise RuntimeError("A or its guards changed")
    if data[1] != guard + b.tobytes() + guard:
        raise RuntimeError("B or its guards changed")
    if data[2][:64] != guard or data[2][-64:] != guard:
        raise RuntimeError("C guards changed")
    if not np.array_equal(got, want):
        row, column = np.argwhere(got != want)[0]
        raise RuntimeError(
            f"C[{row},{column}]={got[row, column]}, expected {want[row, column]}"
        )
    print(
        "SIM CHECK PASS: 16384 outputs match i32 wrapping oracle; "
        "2 programs, persistent SPM, input/output guards intact; "
        f"DDR access counts {access_counts}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
