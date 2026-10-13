# RUN: %PYTHON %s %t.boscame.mlir 2>&1 | FileCheck %s
"""Dense 64x64x64 i8 dot: the pass output is simulated and compared, bit for
bit, against an oracle written independently of the pass."""

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

from boscame_check import Structure, buddy_opt_path, run_pass  # noqa: E402
from boscame_sim import simulate  # noqa: E402

M = N = K = 64
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Inputs",
    "single-dot-64x64x64.ttir",
)


def signed(byte):
    return byte - 256 if byte >= 128 else byte


def wrap32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def oracle(a, b):
    """Independent model: C = A x B, int8 x int8 -> int32, wrapping."""
    result = []
    for i in range(M):
        for j in range(N):
            total = 0
            for k in range(K):
                total += signed(a[i * K + k]) * signed(b[j * K + k])
            result.append(wrap32(total))
    return result


def main():
    output = sys.argv[1]
    code, stderr = run_pass(buddy_opt_path(), FIXTURE, output)
    if code != 0:
        print("FAIL: the pass failed:", stderr)
        return 1

    random.seed(20261012)
    a = bytearray(
        random.choice([random.randrange(256), -128 & 0xFF, 127, 0])
        for _ in range(M * K)
    )
    b = bytearray(
        random.choice([random.randrange(256), -128 & 0xFF, 127, 0])
        for _ in range(N * K)
    )
    c = bytearray([0] * (M * N * 4))

    machine, _ = simulate(
        output,
        "single_dot64",
        {"%arg0": a, "%arg1": b, "%arg2": c, "%arg3": bytearray(16384)},
    )
    # CHECK: PASS oracle: 4096 of 4096 outputs match bit for bit
    expected = oracle(a, b)
    actual = [
        int.from_bytes(
            c[(i * N + j) * 4 : (i * N + j) * 4 + 4], "little", signed=True
        )
        for i in range(M)
        for j in range(N)
    ]
    mismatches = sum(1 for want, got in zip(expected, actual) if want != got)
    if mismatches:
        print(
            f"FAIL oracle: {mismatches} of {len(expected)} outputs differ, first: {next(w for w, g in zip(expected, actual) if w != g)} vs {next(g for w, g in zip(expected, actual) if w != g)}"
        )
        return 1
    print(
        f"PASS oracle: {len(expected)} of {len(expected)} outputs match bit for bit"
    )

    structure = Structure(output)
    reads_a = len(structure.dma_reads("%arg0"))
    reads_b = len(structure.dma_reads("%arg1"))
    writes_scratch = len(structure.dma_writes("%arg3"))
    # CHECK: PASS dma: 64 A rows, 64 B rows, one SPM-to-DDR result transfer
    if (reads_a, reads_b, writes_scratch) != (64, 64, 1):
        print(f"FAIL dma: A={reads_a} B={reads_b} scratch={writes_scratch}")
        return 1
    print("PASS dma: 64 A rows, 64 B rows, one SPM-to-DDR result transfer")

    # CHECK: PASS tile: load_acc=1 load_a=1 load_b=1 mma=1 store_acc=1
    tile = structure.tile
    if tile != {
        "load_acc": 1,
        "load_a": 1,
        "load_b": 1,
        "mma": 1,
        "store_acc": 1,
    }:
        print("FAIL tile: " + str(tile))
        return 1
    print("PASS tile: load_acc=1 load_a=1 load_b=1 mma=1 store_acc=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
