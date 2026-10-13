# RUN: %PYTHON %s %t.boscame.mlir 2>&1 | FileCheck %s
"""K = 256 with a 64-wide AME tile: the reduction must be blocked by a real
`scf.for` that carries the accumulator, with A/B loaded from SPM, and every
staged byte must come from a DMA that ran before the accumulator was loaded."""

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

M = N = K = 256
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Inputs",
    "k-blocked-64x64x256.ttir",
)


def signed(byte):
    return byte - 256 if byte >= 128 else byte


def wrap32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def oracle(a, b):
    result = []
    for i in range(64):
        for j in range(64):
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

    random.seed(20261014)
    a = bytearray(random.randrange(256) for _ in range(64 * K))
    b = bytearray(random.randrange(256) for _ in range(64 * K))
    c = bytearray([0] * (64 * 64 * 4))
    machine, _ = simulate(
        output,
        "k_blocked",
        {"%arg0": a, "%arg1": b, "%arg2": c, "%arg3": bytearray(16384)},
    )

    expected = oracle(a, b)
    actual = [
        int.from_bytes(c[i * 4 : i * 4 + 4], "little", signed=True)
        for i in range(64 * 64)
    ]
    mismatches = sum(1 for want, got in zip(expected, actual) if want != got)
    # CHECK: PASS oracle: 4096 of 4096 outputs match over the full K = 256
    if mismatches:
        print(f"FAIL oracle: {mismatches} outputs differ")
        return 1
    print("PASS oracle: 4096 of 4096 outputs match over the full K = 256")

    structure = Structure(output)
    loops = structure.loops_text()
    carried = step_ok = False
    for text in loops:
        if "iter_args" in text and "vector<64x64xi32>" in text:
            carried = True
            if "step %c64" in text:
                step_ok = True
    # CHECK: PASS structure: one accumulator-carrying scf.for with a 64-wide step
    if not (carried and step_ok and structure.accumulator_loops == 1):
        print(
            f"FAIL structure: accumulator loop {carried} step {step_ok} loops {structure.accumulator_loops}"
        )
        return 1
    print(
        "PASS structure: one accumulator-carrying scf.for with a 64-wide step"
    )

    # CHECK: PASS structure: the loop body issues no DMA and holds the single mma
    if structure.dma_in_loop != 0 or structure.tile["mma"] != 1:
        print(
            "FAIL structure: dma in loop {} mma {}".format(
                structure.dma_in_loop, structure.tile["mma"]
            )
        )
        return 1
    print(
        "PASS structure: the loop body issues no DMA and holds the single mma"
    )

    reads_a = len(structure.dma_reads("%arg0"))
    reads_b = len(structure.dma_reads("%arg1"))
    # CHECK: PASS staging: 64 A rows and 64 B rows arrive by DMA before the loop
    if (reads_a, reads_b) != (64, 64):
        print(f"FAIL staging: A={reads_a} B={reads_b}")
        return 1
    print("PASS staging: 64 A rows and 64 B rows arrive by DMA before the loop")

    lengths = {entry["length"] for entry in structure.dma_reads("%arg0")}
    # CHECK: PASS staging: every A row is one 256-byte transfer
    if lengths != {256}:
        print("FAIL staging: A transfer lengths " + str(lengths))
        return 1
    print("PASS staging: every A row is one 256-byte transfer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
