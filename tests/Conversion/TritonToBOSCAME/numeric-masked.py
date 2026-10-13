# RUN: %PYTHON %s %t.boscame.mlir 2>&1 | FileCheck %s
"""64x64x64 i8 dot with 63 valid columns per row, A rows 128 bytes apart.

The DDR bytes the masks exclude hold a poison pattern, so the test can tell
"the pass read garbage and zeroed it" from "the pass never read it": the
results must match an oracle that uses `other` = 0, and the DDR access record
must not contain a single excluded byte.  A 63-byte row is not a 64-byte
transfer, so the valid elements are packed into DDR scratch before a complete-row DMA.
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

from boscame_check import Structure, buddy_opt_path, run_pass  # noqa: E402
from boscame_sim import simulate  # noqa: E402

M = N = K = 64
VALID = 63
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Inputs",
    "masked-a-stride128.ttir",
)


def signed(byte):
    return byte - 256 if byte >= 128 else byte


def wrap32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def oracle(a, b):
    result = []
    for i in range(M):
        for j in range(N):
            total = 0
            for k in range(VALID):
                total += signed(a[i * 128 + k]) * signed(b[j * K + k])
            result.append(wrap32(total))
    return result


def main():
    output = sys.argv[1]
    code, stderr = run_pass(buddy_opt_path(), FIXTURE, output)
    if code != 0:
        print("FAIL: the pass failed:", stderr)
        return 1

    random.seed(20261013)
    a = bytearray(random.randrange(256) for _ in range(M * 128))
    b = bytearray(random.randrange(256) for _ in range(N * K))
    for i in range(M):
        a[i * 128 + VALID] = 0x7F  # poison: never a valid A byte
    for j in range(N):
        b[j * K + VALID] = 0x7F  # poison: never a valid B byte
    c = bytearray([0] * (M * N * 4))

    machine, _ = simulate(
        output,
        "masked_dot",
        {"%arg0": a, "%arg1": b, "%arg2": c, "%arg3": bytearray(16384)},
    )

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
        print(f"FAIL oracle: {mismatches} of {len(expected)} outputs differ")
        return 1
    # CHECK: PASS oracle: 4096 of 4096 outputs match the `other` = 0 oracle
    print("PASS oracle: 4096 of 4096 outputs match the `other` = 0 oracle")

    poison = [
        (entry["buffer"], entry["start"])
        for entry in machine.reads()
        if (entry["buffer"] == "%arg0" and entry["start"] % 128 == VALID)
        or (entry["buffer"] == "%arg1" and entry["start"] % K == VALID)
    ]
    # CHECK: PASS memory: 0 DDR reads of the bytes the masks exclude
    if poison:
        print(
            f"FAIL memory: {len(poison)} excluded DDR reads, e.g. {poison[:3]}"
        )
        return 1
    print("PASS memory: 0 DDR reads of the bytes the masks exclude")

    structure = Structure(output)
    reads_a = len(structure.dma_reads("%arg0"))
    reads_b = len(structure.dma_reads("%arg1"))
    writes_scratch = structure.dma_writes("%arg3")
    # CHECK: PASS dma: no 64 B transfer for the 63 valid bytes of a row
    if reads_a or reads_b:
        print(
            f"FAIL dma: A reads {reads_a} B reads {reads_b}; a 63-byte row must not use a "
            "64 B transfer"
        )
        return 1
    print("PASS dma: no 64 B transfer for the 63 valid bytes of a row")

    # CHECK: PASS write-back: one 16384-byte DMA to DDR scratch, masked CPU scatter
    if len(writes_scratch) != 1 or writes_scratch[0]["length"] != M * N * 4:
        print("FAIL write-back: " + str(writes_scratch))
        return 1
    print(
        "PASS write-back: one 16384-byte DMA to DDR scratch, masked CPU scatter"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
