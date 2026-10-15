# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Full example grid: signed i8, physical [N,K] B, nonzero C and i32 wrap."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def main():
    source = Path(__file__).parent / "Inputs/matmul-i8.ttir"
    output = sys.argv[1] + ".boscame.mlir"
    loops = sys.argv[1] + ".loops.mlir"
    opt = buddy_opt_path()
    subprocess.run(
        [
            opt,
            str(source),
            "--triton-to-linalg-experimental=boscame=true dma-abi=args_aligned64_nonalias",
            "-o",
            output,
        ],
        check=True,
        capture_output=True,
    )
    text = Path(output).read_text()
    assert "bosc_ame.tile.mma" in text and "linalg.matmul" not in text
    assert "bosc_ame.lower_to_ame" not in text
    assert re.search(r"arith\.addi[^\n]*: vector<\d+xi32>", text), (
        "integer accumulation must use SPM vectors"
    )
    assert "memref.memory_space_cast" not in text
    assert "bufferization.to_tensor" not in text
    assert "bufferization.to_buffer" not in text
    subprocess.run(
        [
            opt,
            output,
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            loops,
        ],
        check=True,
        capture_output=True,
    )
    rng = np.random.default_rng(923)
    m, n, k = 128, 1024, 128
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (n, k), dtype=np.int8)
    # Alternating extrema and nonsymmetric inputs distinguish signedness/layout.
    a[0, ::2] = -128
    b[0, 1::2] = 127
    initial = rng.integers(-(1 << 31), 1 << 31, (m, n), dtype=np.int32)
    initial[0, :16] = np.iinfo(np.int32).max
    guard = bytes([0xB9]) * 1024
    buffers = {
        "%arg0": bytearray(a.tobytes() + guard),
        "%arg1": bytearray(b.tobytes() + guard),
        "%arg2": bytearray(initial.tobytes() + guard),
    }
    snapshots = {
        name: bytes(data) for name, data in buffers.items() if name != "%arg2"
    }
    machine = Machine(loops, "triton_matmul_i8_128x1024x128")
    # The independent oracle uses wide exact accumulation and wraps only at
    # the final i32 conversion, never any generated staging or tile indices.
    expected = (
        a.astype(np.int64) @ b.astype(np.int64).T + initial.astype(np.int64)
    ).astype(np.int32)
    for i in range(m // 16):
        for j in range(n // 16):
            machine.launch(
                buffers,
                {
                    "%arg3": m // 16,
                    "%arg4": n // 16,
                    "%arg5": 1,
                    "%arg6": i,
                    "%arg7": j,
                },
            )
    actual = np.frombuffer(
        buffers["%arg2"], dtype=np.int32, count=m * n
    ).reshape(m, n)
    np.testing.assert_array_equal(actual, expected)
    assert bytes(buffers["%arg2"][-len(guard) :]) == guard
    for name, snapshot in snapshots.items():
        assert bytes(buffers[name]) == snapshot
    # Only user ranges are logged; ordinary packing buffers have separate storage.
    limits = {"%arg0": a.nbytes, "%arg1": b.nbytes, "%arg2": initial.nbytes}
    assert all(
        0 <= start < end <= limits[name]
        for name, start, end, _ in machine.accesses
    )
    assert all(
        name == "%arg2"
        for name, _, _, kind in machine.accesses
        if kind == "write"
    )
    # CHECK: PASS GEMM: 512 programs, 131072 exact i32 outputs (signed i8, nonzero C, wrap, physical B)
    print(
        "PASS GEMM: 512 programs, 131072 exact i32 outputs (signed i8, nonzero C, wrap, physical B)"
    )
    # CHECK: PASS guards: no user DDR overread/overwrite, persistent SPM, two outer K blocks
    print(
        "PASS guards: no user DDR overread/overwrite, persistent SPM, two outer K blocks"
    )


if __name__ == "__main__":
    main()
