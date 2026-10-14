# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Real exported TTIR, persistent SPM, guards, and an independent padded oracle."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def lower(source, output, boscame):
    opt = buddy_opt_path()
    command = [
        opt,
        str(source),
        "--triton-to-linalg-experimental"
        + ("=boscame=true dma-abi=args_aligned64_nonalias" if boscame else ""),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    loops = str(output) + ".loops.mlir"
    command = [opt, str(output)]
    if not boscame:
        command += [
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries=true",
            "--promote-buffers-to-stack",
        ]
    command += [
        "--convert-linalg-to-loops",
        "--lower-affine",
        "--canonicalize",
        "-o",
        loops,
    ]
    subprocess.run(command, check=True, capture_output=True)
    return loops


def main():
    prefix = Path(sys.argv[1])
    source = Path(__file__).parent / "Inputs/layernorm.ttir"
    candidate = lower(source, str(prefix) + ".boscame.mlir", True)
    baseline = lower(source, str(prefix) + ".default.mlir", False)
    rng = np.random.default_rng(71)
    rows, width = 197, 768
    x = rng.normal(0.8, 1.5, (rows, width)).astype(np.float32)
    weight = rng.uniform(-1, 2, width).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, width).astype(np.float32)
    guard = bytes([0xC7]) * 1024
    buffers = {
        "%arg0": bytearray(x.tobytes() + guard),
        "%arg1": bytearray(weight.tobytes() + guard),
        "%arg2": bytearray(bias.tobytes() + guard),
        "%arg3": bytearray(bytes([0xDA]) * x.nbytes + guard),
    }
    reference_buffers = {
        name: bytearray(data) for name, data in buffers.items()
    }
    snapshots = {
        name: bytes(data) for name, data in buffers.items() if name != "%arg3"
    }
    machine = Machine(candidate, "triton_layernorm_197x768")
    reference = Machine(baseline, "triton_layernorm_197x768")
    # Reversed/interleaved programs reuse poisoned SPM banks and include the last row.
    order = list(range(0, rows, 2)) + list(range(1, rows, 2))
    if os.environ.get("BOSCAME_NUMERIC_QUICK"):
        order = [196, 0, 1]
    max_error = 0.0
    for row in order:
        scalars = {"%arg4": rows, "%arg5": 1, "%arg6": 1, "%arg7": row}
        machine.launch(buffers, scalars)
        reference.launch(reference_buffers, scalars)
        start, end = row * width * 4, (row + 1) * width * 4
        actual = np.frombuffer(buffers["%arg3"][start:end], dtype=np.float32)
        assert (
            buffers["%arg3"][start:end] == reference_buffers["%arg3"][start:end]
        ), "default experimental path differs"
        # Original TTIR centers ALL 1024 lanes before the second reduction.
        # The 256 padded X lanes therefore contribute mean**2 to variance.
        padded = np.pad(x[row].astype(np.float64), (0, 256))
        mean = padded.sum() / width
        variance = np.square(padded - mean).sum() / width
        expected = (
            (padded[:width] - mean) / np.sqrt(variance + np.float32(1.0e-12))
        ) * weight + bias
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        # An independent f32 oracle spells out each TTIR operation and the
        # sequential reduction selected by the unchanged Linalg loop path.
        # Compare exactly; the f64 formula above measures accumulated rounding,
        # including cancellation in the affine epilogue near zero.
        padded32 = np.pad(x[row], (0, 256))
        mean32 = np.float32(
            np.cumsum(padded32, dtype=np.float32)[-1] / np.float32(width)
        )
        centered32 = np.subtract(padded32, mean32, dtype=np.float32)
        squares32 = np.multiply(centered32, centered32, dtype=np.float32)
        var32 = np.float32(
            np.cumsum(squares32, dtype=np.float32)[-1] / np.float32(width)
        )
        shifted32 = np.float32(var32 + np.float32(1.0e-12))
        inverse32 = np.float32(1.0 / np.sqrt(np.float64(shifted32)))
        oracle32 = np.add(
            np.multiply(
                np.multiply(centered32[:width], inverse32, dtype=np.float32),
                weight,
                dtype=np.float32,
            ),
            bias,
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, oracle32)
        accesses = machine.accesses[-4:]
        assert accesses == [
            ("%arg0", start, end, "read"),
            ("%arg1", 0, width * 4, "read"),
            ("%arg2", 0, width * 4, "read"),
            ("%arg3", start, end, "write"),
        ], accesses
    for name, snapshot in snapshots.items():
        assert bytes(buffers[name]) == snapshot, "input/guard modified"
    assert bytes(buffers["%arg3"][-len(guard) :]) == guard
    # CHECK: PASS LayerNorm: 197 programs, three 3072 B loads and one 3072 B store per program
    print(
        f"PASS LayerNorm: {len(order)} programs, three 3072 B loads and one 3072 B store per program"
    )
    # CHECK: PASS numerics: bit-identical to default experimental buffer IR; independent padded TTIR oracle
    print(
        f"PASS numerics: bit-identical to default experimental buffer IR; independent padded TTIR oracle (max absolute error {max_error:.3g})"
    )
    # CHECK: PASS guards: last row, Weight/Bias tails and persistent poisoned SPM
    print(
        "PASS guards: last row, Weight/Bias tails and persistent poisoned SPM"
    )


if __name__ == "__main__":
    main()
