#!/usr/bin/env python3
"""Check standalone padded LayerNorm numerics and authorized DDR accesses.

This functional model compares against the ordinary buffer-form Linalg path
and an independent ordered-f32 oracle. It is not a board/performance test.
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


def oracle(row, weight, bias):
    """Round each original TTIR operation and each ordered addition to f32."""
    padded = np.pad(row, (0, 256))
    mean = np.float32(np.cumsum(padded, dtype=np.float32)[-1] / np.float32(768))
    centered = np.subtract(padded, mean, dtype=np.float32)
    centered[768:] = np.float32(0)
    squares = np.multiply(centered, centered, dtype=np.float32)
    variance = np.float32(
        np.cumsum(squares, dtype=np.float32)[-1] / np.float32(768)
    )
    shifted = np.float32(variance + np.float32(1.0e-12))
    # Match the emitted rsqrtf library call: sqrtf and division each round.
    inverse = np.divide(
        np.float32(1), np.sqrt(shifted, dtype=np.float32), dtype=np.float32
    )
    return np.add(
        np.multiply(
            np.multiply(centered[:768], inverse, dtype=np.float32),
            weight,
            dtype=np.float32,
        ),
        bias,
        dtype=np.float32,
    )


def prepare(opt, module, output):
    subprocess.run(
        [
            str(opt),
            str(module),
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            str(output),
        ],
        check=True,
    )
    return Machine(output, "layernorm_program")


def argument_names(machine):
    signature = machine.function.text.split("(", 1)[1].split(")", 1)[0]
    names = re.findall(r"(%[\w.\-]+)\s*:", signature)
    if len(names) != 4:
        raise RuntimeError("expected exactly four ranked memref arguments")
    return names


def execute(machine, names, data, row):
    starts = (64 + row * 3072, 64, 64, 64 + row * 3072)
    machine.values = {
        argument: Buffer(storage, "f32", [768], base=start, name=name)
        for argument, storage, start, name in zip(
            names, data, starts, ("X", "Weight", "Bias", "Out")
        )
    }
    first_access = len(machine.accesses)
    machine.block(machine.function.blocks[0])
    if machine.pending or machine.live_matrix:
        raise RuntimeError("unfinished DMA or matrix values at return")
    return machine.accesses[first_access:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        type=Path,
        default=REPO / "build/examples/linalg-boscame/layernorm.boscame.mlir",
    )
    parser.add_argument(
        "--buddy-opt", type=Path, default=REPO / "build/bin/buddy-opt"
    )
    parser.add_argument(
        "--quick", action="store_true", help="check rows 196, 0 and 1"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(73)
    x = rng.normal(0.8, 1.5, (197, 768)).astype("<f4")
    weight = rng.uniform(-1, 2, 768).astype("<f4")
    bias = rng.uniform(-0.5, 0.5, 768).astype("<f4")
    guard = bytes([0xC7]) * 64
    data = [
        bytearray(guard + value.tobytes() + guard)
        for value in (x, weight, bias)
    ]
    data.append(bytearray(guard + bytes([0xDA]) * x.nbytes + guard))
    inputs = [bytes(value) for value in data[:3]]
    baseline_data = [bytearray(value) for value in data]
    expected_output = bytearray(data[3])
    order = (
        [196, 0, 1]
        if args.quick
        else list(range(0, 197, 2)) + list(range(1, 197, 2))
    )
    with tempfile.TemporaryDirectory() as temp:
        candidate = prepare(
            args.buddy_opt, args.module, Path(temp) / "candidate.mlir"
        )
        baseline = prepare(
            args.buddy_opt,
            HERE / "layernorm.bufferized.mlir",
            Path(temp) / "baseline.mlir",
        )
    names = argument_names(candidate)
    baseline_names = argument_names(baseline)
    max_error = 0.0
    for row in order:
        accesses = execute(candidate, names, data, row)
        execute(baseline, baseline_names, baseline_data, row)
        begin, end = 64 + row * 3072, 64 + (row + 1) * 3072
        actual = np.frombuffer(data[3][begin:end], dtype="<f4")
        expected = oracle(x[row], weight, bias)
        ordinary = np.frombuffer(baseline_data[3][begin:end], dtype="<f4")
        for reference, label in (
            (expected, "ordered padded f32 oracle"),
            (ordinary, "ordinary Linalg lowering"),
        ):
            np.testing.assert_allclose(
                actual,
                reference,
                rtol=2.0e-5,
                atol=2.0e-5,
                err_msg=f"row {row} differs from {label}",
            )
            max_error = max(
                max_error, float(np.max(np.abs(actual - reference)))
            )
        # Tolerance applies only to computed values. Keep exact checks for
        # every guard byte and every row outside the selected launch order.
        expected_output[begin:end] = data[3][begin:end]
        if accesses != [
            ("X", begin, end, "read"),
            ("Weight", 64, 64 + 3072, "read"),
            ("Bias", 64, 64 + 3072, "read"),
            ("Out", begin, end, "write"),
        ]:
            raise RuntimeError(
                f"row {row} has unauthorized DDR accesses: {accesses}"
            )
    if any(
        bytes(value) != original for value, original in zip(data[:3], inputs)
    ):
        raise RuntimeError("an input or its guards changed")
    if data[3] != expected_output:
        raise RuntimeError("output guard or an excluded row changed")
    print(
        f"SIM CHECK PASS: LayerNorm {len(order)} rows, rtol=2e-5, atol=2e-5 against ordinary "
        "Linalg and padded ordered-f32 oracle; three 3072 B reads and one "
        "3072 B write per row; last row, guards and persistent SPM checked; "
        f"max absolute error {max_error:.3g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
