#!/usr/bin/env python3
"""Check genuine TTIR-derived backend output against independent host oracles."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from simcheck_layernorm import oracle as layernorm_oracle

from build import REPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from boscame_buffer_sim import Buffer, Machine  # noqa: E402


def prepare(opt, directory, kernel):
    with tempfile.TemporaryDirectory() as temp:
        loops = Path(temp) / "kernel.loops.mlir"
        subprocess.run(
            [
                str(opt),
                str(directory / "kernel.boscame.mlir"),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            check=True,
        )
        return Machine(loops, "triton_" + kernel)


def launch(machine, roots, grid, pid):
    signature = machine.function.text.split("(", 1)[1].split(")", 1)[0]
    arguments = re.findall(r"(%[\w.\-]+)\s*:", signature)
    values = [*roots, *grid, *pid]
    if len(arguments) != len(values):
        raise RuntimeError("unexpected actual frontend argument count")
    machine.values = dict(zip(arguments, values))
    first = len(machine.accesses)
    machine.block(machine.function.blocks[0])
    if machine.pending or machine.live_matrix:
        raise RuntimeError("unfinished DMA or matrix values at return")
    return machine.accesses[first:]


def check_gemm(opt, directory):
    row = np.arange(128, dtype=np.int64)[:, None]
    depth = np.arange(1024, dtype=np.int64)[None, :]
    a = ((row * 13 + depth * 7) % 256 - 128).astype(np.int8)
    b = ((row * 29 + depth * 11 + 3) % 256 - 128).astype(np.int8)
    c = (
        (np.arange(16384, dtype=np.int64) * 123456789 + 0x7FFF0000)
        .astype("<u4")
        .view("<i4")
        .reshape(128, 128)
    )
    guard = bytes([0xA7]) * 64
    data = [bytearray(guard + value.tobytes() + guard) for value in (a, b, c)]
    roots = [
        Buffer(storage, element, [value.size], base=64, name=name)
        for storage, value, element, name in zip(
            data, (a, b, c), ("i8", "i8", "i32"), ("A", "B", "C")
        )
    ]
    machine = prepare(opt, directory, "gemm")
    for pid in (1, 0):
        accesses = launch(machine, roots, (2, 1, 1), (pid, 0, 0))
        a_begin = 64 + pid * 65536
        c_begin = 64 + pid * 32768
        expected = [
            ("A", a_begin, a_begin + 65536, "read"),
            ("B", 64, 64 + 131072, "read"),
            ("C", c_begin, c_begin + 32768, "read"),
            ("C", c_begin, c_begin + 32768, "write"),
        ]
        if sorted(accesses) != sorted(expected):
            raise RuntimeError(
                f"GEMM pid={pid} has unexpected user DDR accesses: {accesses[:8]}"
            )
    want = (
        c.astype(np.int64) + a.astype(np.int64) @ b.astype(np.int64).T
    ).astype("<i4")
    got = np.frombuffer(data[2], dtype="<i4", count=16384, offset=64).reshape(
        128, 128
    )
    if not np.array_equal(got, want):
        raise RuntimeError(
            "TTIR-derived GEMM differs from wrapping host oracle"
        )
    if (
        data[0] != guard + a.tobytes() + guard
        or data[1] != guard + b.tobytes() + guard
    ):
        raise RuntimeError("TTIR-derived GEMM changed an input or its guards")
    if data[2][:64] != guard or data[2][-64:] != guard:
        raise RuntimeError("TTIR-derived GEMM changed output guards")
    print(
        "FRONTEND SIM PASS: GEMM 16384 exact i32 outputs, 2 reverse programs, four authorized user DDR transfers/program"
    )


def check_layernorm(opt, directory, quick):
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
    expected_output = bytearray(data[3])
    roots = [
        Buffer(storage, "f32", [count], base=64, name=name)
        for storage, count, name in zip(
            data, (x.size, 768, 768, x.size), ("X", "Weight", "Bias", "Out")
        )
    ]
    machine = prepare(opt, directory, "layernorm")
    order = (196, 0, 1) if quick else range(196, -1, -1)
    max_error = 0.0
    for row in order:
        accesses = launch(machine, roots, (197, 1, 1), (row, 0, 0))
        begin, end = 64 + row * 3072, 64 + (row + 1) * 3072
        expected = [
            ("X", begin, end, "read"),
            ("Weight", 64, 3136, "read"),
            ("Bias", 64, 3136, "read"),
            ("Out", begin, end, "write"),
        ]
        if sorted(accesses) != sorted(expected):
            raise RuntimeError(
                f"LayerNorm row={row} has unexpected user DDR accesses: {accesses[:8]}"
            )
        want = layernorm_oracle(x[row], weight, bias)
        got = np.frombuffer(data[3][begin:end], dtype="<f4")
        np.testing.assert_allclose(
            got,
            want,
            rtol=2.0e-5,
            atol=2.0e-5,
            err_msg=f"TTIR-derived LayerNorm row={row} differs from ordered f32 host oracle",
        )
        max_error = max(max_error, float(np.max(np.abs(got - want))))
        expected_output[begin:end] = data[3][begin:end]
    if any(
        bytes(storage) != original
        for storage, original in zip(data[:3], inputs)
    ):
        raise RuntimeError("TTIR-derived LayerNorm changed inputs or guards")
    if data[3] != expected_output:
        raise RuntimeError(
            "TTIR-derived LayerNorm changed output guards/excluded rows"
        )
    print(
        f"FRONTEND SIM PASS: LayerNorm {len(order)} rows, padded ordered-f32 oracle rtol=2e-5, atol=2e-5; guards, last row and persistent SPM checked; max absolute error {max_error:.3g}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel", choices=("gemm", "layernorm", "all"), default="all"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "build/examples/linalg-boscame/frontend",
    )
    parser.add_argument(
        "--buddy-opt", type=Path, default=REPO / "build/bin/buddy-opt"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.kernel in ("gemm", "all"):
        check_gemm(args.buddy_opt, args.out_dir / "gemm")
    if args.kernel in ("layernorm", "all"):
        check_layernorm(args.buddy_opt, args.out_dir / "layernorm", args.quick)


if __name__ == "__main__":
    main()
