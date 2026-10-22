# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Exercise post-execution DDR validation with independently built golden dumps."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from check_experimental_board import Symbol, check_dump  # noqa: E402

BASE = 0x80000000


class Fixture:
    def __init__(self):
        self.data = bytearray(3 * 1024 * 1024)
        self.symbols = {}
        self.next = 0x100000

    def scalar(self, name, value):
        offset = 0x1000 + 4 * len(self.symbols)
        self.symbols[name] = Symbol(BASE + offset, 4)
        self.data[offset : offset + 4] = value.to_bytes(4, "little")

    def guarded(self, name, values, guard):
        offset = self.next
        raw = np.ascontiguousarray(values).tobytes()
        guards = guard.to_bytes(4, "little") * 256
        payload = guards + raw + guards
        self.symbols[name] = Symbol(BASE + offset, len(payload))
        self.data[offset : offset + len(payload)] = payload
        self.next += (len(payload) + 4095) // 4096 * 4096


def layernorm_fixture():
    fixture = Fixture()
    # Build data independently of the checker's vectorized generators/oracle.
    x = np.empty((197, 768), dtype="<f4")
    weight, bias = np.empty(768, "<f4"), np.empty(768, "<f4")
    for column in range(768):
        weight[column] = (((column * 11 + 19) % 65) - 24) / 16
        bias[column] = (((column * 7 + 31) % 49) - 24) / 32
        for row in range(197):
            x[row, column] = (
                ((row * 37 + column * 13 + 71) % 511) - 255
            ) / 64 + (row % 7) / 4
    out = np.empty_like(x)
    f = np.float32
    for row in range(197):
        total = f(0)
        for value in x[row]:
            total = f(total + value)
        mean = f(total / f(768))
        squares = f(0)
        for column in range(1024):
            centered = f(x[row, column] - mean) if column < 768 else f(0)
            squares = f(squares + f(centered * centered))
        shifted = f(f(squares / f(768)) + f(1.0e-12))
        inverse = f(f(1) / f(math.sqrt(float(shifted))))
        for column in range(768):
            out[row, column] = f(
                f(f(f(x[row, column] - mean) * inverse) * weight[column])
                + bias[column]
            )
    for name, values in (
        ("x", x),
        ("weight", weight),
        ("bias", bias),
        ("out", out),
    ):
        fixture.guarded("layernorm_" + name, values, 0x4F400001)
    for name, value in (
        ("round", 1),
        ("checked", 302592),
        ("errors", 0),
        ("guard_errors", 0),
    ):
        fixture.scalar("layernorm_" + name, value)
    return fixture


def gemm_fixture():
    fixture = Fixture()
    # Broadcast and einsum independently of the checker's matmul implementation.
    i, j, q = np.indices((128, 1024, 128), sparse=True)
    a = (((i[:, 0, :] * 13 + q[0, 0, :] * 7 + 71) % 256) - 128).astype("i1")
    b = (((j[0, :, :] * 11 + q[0, 0, :] * 5 + 37) % 256) - 128).astype("i1")
    initial = 0x7FFFFF00 + ((i[:, :, 0] * 3 + j[:, :, 0] + 97) % 512)
    product = np.einsum("ik,jk->ij", a.astype(np.int64), b.astype(np.int64))
    c = np.remainder(initial + product, 2**32).astype("<u4")
    for name, values in (("a", a), ("b", b), ("c", c)):
        fixture.guarded(name + "_storage", values, 0x6E5A1234)
    for name, value in (
        ("round", 1),
        ("checked_rows", 256),
        ("errors", 0),
        ("guard_errors", 0),
    ):
        fixture.scalar("e6_diag_" + name, value)
    return fixture


def rejected(kind, fixture, data=None, symbols=None):
    report = check_dump(
        kind,
        fixture.data if data is None else data,
        fixture.symbols if symbols is None else symbols,
    )
    assert not report["passed"], report
    assert report["failures"], report
    return report


def check_corruptions(kind, fixture):
    report = check_dump(kind, fixture.data, fixture.symbols)
    assert report["passed"], report
    assert report["output_mismatches"] == 0
    assert (
        report["exact_final_round_values"]
        == report["final_round_values_checked"]
    )
    rejected(kind, fixture, data=fixture.data[:4096])
    for name, symbol in fixture.symbols.items():
        corrupted = bytearray(fixture.data)
        offset = symbol.address - BASE
        corrupted[offset] ^= 1
        rejected(kind, fixture, data=corrupted)
        if symbol.size > 4:
            corrupted = bytearray(fixture.data)
            corrupted[offset + symbol.size - 1] ^= 1
            rejected(kind, fixture, data=corrupted)
            # Damage payload: both output numerics and read-only inputs matter.
            corrupted = bytearray(fixture.data)
            corrupted[offset + 1024 : offset + 1028] = b"\0\0\xc0\x7f"
            rejected(kind, fixture, data=corrupted)
        else:
            missing = dict(fixture.symbols)
            del missing[name]
            rejected(kind, fixture, symbols=missing)
    oversized = dict(fixture.symbols)
    name = "layernorm_out" if kind == "layernorm" else "c_storage"
    oversized[name] = Symbol(fixture.symbols[name].address, 4)
    rejected(kind, fixture, symbols=oversized)
    overlap = dict(fixture.symbols)
    source = "layernorm_x" if kind == "layernorm" else "a_storage"
    overlap[name] = Symbol(overlap[source].address, overlap[name].size)
    rejected(kind, fixture, symbols=overlap)


def check_cli(work, fixture):
    # Supply a deterministic nm tool so this test needs no target linker.
    # The CLI still reads real dump/ELF files and must return nonzero on failure.
    work.mkdir(parents=True, exist_ok=True)
    nm = work / "fixture-nm"
    table = "\n".join(
        f"{name} B {symbol.address:x} {symbol.size:x}"
        for name, symbol in fixture.symbols.items()
    )
    nm.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "assert '--extern-only' in sys.argv\nprint(" + repr(table) + ")\n"
    )
    nm.chmod(0o755)
    elf, dump, report = (
        work / name for name in ("fixture.elf", "ddr.bin", "report.json")
    )
    elf.write_bytes(b"ELF identity for the mocked nm fixture")
    dump.write_bytes(fixture.data)
    command = [
        sys.executable,
        str(TOOLS / "check_experimental_board.py"),
        "gemm",
        "--elf",
        str(elf),
        "--dump",
        str(dump),
        "--nm",
        str(nm),
        "--report",
        str(report),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report.read_text())["passed"]
    dump.write_bytes(fixture.data[:4096])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not json.loads(report.read_text())["passed"]
    assert "does not cover" in result.stdout


def main():
    layernorm, gemm = layernorm_fixture(), gemm_fixture()
    check_corruptions("layernorm", layernorm)
    check_corruptions("gemm", gemm)
    check_cli(Path(sys.argv[1]), gemm)
    # CHECK: PASS board readback: full independent LayerNorm and signed/wrapping GEMM oracles
    print(
        "PASS board readback: full independent LayerNorm and signed/wrapping GEMM oracles"
    )
    # CHECK: PASS rejection: guards, inputs, outputs, incomplete diagnostics, short dumps and symbol errors
    print(
        "PASS rejection: guards, inputs, outputs, incomplete diagnostics, short dumps and symbol errors"
    )
    # CHECK: PASS CLI: JSON report and nonzero exit on invalid DDR
    print("PASS CLI: JSON report and nonzero exit on invalid DDR")


if __name__ == "__main__":
    main()
