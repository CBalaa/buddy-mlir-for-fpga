#!/usr/bin/env python3
"""Exercise board-result rejection without contacting a board."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import manifest
import numpy as np
import run_board
from check_board import (
    DDR,
    check_counters,
    check_gemm,
    check_layernorm,
    check_uart,
    sha256,
    verify_manifest,
)
from check_experimental_board import Symbol, layernorm_inputs, layernorm_oracle


class DumpFixture:
    def __init__(self, kernel, variant):
        self.kernel, self.variant = kernel, variant
        self.data = bytearray()
        self.symbols = {}
        count, programs = (16384, 2) if kernel == "gemm" else (151296, 197)
        if kernel == "gemm":
            row = np.arange(128, dtype=np.int64)[:, None]
            inner = np.arange(1024, dtype=np.int64)
            a = ((row * 13 + inner * 7 + 71) % 256 - 128).astype("i1")
            b = ((row * 11 + inner * 5 + 37) % 256 - 128).astype("i1")
            initial = 0x7FFFFF00 + (row * 3 + row.T + 97) % 512
            output = (initial + a.astype("i8") @ b.astype("i8").T).astype("<u4")
            arrays = {"a": a, "b": b, "c": output}
            sentinel = 0x6E5A1234
        else:
            x, weight, bias = layernorm_inputs(1)
            arrays = {
                "x": x,
                "weight": weight,
                "bias": bias,
                "out": layernorm_oracle(x, weight, bias),
            }
            sentinel = 0x4F400001
        for name, value in arrays.items():
            guard = np.full(256, sentinel, dtype="<u4").tobytes()
            self.add(kernel + "_" + name, guard + value.tobytes() + guard)
        for name, value in {
            "round": 1,
            "checked": 2 * count,
            "exact": 2 * count,
            "errors": 0,
            "guard_errors": 0,
        }.items():
            self.add(
                kernel + "_" + name, np.array([value], dtype="<u4").tobytes()
            )
        requests, total = {
            ("gemm", "buffer"): (8, 524288),
            ("gemm", "frontend"): (40, 2359296),
            ("layernorm", "buffer"): (4925, 17752064),
            ("layernorm", "frontend"): (6501, 24207360),
        }[kernel, variant]
        counters = {
            "programs": programs,
            "dma_starts": requests,
            "dma_waits": requests,
            "dma_bytes": total,
            "user_reads": programs * 3,
            "user_writes": programs,
            "user_read_bytes": 458752 if kernel == "gemm" else 1815552,
            "user_write_bytes": 65536 if kernel == "gemm" else 605184,
            "kernel_cycles": 1000,
        }
        for name, value in counters.items():
            self.add(
                "board_" + name,
                np.full(
                    2, value, dtype="<u4" if name == "programs" else "<u8"
                ).tobytes(),
            )
        self.add("board_app_cycles", np.array([3000], dtype="<u8").tobytes())
        for name in ("mcause", "dma_reason", "dma_status"):
            self.add("e6_diag_" + name, bytes(4 if name == "dma_reason" else 8))

    def add(self, name, value):
        self.symbols[name] = Symbol(0x80000000 + len(self.data), len(value))
        self.data.extend(value)

    def corrupt(self, name, offset=0):
        address = self.symbols[name].address - 0x80000000 + offset
        self.data[address] ^= 1

    def failures(self):
        failures = []
        ddr = DDR(self.data, self.symbols, 0x80000000)
        check = check_gemm if self.kernel == "gemm" else check_layernorm
        report = check(ddr, failures)
        check_counters(ddr, self.kernel, self.variant, failures)
        if (
            report["exact_final_round_values"]
            != report["final_round_values_checked"]
        ):
            failures.append("output differs bit for bit")
        return failures


class BoardCheckerTest(unittest.TestCase):
    def test_valid_dump_and_corruptions(self):
        for kernel in ("gemm", "layernorm"):
            for variant in ("buffer", "frontend"):
                with self.subTest(kernel=kernel, variant=variant):
                    fixture = DumpFixture(kernel, variant)
                    self.assertEqual(fixture.failures(), [])
                    historical = {
                        ("gemm", "frontend"): (32, 1835008),
                        ("layernorm", "buffer"): (4137, 15331328),
                        ("layernorm", "frontend"): (5713, 21786624),
                    }.get((kernel, variant))
                    if historical:
                        original = fixture.data[:]
                        requests, total = historical
                        for name, value in (
                            ("dma_starts", requests),
                            ("dma_waits", requests),
                            ("dma_bytes", total),
                        ):
                            address = (
                                fixture.symbols["board_" + name].address
                                - 0x80000000
                            )
                            fixture.data[address : address + 16] = np.full(
                                2, value, dtype="<u8"
                            ).tobytes()
                        failures = fixture.failures()
                        for name in ("dma_starts", "dma_waits", "dma_bytes"):
                            self.assertTrue(
                                any("invalid " + name in f for f in failures),
                                "old vector-copy traffic accepted as DMA-copy image",
                            )
                        fixture.data[:] = original
                    output = kernel + ("_c" if kernel == "gemm" else "_out")
                    for symbol, offset in (
                        (output, 0),
                        (output, fixture.symbols[output].size - 1025),
                        (
                            kernel + ("_b" if kernel == "gemm" else "_weight"),
                            1024,
                        ),
                        ("board_dma_starts", 0),
                        ("board_dma_bytes", 0),
                        ("board_user_read_bytes", 0),
                        ("e6_diag_mcause", 0),
                    ):
                        fixture.corrupt(symbol, offset)
                        self.assertTrue(
                            fixture.failures(),
                            (kernel, variant, symbol, offset),
                        )
                        fixture.corrupt(symbol, offset)
                    fixture.corrupt("board_dma_starts")
                    fixture.corrupt("board_dma_waits")
                    self.assertTrue(
                        fixture.failures(), "equal but incorrect starts/waits"
                    )
                    fixture.corrupt("board_dma_starts")
                    fixture.corrupt("board_dma_waits")
                    address = (
                        fixture.symbols["board_app_cycles"].address - 0x80000000
                    )
                    fixture.data[address : address + 8] = (1000).to_bytes(
                        8, "little"
                    )
                    self.assertTrue(
                        fixture.failures(), "application excludes kernel time"
                    )

    def test_truncated_or_missing_dump_data(self):
        fixture = DumpFixture("gemm", "buffer")
        fixture.data.pop()
        with self.assertRaisesRegex(ValueError, "does not cover"):
            fixture.failures()
        fixture = DumpFixture("layernorm", "frontend")
        del fixture.symbols["layernorm_out"]
        with self.assertRaisesRegex(ValueError, "missing ELF symbol"):
            fixture.failures()

    def test_uart_terminal_verdict(self):
        prefix = "[5.5296MHz][linalg-boscame-gemm] "
        failures = []
        check_uart(prefix + "PASS\r\n", "gemm", failures)
        self.assertEqual(failures, [])
        for text in (
            "BYPASS",
            prefix + "FAIL\r\n",
            prefix + "PASS\r\n" + prefix + "TIMEOUT\r\n",
            (prefix + "PASS\r\n") * 2,
            prefix.replace("gemm", "layernorm") + "PASS\r\n",
        ):
            failures = []
            check_uart(text, "gemm", failures)
            self.assertTrue(failures, text)

    def test_manifest_artifact_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            elf, binary, llvm_ir = (
                root / "gemm.elf",
                root / "gemm.bin",
                root / "kernel.ll",
            )
            for artifact in (elf, binary, llvm_ir):
                artifact.write_bytes(artifact.name.encode())
            manifest = {
                "compiler_commit": "1" * 40,
                "compiler_sha256": "2" * 64,
                "source_sha256": {"examples/kernel.c": "3" * 64},
                "llvm_ir_sha256": sha256(llvm_ir),
                "elf_sha256": sha256(elf),
                "bin_sha256": sha256(binary),
                "kernel": "gemm",
                "variant": "frontend",
            }
            path = root / "build-manifest.json"
            path.write_text(json.dumps(manifest))
            report = verify_manifest(path, elf, "gemm", "frontend")
            self.assertEqual(report["compiler_sha256"], "2" * 64)
            for artifact in (elf, binary, llvm_ir):
                original = artifact.read_bytes()
                artifact.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    verify_manifest(path, elf, "gemm", "frontend")
                artifact.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "variant disagrees"):
                verify_manifest(path, elf, "gemm", "buffer")
            del manifest["compiler_sha256"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "valid compiler_sha256"):
                verify_manifest(path, elf, "gemm", "frontend")

    def test_capture_rejects_changed_build_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "kernel.c").write_text("initial source")
            tool = root / "tool"
            tool.write_bytes(b"compiler")
            out = root / "image"
            out.mkdir()
            manifest_path = out / "build-manifest.json"
            manifest_path.write_text("stale manifest")
            common = [
                "--out",
                str(out),
                "--buddy-opt",
                str(tool),
                "--buddy-translate",
                str(tool),
                "--llc",
                str(tool),
                "--kernel",
                "gemm",
                "--variant",
                "buffer",
            ]
            with (
                patch.object(manifest, "REPO", root),
                patch.object(manifest, "SOURCE_DIRS", ("source",)),
                patch.object(
                    manifest.subprocess, "check_output", return_value="1" * 40
                ),
            ):
                with patch.object(
                    sys, "argv", ["manifest.py", "capture", *common]
                ):
                    manifest.main()
                self.assertFalse(manifest_path.exists())
                (source / "kernel.c").write_text("changed source")
                with (
                    patch.object(
                        sys, "argv", ["manifest.py", "finish", *common]
                    ),
                    self.assertRaisesRegex(SystemExit, "changed during build"),
                ):
                    manifest.main()

    def test_runner_rejects_image_rebinding(self):
        original_popen = subprocess.Popen
        for rebind, runner_status, idle_seconds in (
            (False, 0, None),
            (True, 0, None),
            (False, 1, None),
            (False, 0, 240),
        ):
            with (
                self.subTest(
                    rebind=rebind,
                    runner_status=runner_status,
                    idle_seconds=idle_seconds,
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                elf, binary, llvm_ir = (
                    root / "gemm.elf",
                    root / "gemm.bin",
                    root / "kernel.ll",
                )
                for artifact in (elf, binary, llvm_ir):
                    artifact.write_bytes(artifact.name.encode())
                identity = {
                    "compiler_commit": "1" * 40,
                    "compiler_sha256": "2" * 64,
                    "source_sha256": {"kernel.c": "3" * 64},
                    "elf_sha256": sha256(elf),
                    "bin_sha256": sha256(binary),
                    "llvm_ir_sha256": sha256(llvm_ir),
                    "kernel": "gemm",
                    "variant": "buffer",
                }
                path = root / "build-manifest.json"
                path.write_text(json.dumps(identity))
                dump_data = bytes(3 * 1024 * 1024)
                checker_calls = []

                def fake_command(
                    *arguments,
                    dump_data=dump_data,
                    checker_calls=checker_calls,
                    **options,
                ):
                    if arguments[0] == "scp" and str(arguments[-2]).endswith(
                        ".final-ddr.bin"
                    ):
                        Path(arguments[-1]).write_bytes(dump_data)
                    if arguments[0] == sys.executable:
                        checker_calls.append(arguments)

                def fake_hash(
                    arguments, dump_data=dump_data, identity=identity, **options
                ):
                    if "final-ddr.bin" in arguments[-1]:
                        return hashlib.sha256(dump_data).hexdigest() + " dump\n"
                    return identity["bin_sha256"] + " image\n"

                def fake_runner(
                    *arguments,
                    rebind=rebind,
                    runner_status=runner_status,
                    idle_seconds=idle_seconds,
                    identity=identity,
                    path=path,
                    **options,
                ):
                    self.assertIn(
                        f"--delay={idle_seconds or 60} ", arguments[0][-1]
                    )
                    if rebind:
                        identity["compiler_sha256"] = "4" * 64
                        path.write_text(json.dumps(identity))
                    code = """import select
import sys
print("INFO:readback DDR success", flush=True)
print("[linalg-boscame-gemm] app_cycles=0x01", flush=True)
print("[5.5296MHz][linalg-boscame-gemm] start", flush=True)
print("[5.5296MHz][linalg-boscame-gemm] PASS", flush=True)
print("[linalg-boscame-gemm] app_cycles=0x02", flush=True)
assert sys.stdin.readline().startswith("readmem -rtl ")
print("INFO:readback DDR success", flush=True)
assert not select.select([sys.stdin], [], [], 0.2)[0], "readback must await idle cleanup"
print("Total ERROR:0", flush=True)
"""
                    code += f"sys.exit({runner_status})\n"
                    return original_popen(
                        [sys.executable, "-c", code],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )

                arguments = [
                    "run_board.py",
                    "gemm",
                    "--variant",
                    "buffer",
                    "--image-dir",
                    str(root),
                    "--host",
                    "unused",
                    "--remote-root",
                    "/unused",
                    "--timeout",
                    "10",
                ]
                if idle_seconds is not None:
                    arguments.extend(["--idle-seconds", str(idle_seconds)])
                with (
                    patch.object(sys, "argv", arguments),
                    patch.object(
                        run_board, "command", side_effect=fake_command
                    ),
                    patch.object(
                        run_board.subprocess,
                        "check_output",
                        side_effect=fake_hash,
                    ),
                    patch.object(
                        run_board.subprocess, "Popen", side_effect=fake_runner
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if runner_status:
                        with self.assertRaisesRegex(
                            RuntimeError, "runner status=1"
                        ):
                            run_board.main()
                    elif rebind:
                        with self.assertRaisesRegex(
                            RuntimeError, "changed during the board run"
                        ):
                            run_board.main()
                    else:
                        run_board.main()
                self.assertEqual(
                    len(checker_calls), 0 if rebind or runner_status else 1
                )
                uart = (root / "fpga4.uart.txt").read_text()
                self.assertNotIn("app_cycles=0x01", uart)
                self.assertIn("app_cycles=0x02", uart)


if __name__ == "__main__":
    unittest.main()
