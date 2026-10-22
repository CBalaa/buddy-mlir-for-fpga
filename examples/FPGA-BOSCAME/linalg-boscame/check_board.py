#!/usr/bin/env python3
"""Validate actual post-execution DDR for either canonical NH/E6 image."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "tools"))
from check_experimental_board import (  # noqa: E402
    DDR,
    check_layernorm,
    compare_bits,
    load_symbols,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(manifest_path, elf, kernel, variant):
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("build manifest must be an object")
    for name in (
        "compiler_sha256",
        "llvm_ir_sha256",
        "elf_sha256",
        "bin_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(name, ""))):
            raise ValueError("build manifest lacks a valid " + name)
    if not re.fullmatch(
        r"[0-9a-f]{40,64}", str(manifest.get("compiler_commit", ""))
    ):
        raise ValueError("build manifest lacks a valid compiler_commit")
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("build manifest lacks source hashes")
    for source, digest in sources.items():
        path = Path(source)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        ):
            raise ValueError("invalid source hash in build manifest")
    for name, wanted in (("kernel", kernel), ("variant", variant)):
        if name in manifest and manifest[name] != wanted:
            raise ValueError(
                "build manifest " + name + " disagrees with this check"
            )
    for name, artifact in (
        ("elf_sha256", elf),
        ("bin_sha256", elf.with_suffix(".bin")),
        ("llvm_ir_sha256", elf.parent / "kernel.ll"),
    ):
        if sha256(artifact) != manifest[name]:
            raise ValueError(name + " does not match the build manifest")
    result = {
        name: manifest[name]
        for name in (
            "compiler_commit",
            "compiler_sha256",
            "source_sha256",
            "llvm_ir_sha256",
            "elf_sha256",
            "bin_sha256",
        )
    }
    for name in ("compiler_translate_sha256", "llc_sha256"):
        if name in manifest:
            if not re.fullmatch(r"[0-9a-f]{64}", str(manifest[name])):
                raise ValueError("invalid build manifest " + name)
            result[name] = manifest[name]
    result["build_manifest_sha256"] = sha256(manifest_path)
    return result


def check_gemm(ddr, failures):
    m, n, k = 128, 128, 1024
    a = ddr.guarded("gemm_a", "i1", m * k, 0x6E5A1234, failures)
    b = ddr.guarded("gemm_b", "i1", n * k, 0x6E5A1234, failures)
    c = ddr.guarded("gemm_c", "<u4", m * n, 0x6E5A1234, failures)
    row = np.arange(m, dtype=np.int64)[:, None]
    column = np.arange(n, dtype=np.int64)[:, None]
    inner = np.arange(k, dtype=np.int64)
    wanted_a = ((row * 13 + inner * 7 + 71) % 256 - 128).astype(np.int8)
    wanted_b = ((column * 11 + inner * 5 + 37) % 256 - 128).astype(np.int8)
    initial = (0x7FFFFF00 + (row * 3 + column.T + 97) % 512).astype(np.int64)
    wanted = (
        initial + wanted_a.astype(np.int64) @ wanted_b.astype(np.int64).T
    ).astype(np.uint32)
    compare_bits("gemm_a", a, wanted_a, failures)
    compare_bits("gemm_b", b, wanted_b, failures)
    errors = int(np.count_nonzero(c.reshape(m, n) != wanted))
    if errors:
        failures.append(f"gemm_c: {errors} i32 values differ")
    return {
        "shape": [m, n, k],
        "final_round_values_checked": m * n,
        "exact_final_round_values": m * n - errors,
        "output_mismatches": errors,
        "oracle": "independent i64 dot plus nonzero C, explicit i32 wrap",
        "max_absolute_error": 0 if not errors else None,
    }


def check_counters(ddr, kernel, variant, failures):
    count, programs = (16384, 2) if kernel == "gemm" else (151296, 197)
    for suffix, expected in (
        ("round", 1),
        ("checked", 2 * count),
        ("exact", 2 * count),
        ("errors", 0),
        ("guard_errors", 0),
    ):
        actual = ddr.scalar(f"{kernel}_{suffix}")
        if actual != expected:
            failures.append(f"{kernel}_{suffix}: {actual}, expected {expected}")
    result = {}
    for name, dtype in (
        ("programs", "<u4"),
        ("dma_starts", "<u8"),
        ("dma_waits", "<u8"),
        ("dma_bytes", "<u8"),
        ("user_reads", "<u8"),
        ("user_writes", "<u8"),
        ("user_read_bytes", "<u8"),
        ("user_write_bytes", "<u8"),
        ("kernel_cycles", "<u8"),
    ):
        result[name] = [int(v) for v in ddr.read("board_" + name, dtype, 2)]
    read_bytes = 458752 if kernel == "gemm" else 197 * 3 * 3072
    write_bytes = 65536 if kernel == "gemm" else 197 * 3072
    # Current images use DMA for pure SPM-to-SPM copies. Historical board
    # reports retain their original image identities and traffic counts.
    starts_per_program, bytes_per_program = {
        ("gemm", "buffer"): (4, 262144),
        ("gemm", "frontend"): (20, 1179648),
        ("layernorm", "buffer"): (25, 90112),
        ("layernorm", "frontend"): (33, 122880),
    }[kernel, variant]
    for round_index in range(2):
        for name, expected in (
            ("programs", programs),
            ("user_reads", programs * 3),
            ("user_writes", programs),
            ("user_read_bytes", read_bytes),
            ("user_write_bytes", write_bytes),
            ("dma_starts", programs * starts_per_program),
            ("dma_waits", programs * starts_per_program),
            ("dma_bytes", programs * bytes_per_program),
        ):
            if result[name][round_index] != expected:
                failures.append(f"round {round_index}: invalid {name}")
        if result["kernel_cycles"][round_index] <= 0:
            failures.append(f"round {round_index}: missing kernel cycles")
    result["app_cycles"] = int(ddr.read("board_app_cycles", "<u8", 1)[0])
    if result["app_cycles"] < sum(result["kernel_cycles"]):
        failures.append("application cycles do not cover both kernel launches")
    for name in ("e6_diag_mcause", "e6_diag_dma_reason", "e6_diag_dma_status"):
        dtype = "<u4" if name.endswith("reason") else "<u8"
        if int(ddr.read(name, dtype, 1)[0]):
            failures.append(f"nonzero {name}")
    result["scratch_dma_bytes"] = [
        total - read_bytes - write_bytes for total in result["dma_bytes"]
    ]
    if min(result["scratch_dma_bytes"]) < 0:
        failures.append("total DMA bytes below required user traffic")
    return result


def check_uart(uart, kernel, failures):
    prefix = rf"\[5\.5296MHz\]\[linalg-boscame-{kernel}\] "
    verdicts = re.findall(
        rf"^{prefix}(PASS|FAIL|E6 TRAP|NH TRAP|TIMEOUT)\r?$", uart, re.MULTILINE
    )
    if verdicts != ["PASS"]:
        failures.append(
            "UART must contain one successful terminal verdict for this kernel"
        )
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", choices=("gemm", "layernorm"))
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fpga", type=int, required=True)
    parser.add_argument(
        "--variant", choices=("buffer", "frontend"), required=True
    )
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--nm", type=Path, default=REPO / "llvm/build/bin/llvm-nm"
    )
    args = parser.parse_args()
    failures = []
    report = {
        "kernel": args.kernel,
        "variant": args.variant,
        "fpga": args.fpga,
        "clock_hz": 5529600,
        "timing_scope": "two independent kernel launches; wrappers count actual DMA",
    }
    try:
        report.update(
            verify_manifest(
                args.manifest or args.elf.parent / "build-manifest.json",
                args.elf,
                args.kernel,
                args.variant,
            )
        )
        report["dump_sha256"] = sha256(args.dump)
        ddr = DDR(
            args.dump.read_bytes(), load_symbols(args.elf, args.nm), 0x80000000
        )
        report.update(
            check_gemm(ddr, failures)
            if args.kernel == "gemm"
            else check_layernorm(ddr, failures)
        )
        report["counters"] = check_counters(
            ddr, args.kernel, args.variant, failures
        )
        report["uart_verdicts"] = check_uart(
            args.uart.read_text(errors="replace"), args.kernel, failures
        )
        report["uart_sha256"] = sha256(args.uart)
        if (
            report["exact_final_round_values"]
            != report["final_round_values_checked"]
        ):
            failures.append(
                "final output differs bit-for-bit from ordered oracle"
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        failures.append(str(error))
    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
