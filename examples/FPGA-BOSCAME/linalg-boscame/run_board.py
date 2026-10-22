#!/usr/bin/env python3
"""Run a built image and validate post-execution DDR through an SSH runner."""

import argparse
import contextlib
import hashlib
import math
import os
import select
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent

from check_board import verify_manifest  # noqa: E402


def command(*args, **options):
    return subprocess.run([str(arg) for arg in args], check=True, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", choices=("gemm", "layernorm"))
    parser.add_argument(
        "--variant", choices=("buffer", "frontend"), required=True
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--host", required=True, help="existing passwordless SSH target"
    )
    parser.add_argument(
        "--remote-root", required=True, help="directory containing run.sh"
    )
    parser.add_argument("--fpga", type=int, choices=(4, 5, 6, 7), default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=60,
        help="remote runner inactivity window before natural cleanup (default: 60)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.idle_seconds) or args.idle_seconds <= 0:
        parser.error("--idle-seconds must be finite and positive")
    if any(character in args.remote_root for character in "{}\\\r\n\0"):
        parser.error("--remote-root contains an unsupported Tcl path character")
    image_dir = args.image_dir.resolve()
    image = image_dir / (args.kernel + ".bin")
    elf = image_dir / (args.kernel + ".elf")
    manifest = image_dir / "build-manifest.json"
    identity = verify_manifest(manifest, elf, args.kernel, args.variant)
    digest = identity["bin_sha256"]
    # Separate repeated executions of one image, including stale remote dumps.
    name = f"linalg-boscame-{args.kernel}-{args.variant}-{digest[:16]}-{uuid.uuid4().hex[:8]}"
    remote_root = args.remote_root.rstrip("/") or "/"
    remote_image = remote_root + "/test/" + name + ".bin"
    remote_dump = remote_root + "/runtime-output/" + name + ".final-ddr.bin"
    ssh_options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]
    ssh = ["ssh", *ssh_options, args.host]
    command(
        *ssh,
        "mkdir -p -- "
        + shlex.quote(remote_root + "/test")
        + " "
        + shlex.quote(remote_root + "/runtime-output"),
    )
    command("scp", *ssh_options, image, args.host + ":" + remote_image)
    uploaded = subprocess.check_output(
        [*ssh, "sha256sum " + shlex.quote(remote_image)], text=True
    ).split()[0]
    if uploaded != digest:
        raise SystemExit("uploaded image checksum differs")
    runner = (
        "cd "
        + shlex.quote(remote_root)
        + f" && ./run.sh --fpga={args.fpga} --delay={args.idle_seconds:g} "
        + shlex.quote("test=test/" + name + ".bin")
    )
    uart = image_dir / f"fpga{args.fpga}.uart.txt"
    dump = image_dir / f"fpga{args.fpga}.ddr.bin"
    report = image_dir / f"fpga{args.fpga}.report.json"
    raw_fd, raw_name = tempfile.mkstemp(prefix=name + "-", suffix=".uart.txt")
    # Keep stdin open: the final readmem must execute in this same live UVHS
    # session after the CPU's terminal verdict and diagnostics, never at load.
    process = subprocess.Popen(
        [*ssh[:-1], "-tt", ssh[-1], runner],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    pending, transcript = b"", []
    requested = readback = False
    started = terminal = False
    lifecycle = f"[5.5296MHz][linalg-boscame-{args.kernel}] "
    diagnostic = f"[linalg-boscame-{args.kernel}] "
    deadline = time.monotonic() + args.timeout
    try:
        with os.fdopen(raw_fd, "wb") as raw:
            while (
                process.poll() is None
                or select.select([process.stdout], [], [], 0)[0]
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "FPGA run exceeded the configured timeout"
                    )
                if not select.select([process.stdout], [], [], 1)[0]:
                    continue
                data = os.read(process.stdout.fileno(), 65536)
                if not data:
                    break
                raw.write(data)
                raw.flush()
                pending += data.replace(b"\r", b"")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    text = line.decode(errors="replace")
                    if text == lifecycle + "start":
                        started = True
                    if started and text.startswith((lifecycle, diagnostic)):
                        transcript.append(text)
                        # Progress rows are retained in the sanitized log.
                        if "E6 progress=" not in text:
                            print(text, flush=True)
                    verdict = (
                        text.removeprefix(lifecycle)
                        if text.startswith(lifecycle)
                        else ""
                    )
                    if started and verdict in (
                        "PASS",
                        "FAIL",
                        "E6 TRAP",
                        "NH TRAP",
                        "TIMEOUT",
                    ):
                        terminal = True
                    finished = (
                        started
                        and terminal
                        and (
                            text.startswith(diagnostic + "app_cycles=0x")
                            or verdict in ("E6 TRAP", "NH TRAP", "TIMEOUT")
                        )
                    )
                    if finished and not requested:
                        # Each DDR entry is 64 bytes: 49152 entries cover the
                        # 3 MiB containing code diagnostics and guarded arrays.
                        tcl = (
                            "readmem -rtl {nanhu_xilinx.i_ddr[49151:0]} -file {"
                            + remote_dump
                            + "} -file_type bin\n"
                        )
                        process.stdin.write(tcl.encode())
                        process.stdin.flush()
                        requested = True
                    if requested and "INFO:readback DDR success" in text:
                        readback = True
                        # Let run.sh finish through its idle cleanup. Sending
                        # exit here makes its event loop misclassify the
                        # normal UVHS "Total ERROR:0" summary as an error.
            status = process.wait(timeout=20)
        if status or not readback:
            raise RuntimeError(
                f"runner status={status}, final DDR readback={readback}; raw log: {raw_name}"
            )
    finally:
        uart.write_text("\n".join(transcript) + "\n")
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.stdin.write(b"exit\n")
                process.stdin.flush()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        with contextlib.suppress(OSError):
            process.stdin.close()
        process.stdout.close()
    command("scp", *ssh_options, args.host + ":" + remote_dump, dump)
    if dump.stat().st_size != 3 * 1024 * 1024:
        raise RuntimeError("final DDR dump does not contain exactly 3 MiB")
    remote_hash = subprocess.check_output(
        [*ssh, "sha256sum " + shlex.quote(remote_dump)], text=True
    ).split()[0]
    if hashlib.sha256(dump.read_bytes()).hexdigest() != remote_hash:
        raise RuntimeError("downloaded final DDR checksum differs")
    if verify_manifest(manifest, elf, args.kernel, args.variant) != identity:
        raise RuntimeError(
            "local image or build manifest changed during the board run"
        )
    command(
        sys.executable,
        HERE / "check_board.py",
        args.kernel,
        "--variant",
        args.variant,
        "--fpga",
        args.fpga,
        "--elf",
        elf,
        "--manifest",
        manifest,
        "--dump",
        dump,
        "--uart",
        uart,
        "--report",
        report,
    )


if __name__ == "__main__":
    main()
