# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Check AME A/B route choices against exact static read traffic."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from boscame_check import buddy_opt_path


def module(m, n, k):
    return f"""module attributes {{
  bosc_ame.dma.abi = "args_aligned64_nonalias",
  bosc_ame.placement = "logical"
}} {{
  func.func @gemm(%a: memref<{m}x{k}xi8>,
                  %b: memref<{k}x{n}xi8, strided<[1, {k}]>>,
                  %c: memref<{m}x{n}xi32>) {{
    linalg.matmul {{bosc_ame.lower_to_ame,
                   cast = #linalg.type_fn<cast_signed>}}
      ins(%a, %b : memref<{m}x{k}xi8>,
                   memref<{k}x{n}xi8, strided<[1, {k}]>>)
      outs(%c : memref<{m}x{n}xi32>)
    return
  }}
}}
"""


def lower(opt, input_path, report=True, spm_enabled=True):
    options = "dma-abi=args_aligned64_nonalias target=nr-fpga"
    if not spm_enabled:
        options += " enable-spm-dma=false"
    if report:
        options += " plan-report=true"
    return subprocess.run(
        [
            str(opt),
            str(input_path),
            f"--linalg-to-boscame={options}",
            "-o",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write(work, name, m, n, k):
    path = work / f"{name}.mlir"
    path.write_text(module(m, n, k))
    return path


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = Path(buddy_opt_path())

    one_tile = write(work, "one-tile", 64, 64, 64)
    direct = lower(opt, one_tile)
    assert direct.returncode == 0, direct.stderr
    route = next(
        line for line in direct.stderr.splitlines() if "plan.route=" in line
    )
    assert route.count('level = "ddr"') == 4, route
    assert "bytes = 4096 : i64" in route
    assert 'reason = "safe-terminal-forwarding"' in route
    assert "bosc_ame.direct_ddr" in direct.stdout
    assert "bosc_ame.dma.start" not in direct.stdout
    lowered = subprocess.run(
        [str(opt), "-", "--lower-bosc-ame", "-o", "-"],
        input=direct.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lowered.returncode == 0, lowered.stderr
    assert "llvm.call @bosc_ame_dma_start" not in lowered.stdout
    plain = lower(opt, one_tile, report=False)
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout == direct.stdout

    # M is already tiled by the frontend. Across two N subtiles, staging A
    # moves its 4096-byte snapshot once; direct A would read 8192 bytes.
    repeated = write(work, "repeated-a", 64, 128, 64)
    staged = lower(opt, repeated)
    assert staged.returncode == 0, staged.stderr
    route = next(
        line for line in staged.stderr.splitlines() if "plan.route=" in line
    )
    assert 'a = {bank = 1 : i64, bytes = 4096 : i64, level = "spm"' in route
    assert 'b = {bytes = 8192 : i64, level = "ddr"' in route
    assert 'accumulator = {bytes = 32768 : i64, level = "ddr"' in route
    assert 'reason = "legal-minimum-external-traffic"' in route
    assert "chunks = {bytes = 64 : i64, count = 64 : i64" in staged.stderr
    assert "high_water = array<i64: 0, 4096>" in staged.stderr
    assert "bosc_ame.dma.start" in staged.stdout
    tile_loads = [
        line
        for line in staged.stdout.splitlines()
        if "bosc_ame.tile.load_" in line
    ]
    assert any(
        "load_a" in line and "bosc_ame.direct_ddr" not in line
        for line in tile_loads
    )
    assert any(
        "load_b" in line and "bosc_ame.direct_ddr" in line
        for line in tile_loads
    )

    # This legal pre-tiled case needs 135168 bytes for A. Its direct route is
    # address-safe, so capacity causes automatic forwarding rather than a
    # request for a user-selected policy.
    capacity = write(work, "capacity-fallback", 64, 128, 2112)
    fallback = lower(opt, capacity)
    assert fallback.returncode == 0, fallback.stderr
    assert 'reason = "capacity-fallback-safe-direct"' in fallback.stderr
    assert "high_water = array<i64: 0, 0>" in fallback.stderr
    assert "bosc_ame.dma.start" not in fallback.stdout

    # Preserve the old untiled sources as explicit contract boundary cases.
    for name, m, n, k in (
        ("untiled-repeated", 128, 128, 64),
        ("untiled-mixed", 128, 64, 64),
        ("untiled-capacity", 256, 128, 640),
    ):
        rejected = lower(opt, write(work, name, m, n, k))
        assert rejected.returncode != 0
        assert "AME plan requires frontend tiling/padding" in rejected.stderr, (
            rejected.stderr
        )
    disabled = lower(opt, one_tile, spm_enabled=False)
    assert disabled.returncode != 0
    assert "requires enabled target engines" in disabled.stderr, disabled.stderr

    # CHECK: PASS AME input route traffic and capacity choices
    print("PASS AME input route traffic and capacity choices")


if __name__ == "__main__":
    main()
