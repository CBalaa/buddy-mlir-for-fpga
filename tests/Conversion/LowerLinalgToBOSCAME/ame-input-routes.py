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
    options = (
        ("enable-spm-dma=true " if spm_enabled else "")
        + "enable-ame=true dma-abi=args_aligned64_nonalias "
        "target=nr-fpga bm=64 bn=64 bk=64"
    )
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
    spm = lower(opt, one_tile)
    assert spm.returncode == 0, spm.stderr
    assert "direct-A=direct-ddr (proven)" in spm.stderr, spm.stderr
    assert "direct-B=direct-ddr (proven)" in spm.stderr, spm.stderr
    assert "direct-C=direct-ddr (proven)" in spm.stderr, spm.stderr
    assert "A=direct-byte-tie direct-bytes=4096 staged-bytes=4096" in spm.stderr
    assert "B=direct-byte-tie direct-bytes=4096 staged-bytes=4096" in spm.stderr
    assert (
        "bosc_ame.tile.load_a" in spm.stdout
        and "bosc_ame.direct_ddr" in spm.stdout
    )
    assert (
        "bosc_ame.tile.load_b" in spm.stdout
        and "bosc_ame.direct_ddr" in spm.stdout
    )
    assert "bosc_ame.dma.start" not in spm.stdout
    lowered = subprocess.run(
        [str(opt), "-", "--lower-bosc-ame", "-o", "-"],
        input=spm.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lowered.returncode == 0, lowered.stderr
    assert "llvm.call @bosc_ame_dma_start" not in lowered.stdout
    plain = lower(opt, one_tile, report=False)
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout == spm.stdout

    repeated = write(work, "repeated-reads", 128, 128, 64)
    staged = lower(opt, repeated)
    assert staged.returncode == 0, staged.stderr
    assert "direct-A=spm-dma (planned)" in staged.stderr, staged.stderr
    assert "direct-B=spm-dma (planned)" in staged.stderr, staged.stderr
    assert (
        "A=staged-lower-read-bytes direct-bytes=8192 staged-bytes=8192"
        not in staged.stderr
    )
    assert (
        "A=staged-lower-read-bytes direct-bytes=16384 staged-bytes=8192"
        in staged.stderr
    )
    assert (
        "B=staged-lower-read-bytes direct-bytes=16384 staged-bytes=8192"
        in staged.stderr
    )

    mixed = write(work, "mixed-input-routes", 128, 64, 64)
    mixed_routes = lower(opt, mixed)
    assert mixed_routes.returncode == 0, mixed_routes.stderr
    assert "direct-A=direct-ddr (proven)" in mixed_routes.stderr
    assert "direct-B=spm-dma (planned)" in mixed_routes.stderr
    assert "direct-C=direct-ddr (proven)" in mixed_routes.stderr
    assert "A=direct-byte-tie direct-bytes=8192 staged-bytes=8192" in (
        mixed_routes.stderr
    )
    assert "B=staged-lower-read-bytes direct-bytes=8192 staged-bytes=4096" in (
        mixed_routes.stderr
    )
    tile_loads = [
        line
        for line in mixed_routes.stdout.splitlines()
        if "bosc_ame.tile.load_" in line
    ]
    assert any(
        "bosc_ame.tile.load_a" in line and "bosc_ame.direct_ddr" in line
        for line in tile_loads
    )
    assert any("bosc_ame.tile.load_b" in line for line in tile_loads)
    assert not any(
        "bosc_ame.tile.load_b" in line and "bosc_ame.direct_ddr" in line
        for line in tile_loads
    )

    direct_only = lower(opt, one_tile, spm_enabled=False)
    assert direct_only.returncode == 0, direct_only.stderr
    assert "A=direct-spm-disabled" in direct_only.stderr, direct_only.stderr
    assert "B=direct-spm-disabled" in direct_only.stderr, direct_only.stderr
    assert "bosc_ame.dma.start" not in direct_only.stdout

    capacity = write(work, "capacity-fallback", 256, 128, 640)
    fallback = lower(opt, capacity)
    assert fallback.returncode == 0, fallback.stderr
    assert "A=direct-capacity-fallback" in fallback.stderr, fallback.stderr
    assert "B=staged-lower-read-bytes" in fallback.stderr, fallback.stderr
    assert "direct-C=direct-ddr (proven)" in fallback.stderr, fallback.stderr

    # CHECK: PASS AME input route traffic and capacity choices
    print("PASS AME input route traffic and capacity choices")


if __name__ == "__main__":
    main()
