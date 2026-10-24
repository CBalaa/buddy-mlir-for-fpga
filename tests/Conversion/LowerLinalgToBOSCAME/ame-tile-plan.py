# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Check normalized AME tile facts and report-only output stability."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from boscame_check import buddy_opt_path

MODULE = """module {
  func.func @gemm() {
    %a_storage = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<10432xi8, #bosc_ame.spm<0>>
    %a = memref.reinterpret_cast %a_storage to offset: [0], sizes: [65, 160],
      strides: [160, 1] : memref<10432xi8, #bosc_ame.spm<0>> to
      memref<65x160xi8, strided<[160, 1]>, #bosc_ame.spm<0>>
    %b_storage = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<12800xi8, #bosc_ame.spm<1>>
    %b = memref.reinterpret_cast %b_storage to offset: [0], sizes: [160, 80],
      strides: [1, 160] : memref<12800xi8, #bosc_ame.spm<1>> to
      memref<160x80xi8, strided<[1, 160]>, #bosc_ame.spm<1>>
    %c_storage = bosc_ame.spm.view {bank = 0 : i32, offset = 10496 : i64}
      : memref<5216xi32, #bosc_ame.spm<0>>
    %c = memref.reinterpret_cast %c_storage to offset: [0], sizes: [65, 80],
      strides: [80, 1] : memref<5216xi32, #bosc_ame.spm<0>> to
      memref<65x80xi32, strided<[80, 1]>, #bosc_ame.spm<0>>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<65x160xi8, strided<[160, 1]>, #bosc_ame.spm<0>>,
                   memref<160x80xi8, strided<[1, 160]>, #bosc_ame.spm<1>>)
      outs(%c : memref<65x80xi32, strided<[80, 1]>, #bosc_ame.spm<0>>)
    return
  }
}
"""

OPTIONS = (
    "enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias "
    "target=nr-fpga"
)


def lower(opt, module, plan_report):
    options = OPTIONS + (" plan-report=true" if plan_report else "")
    return subprocess.run(
        [str(opt), str(module), f"--linalg-to-boscame={options}", "-o", "-"],
        capture_output=True,
        text=True,
        check=False,
    )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    original = work / "ame-tile-plan-original.mlir"
    original.write_text(MODULE)
    opt = Path(buddy_opt_path())
    rejected = lower(opt, original, plan_report=False)
    assert rejected.returncode != 0
    assert "no initialized content version" in rejected.stderr, rejected.stderr

    # A v1 frontend must tile M and pad K before the backend boundary. Fill
    # every physical input explicitly; an SPM descriptor alone is not content.
    supported = MODULE
    for old, new in (
        ("65x160", "64x192"),
        ("160x80", "192x128"),
        ("65x80", "64x128"),
        ("[65, 160]", "[64, 192]"),
        ("[160, 80]", "[192, 128]"),
        ("[65, 80]", "[64, 128]"),
        ("[160, 1]", "[192, 1]"),
        ("[1, 160]", "[1, 192]"),
        ("[80, 1]", "[128, 1]"),
    ):
        supported = supported.replace(old, new)
    supported = supported.replace("10432xi8", "12288xi8")
    supported = supported.replace("12800xi8", "24576xi8")
    supported = supported.replace("5216xi32", "8192xi32")
    supported = supported.replace("offset = 10496", "offset = 12288")
    marker = "    linalg.matmul"
    initialization = """    %one = arith.constant 1 : i8
    %negative = arith.constant -1 : i8
    %seven = arith.constant 7 : i32
    linalg.fill ins(%one : i8) outs(%a : memref<64x192xi8, strided<[192, 1]>, #bosc_ame.spm<0>>)
    linalg.fill ins(%negative : i8) outs(%b : memref<192x128xi8, strided<[1, 192]>, #bosc_ame.spm<1>>)
    linalg.fill ins(%seven : i32) outs(%c : memref<64x128xi32, strided<[128, 1]>, #bosc_ame.spm<0>>)
"""
    supported = supported.replace(marker, initialization + marker)
    module = work / "ame-tile-plan.mlir"
    module.write_text(supported)
    plain = lower(opt, module, plan_report=False)
    assert plain.returncode == 0, plain.stderr
    assert not plain.stderr, plain.stderr
    reported = lower(opt, module, plan_report=True)
    assert reported.returncode == 0, reported.stderr
    assert plain.stdout == reported.stdout
    assert "subtiles = {k = 3 : i64, n = 2 : i64}" in reported.stderr
    assert (
        "tile = {k = 64 : i64, m = 64 : i64, n = 64 : i64}" in reported.stderr
    )
    assert (
        'a = {bank = 0 : i64, bytes = 12288 : i64, level = "spm"'
        in reported.stderr
    )
    assert (
        'b = {bank = 1 : i64, bytes = 24576 : i64, level = "spm"'
        in reported.stderr
    )
    assert (
        'accumulator = {bank = 0 : i64, bytes = 32768 : i64, level = "spm", offset = 12288'
        in reported.stderr
    )

    # CHECK: PASS normalized AME tile plan and report-only IR stability
    print("PASS normalized AME tile plan and report-only IR stability")


if __name__ == "__main__":
    main()
