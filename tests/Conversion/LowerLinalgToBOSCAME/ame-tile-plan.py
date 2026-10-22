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
    "target=nr-fpga bm=128 bn=128 bk=128"
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
    module = work / "ame-tile-plan.mlir"
    module.write_text(MODULE)

    opt = Path(buddy_opt_path())
    plain = lower(opt, module, plan_report=False)
    assert plain.returncode == 0, plain.stderr
    assert not plain.stderr, plain.stderr

    reported = lower(opt, module, plan_report=True)
    assert reported.returncode == 0, reported.stderr
    assert plain.stdout == reported.stdout
    expected = (
        "ame-tile-plan: m=65 n=80 k=160 tile=[64,64,64] "
        "tile-counts=[2,2,3] tail=[1,16,32]"
    )
    assert expected in reported.stderr, reported.stderr

    # CHECK: PASS normalized AME tile plan and report-only IR stability
    print("PASS normalized AME tile plan and report-only IR stability")


if __name__ == "__main__":
    main()
