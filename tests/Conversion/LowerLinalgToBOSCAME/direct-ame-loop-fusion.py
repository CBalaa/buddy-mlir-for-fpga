# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Looped AME accumulator fusion must preserve initialization and C ordering."""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from boscame_check import buddy_opt_path  # noqa: E402

SOURCE = """module attributes {bosc_ame.placement = "logical"} {
  func.func @gemm(%a: memref<64x64xi8>, %b: memref<64x64xi8, strided<[1, 64]>>, %c: memref<64x64xi32>) {
    %zero = memref.alloc() alignment = 64 : memref<64x64xi32>
    %product = memref.alloc() alignment = 64 : memref<64x64xi32>
    %snapshot = memref.alloc() alignment = 64 : memref<64x64xi32>
    %accumulation = memref.alloc() alignment = 64 : memref<64x64xi32>
    %z = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %idx = arith.constant 0 : index
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    linalg.fill ins(%z : i32) outs(%zero : memref<64x64xi32>)
    @ACCUMULATION_INIT@
    scf.for %iv = %lb to %ub step %step {
      memref.copy %zero, %product : memref<64x64xi32> to memref<64x64xi32>
      linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame} ins(%a, %b : memref<64x64xi8>, memref<64x64xi8, strided<[1, 64]>>) outs(%product : memref<64x64xi32>)
      linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%product, %accumulation : memref<64x64xi32>, memref<64x64xi32>) outs(%accumulation : memref<64x64xi32>) {
      ^bb0(%left: i32, %right: i32, %out: i32):
        %sum = arith.addi %left, %right : i32
        linalg.yield %sum : i32
      }
    }
    @BEFORE_C_VIEW@
    %cv = memref.reinterpret_cast %c to offset: [0], sizes: [64,64], strides: [64,1] : memref<64x64xi32> to memref<64x64xi32, strided<[64,1]>>
    memref.copy %cv, %snapshot {bosc_ame.memory = {kind = "load", logical_shape = array<i64: 64, 64>, valid_shape = array<i64: 64, 64>, masked = false}} : memref<64x64xi32, strided<[64,1]>> to memref<64x64xi32>
    @AFTER_SNAPSHOT@
    linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%accumulation, %snapshot : memref<64x64xi32>, memref<64x64xi32>) outs(%accumulation : memref<64x64xi32>) {
    ^bb0(%left: i32, %right: i32, %out: i32):
      %sum = arith.addi %left, %right : i32
      linalg.yield %sum : i32
    }
    memref.copy %accumulation, %cv {bosc_ame.memory = {destination = "external_ddr", kind = "store", logical_shape = array<i64: 64, 64>, valid_shape = array<i64: 64, 64>, masked = false}} : memref<64x64xi32> to memref<64x64xi32, strided<[64,1]>>
    return
  }
}
"""


def lower(
    work,
    name,
    *,
    accumulation_init="memref.copy %zero, %accumulation : memref<64x64xi32> to memref<64x64xi32>",
    before_c_view="",
    after_snapshot="",
):
    source = SOURCE.replace("@ACCUMULATION_INIT@", accumulation_init)
    source = source.replace("@BEFORE_C_VIEW@", before_c_view)
    source = source.replace("@AFTER_SNAPSHOT@", after_snapshot)
    path = work / f"{name}.mlir"
    path.write_text(source)
    return subprocess.run(
        [
            buddy_opt_path(),
            str(path),
            "--linalg-to-boscame=enable-spm-dma=true enable-ame=true "
            "dma-abi=args_aligned64_nonalias target=nr-fpga",
        ],
        capture_output=True,
        text=True,
    )


def check_fusion(result, expected):
    assert result.returncode == 0, result.stderr
    direct_accumulator_ops = re.findall(
        r"bosc_ame\.tile\.(?:load|store)_acc[^\n]*bosc_ame\.direct_ddr",
        result.stdout,
    )
    assert len(direct_accumulator_ops) == (2 if expected else 0), result.stdout


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    # Zero initialization permits the direct accumulator rewrite.
    check_fusion(lower(work, "canonical"), expected=True)

    # The loop-carried accumulation contributes its initial value to the final
    # result, so it cannot be replaced by a direct C accumulator.
    nonzero = lower(
        work,
        "nonzero_accumulator",
        accumulation_init="""%initial = memref.alloc() alignment = 64 : memref<64x64xi32>
    %seven = arith.constant 7 : i32
    linalg.fill ins(%seven : i32) outs(%initial : memref<64x64xi32>)
    memref.copy %initial, %accumulation : memref<64x64xi32> to memref<64x64xi32>""",
    )
    check_fusion(nonzero, expected=False)

    # The final store must overwrite a write occurring after the C snapshot.
    # Moving the direct accumulator store above that write changes the result.
    side_effect = lower(
        work,
        "side_effect_after_snapshot",
        after_snapshot="memref.store %one, %cv[%idx, %idx] : memref<64x64xi32, strided<[64,1]>>",
    )
    check_fusion(side_effect, expected=False)

    # Successor blocks share the function parent but have no common block
    # ordering. Conservatively retain the unfused CFG without asserting.
    successor = lower(
        work,
        "snapshot_in_successor",
        before_c_view="cf.br ^finish\n  ^finish:",
    )
    check_fusion(successor, expected=False)

    # CHECK: PASS looped direct AME fusion: initialization and C ordering
    print("PASS looped direct AME fusion: initialization and C ordering")


if __name__ == "__main__":
    main()
