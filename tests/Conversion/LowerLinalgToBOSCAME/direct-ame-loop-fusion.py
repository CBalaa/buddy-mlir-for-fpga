# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Looped AME accumulator fusion must preserve initialization and C ordering."""

import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from boscame_buffer_sim import Machine  # noqa: E402
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


def normalize_copies(source, seed=0):
    source = source.replace(
        "    %zero = memref.alloc() alignment = 64 : memref<64x64xi32>\n", ""
    )
    source = source.replace(
        "    linalg.fill ins(%z : i32) outs(%zero : memref<64x64xi32>)\n", ""
    )
    source = source.replace(
        "memref.copy %zero, %product : memref<64x64xi32> to memref<64x64xi32>",
        "linalg.fill ins(%z : i32) outs(%product : memref<64x64xi32>)",
    )
    source = source.replace(
        "@ACCUMULATION_INIT@",
        f"%seed = arith.constant {seed} : i32\n    linalg.fill ins(%seed : i32) outs(%accumulation : memref<64x64xi32>)",
    )
    return source.replace("@BEFORE_C_VIEW@", "").replace("@AFTER_SNAPSHOT@", "")


def compile_source(work, name, source):
    path = work / f"{name}.mlir"
    path.write_text(source)
    return subprocess.run(
        [
            str(buddy_opt_path()),
            str(path),
            "--linalg-to-boscame=dma-abi=args_aligned64_nonalias",
        ],
        capture_output=True,
        text=True,
    )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    # Standard normalization now proves private zero copies as well as fills.
    # Retain the original source and run the same ordering/numerical checks.
    raw = lower(work, "original-private-copies")
    assert raw.returncode == 0, raw.stderr

    for initial in (None, 0, 7):
        seed = initial or 0
        name = (
            "original-private-copies"
            if initial is None
            else f"private-loop-{seed}"
        )
        result = (
            raw
            if initial is None
            else compile_source(work, name, normalize_copies(SOURCE, seed))
        )
        assert result.returncode == 0, result.stderr
        text = result.stdout
        acc_ops = re.findall(r"bosc_ame\.tile\.(?:load|store)_acc[^\n]*", text)
        assert len(acc_ops) == 2, text
        assert all(
            "#bosc_ame.spm" in op and "bosc_ame.direct_ddr" not in op
            for op in acc_ops
        ), acc_ops
        assert (
            "contract.loop_carried_slots = [{backedge_producer_compute_ids"
            in text
        )
        assert "bosc_ame.tile.mma" in text
        # The local P/add pair fused into the private loop accumulator. Only
        # the final add of the later C snapshot remains as RVV arithmetic.
        assert len(re.findall(r"arith\.addi [^\n]*: vector<16xi32>", text)) == 1
        assert 'action = "eliminated"' in text
        path = work / f"{name}.boscame.mlir"
        path.write_text(text)
        a_values = [(-128 if i % 2 else 127) for i in range(4096)]
        b_values = [(-127 if i % 3 else 126) for i in range(4096)]
        c_values = [(0x7FFFFFF0 + i * 17) & 0xFFFFFFFF for i in range(4096)]
        expected = []
        for row in range(64):
            for col in range(64):
                dot = sum(
                    a_values[row * 64 + k] * b_values[col * 64 + k]
                    for k in range(64)
                )
                expected.append(
                    (c_values[row * 64 + col] + seed + 2 * dot) & 0xFFFFFFFF
                )
        guard = b"\xd3" * 64
        a_original = bytes(value & 255 for value in a_values) + guard
        b_original = bytes(value & 255 for value in b_values) + guard
        a, b = bytearray(a_original), bytearray(b_original)
        c = bytearray(struct.pack("<4096I", *c_values) + guard)
        machine = Machine(path, "gemm")
        machine.spm[0][:] = b"\x6b" * len(machine.spm[0])
        machine.spm[1][:] = b"\x91" * len(machine.spm[1])
        machine.launch({"%arg0": a, "%arg1": b, "%arg2": c})
        assert list(struct.unpack("<4096I", c[:16384])) == expected
        assert a == a_original and b == b_original and c[16384:] == guard
        last_input = max(
            i
            for i, access in enumerate(machine.accesses)
            if access[0] in ("%arg0", "%arg1")
        )
        first_c = next(
            i
            for i, access in enumerate(machine.accesses)
            if access[0] == "%arg2"
        )
        assert last_input < first_c, "C read moved across the K loop"

    # Preserve the unsafe late scalar write after a supported snapshot. It
    # cannot obtain an external scalar-store recipe or silently move C loads.
    effect = normalize_copies(SOURCE)
    # Place the write immediately after the actual C load, as in the legacy case.
    copy_line = next(
        line
        for line in effect.splitlines()
        if "memref.copy %cv, %snapshot" in line
    )
    effect = effect.replace(
        copy_line,
        copy_line
        + "\n    memref.store %one, %cv[%idx, %idx] : memref<64x64xi32, strided<[64,1]>>",
    )
    rejected = compile_source(work, "late-write", effect)
    assert rejected.returncode != 0
    assert (
        "no independent external or SPM scalar-store recipe" in rejected.stderr
    ), rejected.stderr
    # CHECK: PASS looped AME fusion: private accumulation, nonzero init, late C and exact wraparound
    print(
        "PASS looped AME fusion: private accumulation, nonzero init, late C and exact wraparound"
    )


if __name__ == "__main__":
    main()
