# RUN: %PYTHON %s %t
"""Fuse private i32 products before selecting direct or SPM accumulator routes."""

import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402

TILE = "memref<64x64xi32>"
PREVIOUS = 0x7FFFFFF0


def source(private=False, swapped=False, gap=False):
    inputs = "%c, %p" if swapped else "%p, %c"
    allocation = (
        f"%c = memref.alloc() alignment = 64 : {TILE}\n"
        f"linalg.fill ins(%initial : i32) outs(%c : {TILE})"
        if private
        else ""
    )
    terminal = (
        f"memref.copy %c, %out : {TILE} to {TILE}\nmemref.dealloc %c : {TILE}"
        if private
        else ""
    )
    before = "%before = arith.addi %zero, %zero : i32" if gap else ""
    after = "%after = arith.addi %before, %zero : i32" if gap else ""
    return f"""
#identity = affine_map<(d0, d1) -> (d0, d1)>
module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @accumulate(
      %a: memref<64x64xi8>,
      %b_storage: memref<4096xi8>,
      {"%out" if private else "%c"}: {TILE}) {{
    %zero = arith.constant 0 : i32
    %initial = arith.constant {PREVIOUS} : i32
    %index = arith.constant 0 : index
    %b = memref.reinterpret_cast %b_storage to offset: [0], sizes: [64, 64], strides: [1, 64]
      : memref<4096xi8> to memref<64x64xi8, strided<[1, 64]>>
    {allocation}
    %p = memref.alloc() alignment = 64 : {TILE}
    linalg.fill ins(%zero : i32) outs(%p : {TILE})
    {before}
    linalg.matmul {{cast = #linalg.type_fn<cast_signed>}}
        ins(%a, %b : memref<64x64xi8>, memref<64x64xi8, strided<[1, 64]>>)
        outs(%p : {TILE})
    {after}
    linalg.generic {{
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel", "parallel"]}}
        ins({inputs} : {TILE}, {TILE}) outs(%c : {TILE}) {{
      ^bb0(%lhs: i32, %rhs: i32, %old: i32):
        %sum = arith.addi %lhs, %rhs : i32
        linalg.yield %sum : i32
    }}
    {terminal}
    memref.dealloc %p : {TILE}
    return
  }}
}}
"""


def task(operation):
    return re.search(r"contract.task = \{([^}]*)\}", operation)[1]


def values(record, field):
    found = re.search(rf"\b{field} = array<i64(?:: ([^>]*))?>", record)
    assert found, (field, record)
    return [int(value) for value in found[1].split(",")] if found[1] else []


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(name, text, *passes):
        path = work / f"{name}.mlir"
        path.write_text(text)
        return subprocess.run(
            [opt, str(path), *passes], text=True, capture_output=True
        )

    def plan(name, text):
        return invoke(
            name,
            text,
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
            "--bosc-ame-plan-verify",
            "--mlir-print-ir-after-failure",
        )

    for private in (False, True):
        for swapped in (False, True):
            name = f"{'spm' if private else 'direct'}_{'swapped' if swapped else 'ordered'}"
            result = plan(name, source(private, swapped, gap=swapped))
            assert result.returncode == 0, (name, result.stderr)
            planned = result.stdout
            assert "linalg.generic" not in planned, (name, planned)
            assert planned.count("memref.alloc()") == int(private), name
            assert planned.count("linalg.fill") == int(private), name
            assert planned.count("linalg.matmul") == 1, name
            assert planned.count("contract.task =") == (3 if private else 1)
            assert (
                f"plan.source_task_count = {5 if private else 3} : i64"
                in planned
            ), name
            mappings = re.search(r"plan.task_map = \[([^]]*)\]", planned)[1]
            eliminated = [
                item
                for item in re.findall(r"\{([^}]*)\}", mappings)
                if 'action = "eliminated"' in item
            ]
            assert len(eliminated) == 2, (name, mappings)
            assert all("new_id" not in item for item in eliminated)
            matmul = next(
                line for line in planned.splitlines() if "linalg.matmul" in line
            )
            record = task(matmul)
            assert len(values(record, "input_values")) == 3
            if private:
                fill = next(
                    line
                    for line in planned.splitlines()
                    if "linalg.fill" in line
                )
                assert (
                    values(record, "input_values")[-1]
                    == values(task(fill), "output_values")[0]
                )
                assert values(record, "predecessors") == [0]
                assert (
                    'level = "spm"'
                    in re.search(r"accumulator = \{([^}]*)\}", matmul)[1]
                )
            else:
                assert values(record, "predecessors") == []
                assert (
                    'level = "ddr"'
                    in re.search(r"accumulator = \{([^}]*)\}", matmul)[1]
                )
            emitted = invoke(name + ".planned", planned, "--bosc-ame-emit")
            assert emitted.returncode == 0, (name, emitted.stderr)
            assert "linalg." not in emitted.stdout
            assert emitted.stdout.count("bosc_ame.tile.mma") == 1
            path = work / f"{name}.emitted.mlir"
            path.write_text(emitted.stdout)
            a = bytearray([128]) * 4096 + bytearray([0xDB]) * 64
            b = bytearray([255]) * 4096 + bytearray([0xBC]) * 64
            output = (
                bytearray(struct.pack("<i", PREVIOUS) * 4096)
                + bytearray([0xCE]) * 64
            )
            saved_a, saved_b = bytes(a), bytes(b)
            expected = (PREVIOUS + 64 * 128 + (1 << 31)) % (1 << 32) - (1 << 31)
            machine = Machine(path, "accumulate")
            machine.launch({"%arg0": a, "%arg1": b, "%arg2": output}, {})
            assert output == (
                struct.pack("<i", expected) * 4096 + bytes([0xCE]) * 64
            ), name
            assert bytes(a) == saved_a and bytes(b) == saved_b, name
            assert bool(machine.dma_log) == private, name

    original = source()
    private_original = source(private=True)
    generic = private_original[
        private_original.index("    linalg.generic") : private_original.index(
            "    memref.copy %c, %out"
        )
    ]
    extra = private_original.replace(
        "    memref.copy %c, %out",
        generic + "    memref.copy %c, %out",
    )
    effect = original.replace(
        "    %p = memref.alloc()",
        "    %scratch = memref.alloca() : memref<i32>\n    %p = memref.alloc()",
    ).replace(
        "    linalg.generic {",
        "    linalg.fill ins(%zero : i32) outs(%scratch : memref<i32>)\n"
        "    linalg.generic {",
    )
    # A view of P makes the add's accumulator overlap the product. The full
    # original chain must survive even though the view itself has no effects.
    alias = (
        original.replace(f"%c: {TILE}", f"%out: {TILE}", 1)
        .replace(
            "    linalg.fill ins(%zero",
            f"    %c = memref.cast %p : {TILE} to {TILE}\n"
            "    linalg.fill ins(%zero",
        )
        .replace(
            "    memref.dealloc %p",
            f"    memref.copy %c, %out : {TILE} to {TILE}\n"
            "    memref.dealloc %p",
        )
    )

    def input_alias(offset):
        return (
            original.replace(
                "%a: memref<64x64xi8>", "%backing: memref<32768xi8>", 1
            )
            .replace(f"%c: {TILE}", f"%out: {TILE}", 1)
            .replace(
                "    %b = memref.reinterpret_cast",
                f"    %shift = arith.constant {offset} : index\n"
                "    %a = memref.view %backing[%index][] : memref<32768xi8> to memref<64x64xi8>\n"
                f"    %c = memref.view %backing[%shift][] : memref<32768xi8> to {TILE}\n"
                "    %b = memref.reinterpret_cast",
            )
        )

    for name, text in (
        ("additional_consumer", extra),
        (
            "nonzero_product",
            original.replace(
                "linalg.fill ins(%zero", "linalg.fill ins(%initial"
            ),
        ),
        (
            "overflow_flags",
            original.replace(
                "arith.addi %lhs, %rhs", "arith.addi %lhs, %rhs overflow<nsw>"
            ).replace(
                f"arith.constant {PREVIOUS} : i32", "arith.constant 7 : i32"
            ),
        ),
        ("product_alias", alias),
        ("input_alias", input_alias(0)),
        ("input_partial_overlap", input_alias(64)),
        ("effect_gap", effect),
    ):
        result = plan(name, text)
        assert result.returncode == 0, (name, result.stderr)
        planned = result.stdout
        assert planned.count("memref.alloc()") == (
            2 if name == "additional_consumer" else 1
        ), (name, planned)
        assert planned.count("linalg.fill") == (
            2 if name in ("additional_consumer", "effect_gap") else 1
        ), (name, planned)
        assert planned.count("linalg.matmul") == 1, (name, planned)
        assert planned.count("linalg.generic") == (
            2 if name == "additional_consumer" else 1
        ), (name, planned)
        assert 'action = "eliminated"' not in planned, (name, planned)
        if name == "overflow_flags":
            assert "overflow<nsw>" in planned
        emitted = invoke(name + ".planned", planned, "--bosc-ame-emit")
        assert emitted.returncode == 0, (name, emitted.stderr)
        assert "bosc_ame.tile.mma" in emitted.stdout
        assert (
            "vector.load" in emitted.stdout and "arith.addi" in emitted.stdout
        )
        path = work / f"{name}.emitted.mlir"
        path.write_text(emitted.stdout)
        a = bytearray([128]) * (32768 if name.startswith("input_") else 4096)
        a += bytearray([0xDB]) * 64
        b = bytearray([255]) * 4096 + bytearray([0xBC]) * 64
        initial = 7 if name == "overflow_flags" else PREVIOUS
        output = (
            bytearray(struct.pack("<i", initial) * 4096)
            + bytearray([0xCE]) * 64
        )
        saved_a, saved_b = bytes(a), bytes(b)
        machine = Machine(path, "accumulate")
        machine.launch({"%arg0": a, "%arg1": b, "%arg2": output}, {})
        increment = 64 * 128
        if name.startswith("input_"):
            expected_a = bytearray(saved_a)
            offset = 64 if name == "input_partial_overlap" else 0
            previous = struct.unpack("<i", bytes([128]) * 4)[0]
            expected_a[offset : offset + 16384] = (
                struct.pack("<i", previous + increment) * 4096
            )
            assert a == expected_a, name
            assert (
                output == struct.pack("<i", initial) * 4096 + bytes([0xCE]) * 64
            )
        else:
            total = initial + increment
            if name == "product_alias":
                total = increment * 2
            elif name == "nonzero_product":
                total += PREVIOUS
            elif name == "additional_consumer":
                total += increment
            expected = (total + (1 << 31)) % (1 << 32) - (1 << 31)
            assert (
                output
                == struct.pack("<i", expected) * 4096 + bytes([0xCE]) * 64
            ), name
            assert bytes(a) == saved_a, name
        assert bytes(b) == saved_b, name

    print(
        "PASS production accumulator fusion: nonzero i32 wrap, direct/SPM routes, swapped inputs, graph rebuild, and rejection boundaries"
    )


if __name__ == "__main__":
    main()
