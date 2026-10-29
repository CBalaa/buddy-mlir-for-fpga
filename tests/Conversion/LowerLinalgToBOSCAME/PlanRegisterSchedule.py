# RUN: %PYTHON %s %t
"""Default-V2 complete-K ACC schedules preserve exact integer DDR data."""

import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def source(
    rows,
    columns,
    pitch,
    trips,
    valid_rows=None,
    initial_zero=False,
    post_add=False,
    c_pitch=None,
):
    """A source K loop carries a private buffer, never AME SSA values."""
    valid_rows = valid_rows or rows
    c_pitch = c_pitch or pitch
    tail = valid_rows < rows
    a_type = f"memref<{rows}x128xi8>"
    b_type = f"memref<128x{columns}xi8, strided<[1, 128]>>"
    c_type = f"memref<{rows}x{columns}xi32>"
    a_ext = f"memref<{rows}x128xi8, strided<[{pitch}, 1], offset: ?>>"
    b_ext = f"memref<128x{columns}xi8, strided<[1, {pitch}], offset: ?>>"
    c_ext = f"memref<{rows}x{columns}xi32, strided<[{c_pitch}, 1]>>"
    c_valid = f"memref<{valid_rows}x{columns}xi32, strided<[{c_pitch}, 1]>>"
    c_private_valid = (
        f"memref<{valid_rows}x{columns}xi32, strided<[{columns}, 1]>>"
    )
    c_fill = (
        f"linalg.fill ins(%zero32 : i32) outs(%c : {c_type})"
        if tail or initial_zero or post_add
        else ""
    )
    c_views = (
        f"%cv = memref.subview %c[0, 0] [{valid_rows}, {columns}] [1, 1] : {c_type} to {c_private_valid}\n"
        f"%out = memref.subview %out_full[0, 0] [{valid_rows}, {columns}] [1, 1] : {c_ext} to {c_valid}"
        if tail
        else ""
    )
    c_local, c_target = ("%cv", "%out") if tail else ("%c", "%out_full")
    c_local_type, c_target_type = (
        (c_private_valid, c_valid) if tail else (c_type, c_ext)
    )
    c_load = (
        ""
        if initial_zero or post_add
        else f"memref.copy {c_target}, {c_local} : {c_target_type} to {c_local_type}"
    )
    a_fill = f"linalg.fill ins(%zero8 : i8) outs(%a : {a_type})" if tail else ""
    a_valid = f"memref<{valid_rows}x128xi8, strided<[{pitch}, 1], offset: ?>>"
    a_private_valid = f"memref<{valid_rows}x128xi8, strided<[128, 1]>>"
    a_views = (
        f"%av = memref.subview %a[0, 0] [{valid_rows}, 128] [1, 1] : {a_type} to {a_private_valid}\n"
        f"%input = memref.subview %a_full[0, 0] [{valid_rows}, 128] [1, 1] : {a_ext} to {a_valid}"
        if tail
        else ""
    )
    a_copy = (
        f"memref.copy %input, %av : {a_valid} to {a_private_valid}"
        if tail
        else f"memref.copy %a_full, %a : {a_ext} to {a_type}"
    )
    epilogue = ""
    if post_add:
        old_fill = (
            f"linalg.fill ins(%zero32 : i32) outs(%old : {c_type})"
            if tail
            else ""
        )
        old_view = (
            f"%oldv = memref.subview %old[0, 0] [{valid_rows}, {columns}] [1, 1] : {c_type} to {c_private_valid}"
            if tail
            else ""
        )
        old_local = "%oldv" if tail else "%old"
        epilogue = f"""
      %old = memref.alloc() alignment = 64 : {c_type}
      {old_fill}
      {old_view}
      memref.copy {c_target}, {old_local} : {c_target_type} to {c_local_type}
      linalg.generic {{
        indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
        iterator_types = ["parallel", "parallel"]
      }} ins(%c, %old : {c_type}, {c_type}) outs(%c : {c_type}) {{
      ^bb0(%lhs: i32, %rhs: i32, %old_out: i32):
        %sum = arith.addi %lhs, %rhs : i32
        linalg.yield %sum : i32
      }}
      memref.dealloc %old : {c_type}
        """
    compute = f"""
      %a_full = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [{rows}, 128], strides: [{pitch}, 1]
        : memref<*xi8> to {a_ext}
      %a = memref.alloc() alignment = 64 : {a_type}
      {a_fill}
      {a_views}
      {a_copy}
      %b_full = memref.reinterpret_cast %arg1 to offset: [%offset], sizes: [128, {columns}], strides: [1, {pitch}]
        : memref<*xi8> to {b_ext}
      %b_storage = memref.alloc() alignment = 64 : memref<{128 * columns}xi8>
      %b = memref.reinterpret_cast %b_storage to offset: [0], sizes: [128, {columns}], strides: [1, 128]
        : memref<{128 * columns}xi8> to {b_type}
      memref.copy %b_full, %b : {b_ext} to {b_type}
      linalg.matmul {{cast = #linalg.type_fn<cast_signed>}}
        ins(%a, %b : {a_type}, {b_type}) outs(%c : {c_type})
      memref.dealloc %a : {a_type}
      memref.dealloc %b_storage : memref<{128 * columns}xi8>
    """
    if trips > 1:
        compute = f"""
      scf.for %ki = %zero32 to %trips step %one32 : i32 {{
        %koff = arith.muli %ki, %block32 : i32
        %offset = arith.index_cast %koff : i32 to index
        {compute}
      }}
    """
    else:
        compute = "%offset = arith.constant 0 : index\n" + compute
    return f"""
module attributes {{contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}}} {{
  func.func @matrix(%arg0: memref<*xi8>, %arg1: memref<*xi8>, %arg2: memref<*xi32>) {{
    %zero32 = arith.constant 0 : i32
    %zero8 = arith.constant 0 : i8
    %one32 = arith.constant 1 : i32
    %block32 = arith.constant 128 : i32
    %trips = arith.constant {trips} : i32
    %out_full = memref.reinterpret_cast %arg2 to offset: [0], sizes: [{rows}, {columns}], strides: [{c_pitch}, 1]
      : memref<*xi32> to {c_ext}
    %c = memref.alloc() alignment = 64 : {c_type}
    {c_fill}
    {c_views}
    {c_load}
    {compute}
    {epilogue}
    memref.copy {c_local}, {c_target} : {c_local_type} to {c_target_type}
    memref.dealloc %c : {c_type}
    return
  }}
}}
"""


def signed(value):
    return value if value < 128 else value - 256


def wrapped(value):
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


DIRECT_ARGUMENT_LOOP = """
module attributes {contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}} {
  func.func @matrix(%a: memref<4x64xi8>, %b: memref<64x64xi8, strided<[1, 64]>>, %out: memref<4x64xi32>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %initial = arith.constant 0 : i32
    %c = memref.alloc() alignment = 64 : memref<4x64xi32>
    linalg.fill ins(%initial : i32) outs(%c : memref<4x64xi32>)
    scf.for %i = %zero to %three step %one {
      linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%a, %b : memref<4x64xi8>, memref<64x64xi8, strided<[1, 64]>>)
        outs(%c : memref<4x64xi32>)
    }
    memref.copy %c, %out : memref<4x64xi32> to memref<4x64xi32>
    memref.dealloc %c : memref<4x64xi32>
    return
  }
}
"""


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
        result = invoke(
            name,
            text,
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
            "--bosc-ame-plan-verify",
        )
        assert result.returncode == 0, (name, result.stderr)
        return result.stdout

    def execute_alias_fallback(name, emitted, columns, valid):
        """Execute with A and C sharing storage, twice with dirty workspaces."""
        path = work / f"{name}.emitted.mlir"
        path.write_text(emitted)
        pitch, k = 768, 768
        shared = bytearray().join(
            struct.pack("<i", wrapped(0x7FFFFF00 + i))
            for i in range(valid * pitch)
        )
        shared.extend(bytes([0xF3]) * 64)
        b = bytearray((i * 17 + 211) % 256 for i in range(columns * pitch))
        b.extend(bytes([0xE2]) * 64)
        saved_b = bytes(b)
        machine = Machine(path, "matrix")
        for poison in (0xA5, 0x6B):
            before = bytes(shared)
            expected = bytearray(before)
            for row in range(valid):
                for column in range(columns):
                    old = struct.unpack_from(
                        "<i", before, 4 * (row * pitch + column)
                    )[0]
                    product = sum(
                        signed(before[row * pitch + q])
                        * signed(b[column * pitch + q])
                        for q in range(k)
                    )
                    struct.pack_into(
                        "<i",
                        expected,
                        4 * (row * pitch + column),
                        wrapped(old + product),
                    )
            machine.private_poison = poison
            machine.launch({"%arg0": shared, "%arg1": b, "%arg2": shared}, {})
            # Every byte outside the declared C output, including aliased A
            # bytes, C padding and guards, remains as in the source execution.
            assert shared == expected, name
            assert bytes(b) == saved_b, name
            assert machine.private_top == 0, name
        assert [entry[0] for entry in machine.frame_log] == [
            "acquire",
            "release",
            "acquire",
            "release",
        ], name

    cases = [
        ("single", 64, 64, 128, 1, 64, False, False, 128),
        ("reuse", 32, 128, 768, 6, 32, False, False, 768),
        ("tail", 32, 128, 768, 6, 4, False, False, 768),
        ("zero", 1, 128, 128, 1, 1, True, False, 128),
        ("post_add", 32, 64, 768, 6, 32, False, True, 768),
        ("post_add_tail4", 32, 64, 768, 6, 4, False, True, 768),
        ("post_add_tail5_k24", 32, 64, 3072, 24, 5, False, True, 768),
        ("classifier", 32, 64, 768, 6, 1, False, True, 1000),
        ("classifier_n40", 32, 40, 768, 6, 1, False, True, 1000),
        ("overwrite_classifier", 32, 64, 768, 6, 1, True, False, 1000),
        ("overwrite_classifier_n40", 32, 40, 768, 6, 1, True, False, 1000),
    ]
    saved_plans = {}
    for (
        name,
        rows,
        columns,
        pitch,
        trips,
        valid,
        initial_zero,
        post_add,
        c_pitch,
    ) in cases:
        planned = plan(
            name,
            source(
                rows,
                columns,
                pitch,
                trips,
                valid,
                initial_zero,
                post_add,
                c_pitch,
            ),
        )
        saved_plans[name] = planned
        assert 'kind = "integer-ame-register"' in planned, (name, planned)
        assert f"m = {valid} : i64" in planned, name
        assert "peak_tr = 2 : i64" in planned, name
        tiles = (columns + 63) // 64
        assert f"peak_acc = {tiles} : i64" in planned, name
        assert ('epilogue = "post-loop-modulo-i32-add"' in planned) == post_add
        if post_add:
            assert 'role = "c-post-add"' in planned
            assert 'role = "product-zero"' in planned
        assert 'level = "spm"' not in planned, name
        assert "high_water = array<i64: 0, 0>" in planned, name
        emitted = invoke(name + ".planned", planned, "--bosc-ame-emit")
        assert emitted.returncode == 0, (name, emitted.stderr)
        text = emitted.stdout
        assert "linalg." not in text and "memref.alloc" not in text, (
            name,
            text,
        )
        assert (
            "bosc_ame.dma.start" not in text and "bosc_ame.spm.view" not in text
        )
        assert "bosc_ame.register_tasks" in text
        assert "bosc_ame.register_task_map" in text
        assert "contract.task =" not in text
        assert text.count("bosc_ame.tile.load_a ") == 2, name
        assert text.count("bosc_ame.tile.load_b ") == 2 * tiles, name
        assert text.count("bosc_ame.tile.mma ") == 2 * tiles, name
        assert text.count("bosc_ame.tile.store_acc ") == tiles, name
        assert text.count("bosc_ame.tile.load_acc ") == (
            0 if initial_zero else tiles
        ), name
        if trips > 1:
            assert re.search(
                r"scf.for .*iter_args\([^\n]*\) -> \(vector<", text
            ), text
        if valid < rows and valid == 4:
            assert "vector<4x64xi8>" in text and "vector<4x64xi32>" in text
            assert "vector<32x64" not in text

        path = work / f"{name}.emitted.mlir"
        path.write_text(text)
        k = trips * 128
        a = bytearray((i * 37 + 13) % 256 for i in range(valid * pitch))
        b = bytearray((i * 17 + 211) % 256 for i in range(columns * pitch))
        a.extend(bytes([0xD1]) * 64)
        b.extend(bytes([0xE2]) * 64)
        c = bytearray().join(
            struct.pack("<i", wrapped(0x7FFFFF00 + i))
            for i in range(valid * c_pitch)
        )
        c.extend(bytes([0xF3]) * 64)
        saved_a, saved_b = bytes(a), bytes(b)
        expected = bytearray(c)
        for row in range(valid):
            for column in range(columns):
                old = (
                    0
                    if initial_zero
                    else struct.unpack_from(
                        "<i", c, 4 * (row * c_pitch + column)
                    )[0]
                )
                result = old + sum(
                    signed(a[row * pitch + q]) * signed(b[column * pitch + q])
                    for q in range(k)
                )
                struct.pack_into(
                    "<i",
                    expected,
                    4 * (row * c_pitch + column),
                    wrapped(result),
                )
        machine = Machine(path, "matrix")
        machine.launch({"%arg0": a, "%arg1": b, "%arg2": c}, {})
        assert c == expected, name
        assert bytes(a) == saved_a and bytes(b) == saved_b, name
        assert not machine.dma_log, name
        traffic = {}
        for argument, start, end, kind in machine.accesses:
            traffic[argument, kind] = (
                traffic.get((argument, kind), 0) + end - start
            )
        assert traffic.get(("%arg0", "read"), 0) == valid * k, (name, traffic)
        assert traffic.get(("%arg1", "read"), 0) == columns * k, (name, traffic)
        assert traffic.get(("%arg2", "read"), 0) == (
            0 if initial_zero else valid * columns * 4
        ), (name, traffic)
        assert traffic.get(("%arg2", "write"), 0) == valid * columns * 4, (
            name,
            traffic,
        )

    # A recorded register count is part of the proof, never an emitter hint.
    broken = saved_plans["reuse"].replace(
        "peak_acc = 2 : i64", "peak_acc = 1 : i64", 1
    )
    rejected = invoke("corrupt_count", broken, "--bosc-ame-emit")
    assert rejected.returncode != 0 and "register schedule" in rejected.stderr

    broken, removed = re.subn(
        r",?\s*plan\.register_virtual(?:\s*=\s*\{[^}]*\})?",
        "",
        saved_plans["tail"],
        count=1,
    )
    assert removed == 1 and broken.count("plan.register_virtual") == (
        saved_plans["tail"].count("plan.register_virtual") - 1
    ), "corruption must actually remove one virtual-root ownership marker"
    rejected = invoke("corrupt_virtual", broken, "--bosc-ame-plan-verify")
    assert rejected.returncode != 0 and "register" in rejected.stderr
    assert (
        "expected" not in rejected.stderr
        and "invalid register" in rejected.stderr
    ), rejected.stderr

    # A missing noalias guarantee must keep the memory snapshots available.
    uncertain = source(32, 128, 768, 6).replace(
        "args_disjoint = true", "args_disjoint = false"
    )
    fallback = plan("may_alias", uncertain)
    assert "plan.register_schedule" not in fallback
    emitted = invoke("may_alias.planned", fallback, "--bosc-ame-emit")
    assert emitted.returncode == 0, emitted.stderr
    execute_alias_fallback("may_alias", emitted.stdout, 128, 32)

    # The one-row descriptor normalization is justified by its actual access
    # set. An unaligned physical pitch used by two rows still cannot match.
    rejected_pitch = plan(
        "two_row_unaligned", source(32, 64, 768, 6, 2, True, False, 1000)
    )
    assert "plan.register_schedule" not in rejected_pitch

    # Refuse transformations that would hide an intermediate observation,
    # change a nonzero product initialization, or move a source mutation.
    ordinary = source(32, 64, 768, 6, 5, False, True)
    negatives = {
        "post_add_nsw": ordinary.replace(
            "arith.addi %lhs, %rhs", "arith.addi %lhs, %rhs overflow<nsw>"
        ),
        "post_add_nonzero": ordinary.replace(
            "linalg.fill ins(%zero32 : i32) outs(%c",
            "linalg.fill ins(%one32 : i32) outs(%c",
            1,
        ),
        "post_add_observer": ordinary.replace(
            "      %old = memref.alloc()",
            "      %observe_index = arith.constant 0 : index\n"
            "      %observed = memref.load %c[%observe_index, %observe_index] : memref<32x64xi32>\n"
            "      %old = memref.alloc()",
        ),
        "post_add_source_write": ordinary.replace(
            "      linalg.matmul {cast",
            "      memref.copy %av, %input : memref<5x128xi8, strided<[128, 1]>> to memref<5x128xi8, strided<[768, 1], offset: ?>>\n"
            "      linalg.matmul {cast",
        ),
        "post_add_old_c_write": ordinary.replace(
            "      %old = memref.alloc()",
            "      memref.copy %cv, %out : memref<5x64xi32, strided<[64, 1]>> to memref<5x64xi32, strided<[768, 1]>>\n"
            "      %old = memref.alloc()",
        ),
        "post_add_may_alias": ordinary.replace(
            "args_disjoint = true", "args_disjoint = false"
        ),
        "post_add_unknown_call": ordinary.replace(
            "  func.func @matrix",
            "  func.func private @observe()\n  func.func @matrix",
        ).replace(
            "      %old = memref.alloc()",
            "      func.call @observe() : () -> ()\n"
            "      %old = memref.alloc()",
        ),
        "post_add_partial_fill": ordinary.replace(
            "linalg.fill ins(%zero32 : i32) outs(%c : memref<32x64xi32>)",
            "%partial = memref.subview %c[0, 0] [1, 64] [1, 1] : memref<32x64xi32> to memref<1x64xi32, strided<[64, 1]>>\n"
            "    linalg.fill ins(%zero32 : i32) outs(%partial : memref<1x64xi32, strided<[64, 1]>>)",
            1,
        ),
    }
    for name, text in negatives.items():
        if name in ("post_add_partial_fill", "post_add_old_c_write"):
            rejected = invoke(
                name,
                text,
                "--kernel-contract-annotate",
                "--bosc-ame-plan",
                "--bosc-ame-plan-verify",
            )
            assert rejected.returncode != 0
            diagnostic = (
                "initialized content version"
                if name == "post_add_partial_fill"
                else "reading an external region after its terminal write"
            )
            assert diagnostic in rejected.stderr
            continue
        result = plan(name, text)
        assert "plan.register_schedule" not in result, name
        if name == "post_add_may_alias":
            emitted = invoke(name + ".planned", result, "--bosc-ame-emit")
            assert emitted.returncode == 0, emitted.stderr
            execute_alias_fallback(name, emitted.stdout, 64, 5)

    broken = saved_plans["post_add"].replace(
        'epilogue = "post-loop-modulo-i32-add"', 'epilogue = "none"', 1
    )
    rejected = invoke("corrupt_post_add", broken, "--bosc-ame-plan-verify")
    assert rejected.returncode != 0 and "register schedule" in rejected.stderr
    broken = saved_plans["post_add"].replace(
        'role = "c-post-add"', 'role = "c-store"', 1
    )
    rejected = invoke("corrupt_post_member", broken, "--bosc-ame-emit")
    assert rejected.returncode != 0 and "register schedule" in rejected.stderr
    # Verification reconstructs source effects and arithmetic instead of
    # accepting otherwise intact scheduling and membership dictionaries.
    broken, changed = re.subn(
        r"(arith\.addi %[\w.]+, %[\w.]+)( : i32\n\s*linalg\.yield)",
        r"\1 overflow<nsw>\2",
        saved_plans["post_add"],
        count=1,
    )
    assert changed == 1
    rejected = invoke("corrupt_post_source", broken, "--bosc-ame-emit")
    assert rejected.returncode != 0 and "register schedule" in rejected.stderr
    # A named operation can carry an explicitly serialized scalar region.
    # The proof must verify that region rather than trust the signed-cast attr.
    serialized = invoke(
        "post_add.serialize", saved_plans["post_add"], "--mlir-print-op-generic"
    )
    assert serialized.returncode == 0, serialized.stderr
    broken = serialized.stdout.replace('"arith.extsi"', '"arith.extui"', 1)
    assert broken != serialized.stdout
    rejected = invoke("corrupt_matmul_region", broken, "--bosc-ame-emit")
    assert rejected.returncode != 0 and "register schedule" in rejected.stderr

    # Entry memrefs need no defining address op; carrying C through a source
    # loop must still recognize and preserve these direct endpoints.
    planned = plan("direct_arguments", DIRECT_ARGUMENT_LOOP)
    assert "plan.register_schedule" in planned
    emitted = invoke("direct_arguments.planned", planned, "--bosc-ame-emit")
    assert emitted.returncode == 0, emitted.stderr
    assert "bosc_ame.register_logical_loop" in emitted.stdout
    assert "contract.loop_carried_slots" in emitted.stdout
    path = work / "direct_arguments.emitted.mlir"
    path.write_text(emitted.stdout)
    a, b = bytearray([1]) * 256, bytearray([2]) * 4096
    output = bytearray([0xA5]) * 1024 + bytearray([0xD9]) * 64
    machine = Machine(path, "matrix")
    machine.launch({"%arg0": a, "%arg1": b, "%arg2": output}, {})
    assert output == struct.pack("<i", 384) * 256 + bytes([0xD9]) * 64


if __name__ == "__main__":
    main()
