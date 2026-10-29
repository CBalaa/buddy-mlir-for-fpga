# RUN: %PYTHON %s %t
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Exercise genuine private-to-private copies through transient raw ACC32.

The source is read again after copying, both roots remain separate SPM
windows, and no logical buffer becomes a resident matrix register value.
"""

import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402
from boscame_sim import SimError  # noqa: E402
from PlanBoundaryTransfers import (  # noqa: E402
    dictionaries,
    number,
    replace_field,
)

WIDTHS = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8}
PATTERN = b"".join(
    struct.pack("<I", value)
    for value in (
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0xFFC54321,
        0x7F812345,
        0xFF854321,
        0x00000001,
        0xFFFFFFFF,
    )
)


def source(
    element="f32",
    rows=3,
    cols=20,
    pitch=32,
    axis=1,
    source_offset=0,
    target_offset=0,
    target_pitch=None,
):
    width = WIDTHS[element]
    target_pitch = pitch if target_pitch is None else target_pitch
    capacity = (
        max(
            source_offset + (rows - 1) * pitch + cols,
            target_offset + (rows - 1) * target_pitch + cols,
        )
        * width
    )
    capacity = (capacity + 63) // 64 * 64
    flat = capacity // width
    shape = [rows, cols] if axis else [cols, rows]
    strides = [pitch, 1] if axis else [1, pitch]
    target_strides = [target_pitch, 1] if axis else [1, target_pitch]
    view = (
        f"memref<{shape[0]}x{shape[1]}x{element}, "
        f"strided<{strides}, offset: {source_offset}>>"
    )
    target_view = (
        f"memref<{shape[0]}x{shape[1]}x{element}, "
        f"strided<{target_strides}, offset: {target_offset}>>"
    )
    local_type = f"memref<{capacity}xi8>"
    typed = f"memref<{flat}x{element}>"
    text = f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @internal(%input: {local_type}, %output: {local_type},
                      %again: {local_type}) {{
    %z = arith.constant 0 : index
    %zero = arith.constant 0 : i8
    %src = memref.alloc() alignment = 64 : {local_type}
    %dst = memref.alloc() alignment = 64 : {local_type}
    linalg.fill ins(%zero : i8) outs(%src : {local_type})
    linalg.fill ins(%zero : i8) outs(%dst : {local_type})
    memref.copy %input, %src : {local_type} to {local_type}
    %st = memref.view %src[%z][] : {local_type} to {typed}
    %dt = memref.view %dst[%z][] : {local_type} to {typed}
    %sv = memref.reinterpret_cast %st to offset: [{source_offset}],
        sizes: {shape}, strides: {strides} : {typed} to {view}
    %dv = memref.reinterpret_cast %dt to offset: [{target_offset}],
        sizes: {shape}, strides: {target_strides} : {typed} to {target_view}
    memref.copy %sv, %dv : {view} to {target_view}
    memref.copy %dst, %output : {local_type} to {local_type}
    memref.copy %src, %again : {local_type} to {local_type}
    memref.dealloc %src : {local_type}
    memref.dealloc %dst : {local_type}
    return
  }}
}}
"""
    ranges = [
        (
            (source_offset + row * pitch) * width,
            (target_offset + row * target_pitch) * width,
            cols * width,
        )
        for row in range(rows)
    ]
    return text, capacity, ranges


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(text, *passes):
        result = subprocess.run(
            [opt, *passes], input=text, text=True, capture_output=True
        )
        assert "LLVM ERROR" not in result.stderr, result.stderr
        assert "PLEASE submit" not in result.stderr, result.stderr
        return result

    def reject(text, *passes, diagnostic=None):
        result = invoke(text, *passes)
        assert result.returncode != 0 and "error:" in result.stderr, (
            passes,
            result,
        )
        if diagnostic:
            assert diagnostic in result.stderr, result.stderr
        return result

    def compile_case(name, case):
        text, capacity, ranges = case
        (work / (name + ".mlir")).write_text(text)
        planned = invoke(
            text,
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-version=1",
            "--bosc-ame-plan-verify",
        )
        assert planned.returncode == 0, (name, planned.stderr)
        (work / (name + ".planned.mlir")).write_text(planned.stdout)
        records = dictionaries(planned.stdout, "plan.transfer =")
        internal = [
            (begin, end, value)
            for begin, end, value in records
            if 'role = "internal"' in value
        ]
        assert len(internal) == 1, (name, internal)
        _, _, recipe = internal[0]
        assert 'recipe = "spm-acc32-spm"' in recipe
        assert 'temporary_window = "none"' in recipe
        assert 'ordering = "synchronous-fenced-tiles"' in recipe
        assert number(recipe, "temp_bytes") == 0
        assert number(recipe, "total_bytes") == sum(
            length for _, _, length in ranges
        )
        endpoints = [
            dictionaries(recipe, key + " =")[0][2]
            for key in ("source_endpoint", "target_endpoint")
        ]
        assert [number(item, "operand") for item in endpoints] == [0, 1]
        windows = [
            entry[2] for entry in dictionaries(planned.stdout, "plan.buffer =")
        ]
        assert len(windows) == 2 and all('level = "spm"' in w for w in windows)
        emitted = invoke(planned.stdout, "--bosc-ame-emit")
        assert emitted.returncode == 0, (name, emitted.stderr)
        assert (
            "plan." not in emitted.stdout
            and "memref.alloc" not in emitted.stdout
        )
        path = work / (name + ".emitted.mlir")
        path.write_text(emitted.stdout)
        lowered = invoke(
            emitted.stdout, "--bosc-ame-spm-dma-check", "--lower-bosc-ame"
        )
        assert lowered.returncode == 0, (name, lowered.stderr)
        assert (
            "mlce32.m acc0" in lowered.stdout
            and "msce32.m acc0" in lowered.stdout
        )
        assert "~{acc0},~{memory}" in lowered.stdout
        (work / (name + ".lowered.mlir")).write_text(lowered.stdout)
        guard = bytes([0xD7]) * 64
        raw = (PATTERN * ((capacity + len(PATTERN) - 1) // len(PATTERN)))[
            :capacity
        ]
        expected = bytearray(capacity)
        for source_offset, target_offset, length in ranges:
            expected[target_offset : target_offset + length] = raw[
                source_offset : source_offset + length
            ]
        for poison in (0xA5, 0x3C):
            machine = Machine(path, "internal")
            for bank, spm in enumerate(machine.spm):
                spm[:] = bytes([(poison + bank) & 255]) * len(spm)
            before = [bytes(spm) for spm in machine.spm]
            buffers = {
                "%arg0": bytearray(raw + guard),
                "%arg1": bytearray(bytes([0xCE]) * capacity + guard),
                "%arg2": bytearray(bytes([0xEF]) * capacity + guard),
            }
            machine.launch(buffers)
            assert buffers["%arg0"] == raw + guard
            assert buffers["%arg1"] == expected + guard, name
            assert buffers["%arg2"] == raw + guard, name
            assert not machine.pending and not machine.live_matrix
            assert machine.matrix_log, name
            for bank, spm in enumerate(machine.spm):
                owned = [
                    (number(w, "offset"), number(w, "capacity"))
                    for w in windows
                    if number(w, "bank") == bank
                ]
                changed = {
                    index
                    for start, size in owned
                    for index in range(start, start + size)
                }
                assert all(
                    spm[index] == old
                    for index, old in enumerate(before[bank])
                    if index not in changed
                ), name
        return planned.stdout, internal[0], emitted.stdout

    primary, (begin, end, recipe), _ = compile_case("raw_f32", source())
    for element in WIDTHS:
        compile_case("raw_" + element, source(element))
    compile_case("physical_axis0", source(axis=0))
    compile_case("multi_tile", source(rows=65, cols=69, pitch=80))
    compile_case(
        "offsets_and_pitches",
        source(source_offset=4, target_offset=8, target_pitch=48),
    )

    # Every piece of explicit execution evidence is checked before emission.
    bad = {
        "register": replace_field(recipe, "register_count", 2),
        "mtype": replace_field(recipe, "mtype", 0),
        "temporary": replace_field(recipe, "temp_bytes", 64),
        "completion": replace_field(recipe, "completion", -1),
        "endpoint_bank": replace_field(recipe, "bank", 7),
        "endpoint_range": replace_field(recipe, "end", 131073),
        "endpoint_capacity": replace_field(recipe, "capacity", 1),
        "recipe": recipe.replace(
            'recipe = "spm-acc32-spm"', 'recipe = "tr-acc"'
        ),
        "ordering": recipe.replace(
            'ordering = "synchronous-fenced-tiles"', 'ordering = "asynchronous"'
        ),
        "window": recipe.replace(
            'temporary_window = "none"', 'temporary_window = "implicit"'
        ),
    }
    for name, replacement in bad.items():
        assert replacement != recipe, name
        malformed = primary[:begin] + replacement + primary[end:]
        for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            reject(malformed, gate)
    # A precomputed static origin does not make a dynamically typed endpoint
    # eligible for the target operation's closed static-descriptor contract.
    line = next(
        line for line in primary.splitlines() if 'role = "internal"' in line
    )
    match = re.fullmatch(
        r"(\s*)memref.copy (%[\w]+), (%[\w]+) (\{.*\}) : (memref<.*>) to (memref<.*>)",
        line,
    )
    assert match, line
    indent, source_value, target_value, attrs, source_type, target_type = (
        match.groups()
    )
    for index in (0, 1):
        values, types = [source_value, target_value], [source_type, target_type]
        dynamic = types[index].replace(
            "strided<[32, 1]>", "strided<[32, 1], offset: ?>"
        )
        assert dynamic != types[index]
        cast = (
            f"{indent}%dynamic = memref.cast {values[index]} : "
            f"{types[index]} to {dynamic}\n"
        )
        values[index], types[index] = "%dynamic", dynamic
        replacement = (
            f"{indent}memref.copy {values[0]}, {values[1]} {attrs} : "
            f"{types[0]} to {types[1]}"
        )
        malformed = primary.replace(line, cast + replacement)
        for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            reject(
                malformed, gate, diagnostic="requires static descriptor offsets"
            )
    buffers = dictionaries(primary, "plan.buffer =")
    first_window, second_window = buffers[0][2], buffers[1][2]
    overlap = replace_field(second_window, "bank", number(first_window, "bank"))
    overlap = replace_field(overlap, "offset", number(first_window, "offset"))
    overlap = primary[: buffers[1][0]] + overlap + primary[buffers[1][1] :]
    for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        reject(overlap, gate, diagnostic="overlapping SPM windows")

    # Dynamic descriptor types are rejected by the planner as well.
    text, _, _ = source()
    for endpoint in ("sv", "dv"):
        static = "memref<3x20xf32, strided<[32, 1], offset: 0>>"
        dynamic = "memref<3x20xf32, strided<[32, 1], offset: ?>>"
        casted = text.replace(
            "    memref.copy %sv, %dv",
            f"    %dynamic = memref.cast %{endpoint} : {static} to {dynamic}\n"
            "    memref.copy %sv, %dv",
        )
        if endpoint == "sv":
            casted = casted.replace(
                f"memref.copy %sv, %dv : {static}",
                f"memref.copy %dynamic, %dv : {dynamic}",
            )
        else:
            casted = casted.replace(
                f"memref.copy %sv, %dv : {static} to {static}",
                f"memref.copy %sv, %dynamic : {static} to {dynamic}",
            )
        reject(
            casted,
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-version=1",
            diagnostic="requires static descriptor offsets",
        )
    for target in ("%sv",):
        aliased = text.replace(
            "memref.copy %sv, %dv", "memref.copy %sv, " + target
        )
        reject(
            aliased,
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-version=1",
            diagnostic="distinct nonaliasing roots",
        )
    unsupported = text.replace("f32", "f16").replace("0 : f16", "0.0 : f16")
    reject(
        unsupported,
        "--kernel-contract-annotate",
        "--bosc-ame-plan=plan-version=1",
        diagnostic="no legal internal copy recipe",
    )
    rank_one = text.replace("memref.copy %sv, %dv", "memref.copy %src, %dst")
    rank_one = re.sub(
        r"(memref.copy %src, %dst :) .* to .*",
        r"\1 memref<384xi8> to memref<384xi8>",
        rank_one,
    )
    reject(
        rank_one,
        "--kernel-contract-annotate",
        "--bosc-ame-plan=plan-version=1",
        diagnostic="no legal internal copy recipe",
    )
    too_large, _, _ = source(rows=65, cols=1024, pitch=1024)
    reject(
        too_large,
        "--kernel-contract-annotate",
        "--bosc-ame-plan=plan-version=1",
        diagnostic="SPM workspace capacity exceeded",
    )
    call = text.replace(
        "  func.func @internal",
        "  func.func private @unknown()\n  func.func @internal",
    )
    call = call.replace(
        "    memref.copy %sv, %dv",
        "    func.call @unknown() : () -> ()\n    memref.copy %sv, %dv",
    )
    reject(
        call,
        "--kernel-contract-annotate",
        "--bosc-ame-plan=plan-version=1",
        "--bosc-ame-plan-verify",
    )

    # Validate the physical contract independently of planner evidence. The
    # complete endpoint ranges must fit distinct roots even if one tile would
    # touch a smaller subset, and a scratch ACC bundle cannot cross live state.
    attrs = (
        "source_offset = 0 : i64, target_offset = 0 : i64, "
        "rows = 2 : i64, row_bytes = 64 : i64, "
        "source_pitch = 64 : i64, target_pitch = 64 : i64, "
        "instruction_bits = 32 : i64, register_count = 1 : i64, "
        'mtype = 65602 : i64, register_class = "acc"'
    )
    memory = "memref<2x16xi32, #bosc_ame.spm<0>>"
    physical = f"""module attributes {{bosc_ame.target = "nr-fpga"}} {{
  func.func @physical() {{
    %s = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : {memory}
    %d = bosc_ame.spm.view {{bank = 0 : i32, offset = 128 : i64}} : {memory}
    bosc_ame.matrix.copy %s, %d {{{attrs}}} : {memory}, {memory}
    return
  }}
}}
"""
    physical_valid = invoke(
        physical, "--bosc-ame-spm-dma-check", "--lower-bosc-ame"
    )
    assert physical_valid.returncode == 0, physical_valid.stderr
    for name, invalid in (
        (
            "physical_overlap",
            physical.replace("offset = 128 : i64", "offset = 64 : i64"),
        ),
        ("physical_same_root", physical.replace("copy %s, %d", "copy %s, %s")),
        (
            "physical_bank_bounds",
            physical.replace("offset = 128 : i64", "offset = 131008 : i64"),
        ),
        (
            "physical_negative_offset",
            physical.replace(attrs, replace_field(attrs, "source_offset", -4)),
        ),
    ):
        for gate in ("--bosc-ame-spm-dma-check", "--lower-bosc-ame"):
            reject(invalid, gate)
        # The interpreter independently rejects malformed operations, rather
        # than relying on the compiler's op verifier. Use canonical spellings.
        path = work / (name + ".mlir")
        path.write_text(invalid)
        try:
            Machine(path, "physical").launch({})
        except SimError:
            pass
        else:
            raise AssertionError("simulator accepted " + name)
    live = physical.replace(
        "    bosc_ame.matrix.copy",
        f"    %acc = bosc_ame.tile.load_acc %s : {memory} -> vector<2x16xi32>\n"
        "    bosc_ame.matrix.copy",
    ).replace(
        "    return",
        f"    bosc_ame.tile.store_acc %acc, %s : vector<2x16xi32>, {memory}\n    return",
    )
    for gate in ("--bosc-ame-spm-dma-check", "--lower-bosc-ame"):
        reject(live, gate, diagnostic="live")
    pending = (
        physical.replace("@physical()", "@physical(%arg0: memref<64xi8>)")
        .replace(
            "    bosc_ame.matrix.copy",
            """    %bytes = arith.constant 64 : index
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %token = bosc_ame.dma.start %arg0, %dma, %bytes {channel = 1 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<1>
    bosc_ame.matrix.copy""",
        )
        .replace(
            "    return",
            "    bosc_ame.dma.wait %token : !bosc_ame.dma_token<1>\n    return",
        )
        .replace(
            'bosc_ame.target = "nr-fpga"',
            'bosc_ame.target = "nr-fpga", bosc_ame.dma.abi = "args_aligned64_nonalias"',
        )
    )
    for gate in ("--bosc-ame-spm-dma-check", "--lower-bosc-ame"):
        reject(pending, gate, diagnostic="in flight")
    print("PASS: explicit internal SPM-ACC32-SPM copies and rejection gates")


if __name__ == "__main__":
    main()
