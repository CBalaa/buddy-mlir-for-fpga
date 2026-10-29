# RUN: %PYTHON %s %t
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Prove exact raw32 matrix transport, automatic routes and rejection gates.

All positive cases use the real annotate/plan/verify/emit/check/lower pipeline.
The interpreter compares bytes, never Python floating-point values, so NaN
payloads and signed zero cannot be accidentally canonicalized by the oracle.
"""

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PlanBoundaryTransfers import (  # noqa: E402
    dictionaries,
    number,
    remove_field,
    replace_field,
    roundtrip,
    segments,
    string,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402
from boscame_sim import SimError  # noqa: E402

WIDTHS = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8}
# Raw IEEE-754 patterns: both zeros/infinities, qNaN/sNaN with distinct signs
# and payloads, normal/subnormal extrema, and ordinary finite values.
F32_BITS = (
    0x00000000,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC12345,
    0xFFC54321,
    0x7F812345,
    0xFF854321,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x7F7FFFFF,
    0x3F800000,
    0xBF800000,
)
RAW_PATTERN = b"".join(bits.to_bytes(4, "little") for bits in F32_BITS)


def raw_roundtrip(
    rows,
    columns,
    source_pitch,
    local_pitch,
    element="f32",
    source_offset=0,
    local_offset=0,
    target_pitch=None,
    target_offset=None,
    source_shift=0,
    local_shift=0,
    target_shift=0,
    axis=1,
):
    """Use byte roots and typed views, isolating copy from typed arithmetic.

    Pitches and descriptor offsets are in logical elements; shifts are bytes.
    Both copies retain the same logical rectangle but can have distinct DDR
    source/destination layouts and offsets. Axis 0 covers column-major views.
    """
    width = WIDTHS[element]
    target_pitch = source_pitch if target_pitch is None else target_pitch
    target_offset = source_offset if target_offset is None else target_offset
    counts = [
        offset + (rows - 1) * pitch + columns
        for offset, pitch in (
            (source_offset, source_pitch),
            (local_offset, local_pitch),
            (target_offset, target_pitch),
        )
    ]
    shifts = [source_shift, local_shift, target_shift]
    byte_counts = [
        count * width + shift for count, shift in zip(counts, shifts)
    ]
    byte_counts[1] = (byte_counts[1] + 63) // 64 * 64
    shape = [rows, columns] if axis == 1 else [columns, rows]
    pitches = [source_pitch, local_pitch, target_pitch]
    offsets = [source_offset, local_offset, target_offset]
    view_types = []
    declarations = []
    for name, root, count, byte_count, shift, pitch, offset in zip(
        ("sv", "lv", "dv"),
        ("src", "local", "dst"),
        counts,
        byte_counts,
        shifts,
        pitches,
        offsets,
    ):
        strides = [pitch, 1] if axis == 1 else [1, pitch]
        view_type = (
            f"memref<{shape[0]}x{shape[1]}x{element}, "
            f"strided<[{strides[0]}, {strides[1]}], offset: {offset}>>"
        )
        view_types.append(view_type)
        declarations.append(
            f"    %{name}_shift = arith.constant {shift} : index\n"
            f"    %{name}_base = memref.view %{root}[%{name}_shift][] "
            f": memref<{byte_count}xi8> to memref<{count}x{element}>\n"
            f"    %{name} = memref.reinterpret_cast %{name}_base "
            f"to offset: [{offset}], sizes: [{shape[0]}, {shape[1]}], "
            f"strides: [{strides[0]}, {strides[1]}] "
            f": memref<{count}x{element}> to {view_type}"
        )
    source = f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @boundary(%src: memref<{byte_counts[0]}xi8>,
                      %dst: memref<{byte_counts[2]}xi8>) {{
    %zero = arith.constant 0 : i8
    %local = memref.alloc() alignment = 64 : memref<{byte_counts[1]}xi8>
    linalg.fill ins(%zero : i8) outs(%local : memref<{byte_counts[1]}xi8>)
{chr(10).join(declarations)}
    memref.copy %sv, %lv : {view_types[0]} to {view_types[1]}
    memref.copy %lv, %dv : {view_types[1]} to {view_types[2]}
    memref.dealloc %local : memref<{byte_counts[1]}xi8>
    return
  }}
}}
"""
    source_ranges, local_ranges, target_ranges = [
        [
            ((offset + row * pitch) * width + shift, columns * width)
            for row in range(rows)
        ]
        for offset, pitch, shift in zip(offsets, pitches, shifts)
    ]
    return {
        "source": source,
        "sizes": (byte_counts[0], byte_counts[2]),
        "source_ranges": source_ranges,
        "target_ranges": target_ranges,
        "local_ranges": local_ranges,
        "local_size": byte_counts[1],
        "shape": shape,
        "strides": [[p, 1] if axis == 1 else [1, p] for p in pitches],
        "offsets": offsets,
        "row_bytes": columns * width,
        "rows": rows,
        "element_bits": width * 8,
    }


def list_entries(transfer, field):
    begin = transfer.index(field + " = [") + len(field + " = [")
    end = transfer.index("]", begin)
    return [entry[2] for entry in dictionaries(transfer[begin:end], "")]


def replace_entry(transfer, field, index, replacement):
    begin = transfer.index(field + " = [") + len(field + " = [")
    end = transfer.index("]", begin)
    entries = list_entries(transfer, field)
    if replacement is None:
        del entries[index]
    else:
        entries[index] = replacement
    return transfer[:begin] + ", ".join(entries) + transfer[end:]


def replace_dictionary(transfer, field, replacement):
    begin, end, _ = dictionaries(transfer, field + " =")[0]
    return transfer[:begin] + replacement + transfer[end:]


def dense_array(text, field):
    result = re.search(rf"\b{field} = array<i64: ([^>]*)>", text)
    assert result, (field, text)
    return [int(value.strip()) for value in result[1].split(",")]


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(source, *passes):
        result = subprocess.run(
            [opt, *passes], input=source, text=True, capture_output=True
        )
        assert "LLVM ERROR" not in result.stderr, result.stderr
        assert "PLEASE submit" not in result.stderr, result.stderr
        return result

    def rejected(source, *passes, diagnostic=None):
        result = invoke(source, *passes)
        assert result.returncode != 0, (passes, source)
        assert "error:" in result.stderr, (passes, result.stderr)
        if diagnostic:
            assert diagnostic in result.stderr, (diagnostic, result.stderr)
        return result

    def compile_case(name, source):
        (work / f"{name}.mlir").write_text(source)
        planned = invoke(
            source,
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-version=1 plan-report=true",
            "--bosc-ame-plan-verify",
        )
        assert planned.returncode == 0, (name, planned.stderr)
        (work / f"{name}.planned.mlir").write_text(planned.stdout)
        emitted = invoke(planned.stdout, "--bosc-ame-emit")
        assert emitted.returncode == 0, (name, emitted.stderr)
        assert "plan." not in emitted.stdout and "linalg." not in emitted.stdout
        path = work / f"{name}.emitted.mlir"
        path.write_text(emitted.stdout)
        lowered = invoke(
            emitted.stdout, "--bosc-ame-spm-dma-check", "--lower-bosc-ame"
        )
        assert lowered.returncode == 0, (name, lowered.stderr)
        (work / f"{name}.lowered.mlir").write_text(lowered.stdout)
        # Every transfer is an inseparable side-effecting config/load/store/
        # fence bundle with a visible physical-register and memory clobber.
        copies = emitted.stdout.count("bosc_ame.matrix.copy ")
        if copies:
            assert lowered.stdout.count("mlce32.m acc0") == copies
            assert lowered.stdout.count("msce32.m acc0") == copies
            assert lowered.stdout.count("fence rw, rw") >= copies
            assert lowered.stdout.count("~{acc0},~{memory}") == copies
            bundles = [
                line
                for line in lowered.stdout.splitlines()
                if "mlce32.m acc0" in line
            ]
            assert len(bundles) == copies
            for bundle in bundles:
                assert "has_side_effects" in bundle, bundle
                assert "msettilem" in bundle and "msettilen" in bundle
                assert "msettype" in bundle and "msce32.m acc0" in bundle
                assert "fence rw, rw" in bundle and "call" not in bundle
        return planned.stdout, emitted.stdout, path

    def compile_object(name, lowered_path, matrix_copies):
        """Exercise optimization, fixed-register encoding and strict NR codegen."""
        llvm_ir = work / f"{name}.ll"
        assembly = work / f"{name}.s"
        encoded = work / f"{name}.encoded.s"
        restricted = work / f"{name}.nr.s"
        obj = work / f"{name}.o"

        def command(arguments, input_text=None):
            result = subprocess.run(
                [str(argument) for argument in arguments],
                input=input_text,
                text=True,
                capture_output=True,
            )
            assert result.returncode == 0, (arguments, result.stderr)
            assert "LLVM ERROR" not in result.stderr, result.stderr
            return result.stdout

        command(
            [
                ROOT / "build/bin/buddy-translate",
                "--buddy-to-llvmir",
                lowered_path,
                "-o",
                llvm_ir,
            ]
        )
        command(
            [
                ROOT / "llvm/build/bin/llc",
                llvm_ir,
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-code-model=medium",
                "-o",
                assembly,
            ]
        )
        assembly_text = assembly.read_text()
        assert assembly_text.count("mlce32.m") >= matrix_copies
        assert assembly_text.count("msce32.m") >= matrix_copies
        assert assembly_text.count("fence") >= matrix_copies
        # LLVM may surround user assembly with APP/NO_APP markers. Within
        # each matrix-copy bundle the effective dimensions, configuration,
        # exact load/store and fence must remain adjacent in that order.
        bundle_pattern = re.compile(
            r"msettilem[^\n]*\n\s*msettilen[^\n]*\n"
            r"\s*msettype[^\n]*\n\s*mlce32\.m\s+acc0[^\n]*\n"
            r"\s*msce32\.m\s+acc0[^\n]*\n\s*fence[^\n]*"
        )
        assert len(bundle_pattern.findall(assembly_text)) == matrix_copies, (
            assembly_text
        )
        common = ROOT / "examples/FPGA-BOSCAME/tools"
        encoded.write_text(
            command(
                [
                    sys.executable,
                    common / "ame_to_word.py",
                    "--gpr-mode=fixed",
                    "--preserve-matrix-registers",
                ],
                assembly_text,
            )
        )
        restricted.write_text(
            command(
                [
                    sys.executable,
                    common / "restrict_fpga_assembly.py",
                    "--no-fences",
                    "--order-vector-copies",
                ],
                encoded.read_text(),
            )
        )
        command(
            [
                ROOT / "llvm/build/bin/clang",
                "--target=riscv64-unknown-elf",
                "-march=rv64imafdcv_zicsr_zifencei",
                "-mabi=lp64d",
                "-mno-relax",
                "-c",
                restricted,
                "-o",
                obj,
            ]
        )
        assert obj.stat().st_size > 0

    def simulate(name, case, engine="matrix"):
        planned, emitted, path = compile_case(name, case["source"])
        transfers = [
            entry[2] for entry in dictionaries(planned, "plan.transfer =")
        ]
        assert len(transfers) == 2, (name, transfers)
        assert all(string(item, "engine") == engine for item in transfers)
        assert [string(item, "from") for item in transfers] == ["ddr", "spm"]
        sizes = case["sizes"]
        source_bytes = (
            RAW_PATTERN
            * ((sizes[0] + len(RAW_PATTERN) - 1) // len(RAW_PATTERN))
        )[: sizes[0]]
        guard = bytes([0xD7]) * 64
        machine = Machine(path, "boundary")
        windows = [
            entry[2]
            for entry in dictionaries(planned, "plan.buffer =")
            if 'level = "spm"' in entry[2]
        ]
        assert len(windows) == 1, (name, windows)
        bank, start, capacity = (
            number(windows[0], field)
            for field in ("bank", "offset", "capacity")
        )
        assert capacity >= case["local_size"]
        for poison in (0xA5, 0x3C):
            for index, spm in enumerate(machine.spm):
                spm[:] = bytes([(poison + index) & 255]) * len(spm)
            before_spm = [bytes(spm) for spm in machine.spm]
            buffers = {
                "%arg0": bytearray(source_bytes + guard),
                "%arg1": bytearray(bytes([0xCE]) * sizes[1] + guard),
            }
            expected = bytearray(buffers["%arg1"])
            expected_local = bytearray(case["local_size"])
            source_coverage, target_coverage = Counter(), Counter()
            for (src, length), (dst, other_length), (
                local,
                local_length,
            ) in zip(
                case["source_ranges"],
                case["target_ranges"],
                case["local_ranges"],
            ):
                assert length == other_length == local_length
                expected[dst : dst + length] = source_bytes[src : src + length]
                expected_local[local : local + length] = source_bytes[
                    src : src + length
                ]
                source_coverage.update(range(src, src + length))
                target_coverage.update(range(dst, dst + length))
            machine.accesses.clear()
            machine.dma_log.clear()
            machine.matrix_log.clear()
            machine.launch(buffers)
            assert buffers["%arg0"] == source_bytes + guard, name
            assert buffers["%arg1"] == expected, name
            coverage = {"%arg0": Counter(), "%arg1": Counter()}
            for argument, begin, end, action in machine.accesses:
                assert action == ("read" if argument == "%arg0" else "write")
                coverage[argument].update(range(begin, end))
            assert coverage["%arg0"] == source_coverage, name
            assert coverage["%arg1"] == target_coverage, name
            assert all(
                value == 1
                for counts in coverage.values()
                for value in counts.values()
            ), name
            assert not machine.pending and not machine.live_matrix
            assert (
                machine.spm[bank][start : start + case["local_size"]]
                == expected_local
            )
            for index, spm in enumerate(machine.spm):
                if index != bank:
                    assert bytes(spm) == before_spm[index], name
                else:
                    assert spm[:start] == before_spm[index][:start], name
                    assert (
                        spm[start + case["local_size"] :]
                        == (before_spm[index][start + case["local_size"] :])
                    ), name
            if engine == "matrix":
                assert not machine.dma_log, name
                assert len(machine.matrix_log) == sum(
                    len(list_entries(item, "tiles")) for item in transfers
                )
        return planned, transfers, emitted

    # Full/tail rectangles in both directions: 788 B and 64 B are independent.
    full_case = raw_roundtrip(64, 16, 197, 16)
    full_planned, full_transfers, _ = simulate("pitch788_cols16", full_case)
    report = invoke(
        full_case["source"],
        "--kernel-contract-annotate",
        "--bosc-ame-plan=plan-version=1 plan-report=true",
        "--bosc-ame-plan-verify",
    )
    assert report.returncode == 0, report.stderr
    (work / "pitch788_cols16.plan-report.txt").write_text(report.stderr)
    reported_transfers = dictionaries(report.stderr, "plan.transfer=")
    assert len(reported_transfers) == 2, report.stderr
    for (_, _, transfer), planned in zip(reported_transfers, full_transfers):
        assert transfer == planned, report.stderr
        assert string(transfer, "engine") == "matrix"
        assert string(transfer, "reason") == "dma-illegal-exact-acc32-rectangle"
        assert number(transfer, "total_bytes") == 4096
        assert number(transfer, "temp_bytes") == 0
        tile = list_entries(transfer, "tiles")[0]
        assert number(tile, "register_count") == 1
        assert string(tile, "register_class") == "acc"
        assert number(tile, "instruction_bits") == 32
        for endpoint in ("source_layout", "target_layout"):
            layout = dictionaries(transfer, endpoint + " =")[0][2]
            assert dense_array(layout, "shape") == [64, 16]
            assert number(layout, "element_bits") == 32
            assert number(layout, "row_bytes") == 64
    buffers = dictionaries(report.stderr, "plan.buffer=")
    assert len(buffers) == 1, report.stderr
    buffer = buffers[0][2]
    assert string(buffer, "level") == "spm"
    assert number(buffer, "capacity") == 4096
    assert number(buffer, "offset") == 0
    for _, _, transfer in reported_transfers:
        assert number(transfer, "channel") == number(buffer, "bank") == 0
    budgets = dictionaries(report.stderr, "plan.budget=")
    assert len(budgets) == 1, report.stderr
    budget = budgets[0][2]
    assert number(budget, "spm_banks") == 2
    assert number(budget, "bank_capacity") == 131072
    assert dense_array(budget, "high_water") == [4096, 0]
    compile_object("pitch788_cols16", work / "pitch788_cols16.lowered.mlir", 2)
    tail_case = raw_roundtrip(64, 5, 197, 16)
    _, tail_transfers, _ = simulate("pitch788_cols5", tail_case)
    compile_object("pitch788_cols5", work / "pitch788_cols5.lowered.mlir", 2)
    for transfer, src_pitch, dst_pitch in (
        (full_transfers[0], 788, 64),
        (full_transfers[1], 64, 788),
        (tail_transfers[0], 788, 64),
        (tail_transfers[1], 64, 788),
    ):
        tile = list_entries(transfer, "tiles")[0]
        assert number(tile, "source_pitch") == src_pitch
        assert number(tile, "target_pitch") == dst_pitch
        assert number(tile, "rows") == 64
        assert number(tile, "row_bytes") == number(transfer, "bytes")

    # Descriptor offsets and byte views must each contribute once; destination
    # DDR need not have the source's pitch or descriptor offset.
    simulate(
        "independent_offsets_pitches",
        raw_roundtrip(
            7,
            5,
            197,
            19,
            source_offset=3,
            local_offset=5,
            target_pitch=203,
            target_offset=7,
            source_shift=4,
            local_shift=8,
            target_shift=12,
        ),
    )
    simulate("column_major", raw_roundtrip(9, 5, 197, 19, axis=0))
    simulate("degenerate_common_axis", raw_roundtrip(1, 5, 1, 16, axis=0))

    # Raw e32 describes the instruction, not the logical element type. Typed
    # i8/i16/f64 storage is legal only when every byte range is exactly raw32.
    for element, columns, external_pitch, local_pitch in (
        ("i8", 20, 788, 64),
        ("i16", 10, 394, 32),
        ("i32", 5, 197, 16),
        ("i64", 3, 99, 8),
        ("f32", 5, 197, 16),
        ("f64", 3, 99, 8),
    ):
        simulate(
            "logical_" + element,
            raw_roundtrip(8, columns, external_pitch, local_pitch, element),
        )

    # More rows and columns than one physical ACC32: independent tiles must
    # cover every byte exactly once, including a short row/column remainder.
    chunk_case = raw_roundtrip(65, 69, 103, 80)
    chunk_planned, chunk_transfers, chunk_emitted = simulate(
        "rows_and_columns_chunked", chunk_case
    )
    emitted_tiles = [
        line
        for line in chunk_emitted.splitlines()
        if "bosc_ame.matrix.copy " in line
    ]
    assert len(emitted_tiles) == 8
    assert ["contract.task" in line for line in emitted_tiles] == (
        [False, False, False, True] * 2
    ), emitted_tiles
    for transfer in chunk_transfers:
        tiles = list_entries(transfer, "tiles")
        assert [
            (number(tile, "rows"), number(tile, "row_bytes")) for tile in tiles
        ] == [(64, 256), (64, 20), (1, 256), (1, 20)]
        assert number(transfer, "completion") == 3
        assert number(transfer, "temp_bytes") == 0
        assert number(transfer, "total_bytes") == 65 * 69 * 4
        assert string(transfer, "wait") == "after-each-tile"
        assert string(transfer, "reason") == "dma-illegal-exact-acc32-rectangle"
        for index, tile in enumerate(tiles):
            assert number(tile, "id") == index
            assert number(tile, "predecessor") == index - 1
            assert number(tile, "instruction_bits") == 32
            assert number(tile, "register_count") == 1
            assert number(tile, "mtype") == 65602
            assert string(tile, "register_class") == "acc"
            assert number(tile, "rows") * number(tile, "row_bytes") <= 16384
        for endpoint in ("source_layout", "target_layout"):
            layout = dictionaries(transfer, endpoint + " =")[0][2]
            assert dense_array(layout, "shape") == [65, 69]
            assert number(layout, "element_bits") == 32

    # A single row has no pitch constraint. Its second column tile starts at
    # byte 256, past both otherwise unused parent pitches; it remains in row 0.
    _, single_row_transfers, _ = simulate(
        "single_row_short_parent_pitch", raw_roundtrip(1, 69, 16, 32)
    )
    for transfer, source_pitch, target_pitch in (
        (single_row_transfers[0], 64, 128),
        (single_row_transfers[1], 128, 64),
    ):
        tiles = list_entries(transfer, "tiles")
        assert [
            (
                number(tile, "rows"),
                number(tile, "row_bytes"),
                number(tile, "source_offset"),
                number(tile, "target_offset"),
            )
            for tile in tiles
        ] == [(1, 256, 0, 0), (1, 20, 256, 256)]
        assert all(
            number(tile, "source_pitch") == source_pitch for tile in tiles
        )
        assert all(
            number(tile, "target_pitch") == target_pitch for tile in tiles
        )
        assert number(transfer, "total_bytes") == 276
        assert number(transfer, "temp_bytes") == 0
        assert number(transfer, "completion") == 1

    # Pure DMA is retained; an ineligible rank-1 rectangle keeps exact RVV.
    simulate("legal_dma_retained", raw_roundtrip(4, 16, 32, 16), "dma")
    rank_one, _, _ = roundtrip(1, 5, 5, 16, rank_one=True)
    mixed_parts = []
    for name, source in (
        ("dma_route", raw_roundtrip(4, 16, 32, 16)["source"]),
        ("matrix_route", full_case["source"]),
        ("rvv_route", rank_one),
    ):
        body = source[source.index("  func.func") : source.rfind("}")]
        mixed_parts.append(body.replace("@boundary(", "@" + name + "("))
    mixed = (
        "module attributes {contract.abi = {args_aligned_bytes = 64 : i64, "
        "args_disjoint = true}} {\n" + "\n".join(mixed_parts) + "}\n"
    )
    mixed_planned, _, _ = compile_case("mixed_engines", mixed)
    routes = [
        string(item[2], "engine")
        for item in dictionaries(mixed_planned, "plan.transfer =")
    ]
    assert Counter(routes) == {"dma": 2, "matrix": 2, "segmented": 2}, routes

    # Natural raw32 alignment is independent of logical alignment. i8 short
    # rows and 2-byte offsets/pitches fall back to the existing exact RVV path.
    for name, case in (
        (
            "fallback_byte_offset2",
            raw_roundtrip(3, 20, 788, 64, "i8", source_offset=2),
        ),
        ("fallback_byte_pitch2", raw_roundtrip(3, 20, 790, 64, "i8")),
        ("fallback_short_row", raw_roundtrip(3, 3, 788, 64, "i8")),
    ):
        _, transfers, _ = simulate(name, case, "segmented")
        assert all("matrix_rejection" in transfer for transfer in transfers)
        assert all(segments(transfer) for transfer in transfers)
    for name, case in (
        (
            "halfword_offset2",
            raw_roundtrip(3, 10, 394, 32, "i16", source_offset=1),
        ),
        ("halfword_pitch2", raw_roundtrip(3, 10, 395, 32, "i16")),
        ("halfword_short_row", raw_roundtrip(3, 9, 394, 32, "i16")),
    ):
        (work / (name + ".mlir")).write_text(case["source"])
        rejected(
            case["source"],
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-version=1",
            diagnostic="no exact 16-bit RVV boundary-copy recipe",
        )
    # Floating-point half formats are not in the validated raw ACC32 logical
    # storage set. These are rejection-only text mutations, not byte oracles:
    # their 20-byte rows and 788/64-byte pitches would otherwise permit raw32.
    unsupported_source = raw_roundtrip(3, 10, 394, 32)["source"]
    for element in ("f16", "bf16"):
        invalid = unsupported_source.replace("f32", element)
        (work / f"unsupported_{element}.mlir").write_text(invalid)
        rejected(
            invalid,
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-version=1",
            diagnostic="no exact 16-bit RVV boundary-copy recipe",
        )

    # Mutate only the first multi-tile plan. The standalone verifier and emit's
    # own preflight both reject schema, coverage, ordering and configuration.
    begin, end, transfer = dictionaries(chunk_planned, "plan.transfer =")[0]
    tiles = list_entries(transfer, "tiles")
    first, second = tiles[:2]
    malformed = {
        "missing_tile": replace_entry(transfer, "tiles", 3, None),
        "repeated_tile": replace_entry(transfer, "tiles", 1, first),
        "reordered_tiles": replace_entry(
            replace_entry(transfer, "tiles", 0, second), "tiles", 1, first
        ),
        "wrong_completion": replace_field(transfer, "completion", 1),
        "missing_completion": remove_field(transfer, "completion"),
        "wrong_total": replace_field(transfer, "total_bytes", 17944),
        "nonzero_temporary": replace_field(transfer, "temp_bytes", 64),
        "wrong_wait": transfer.replace(
            'wait = "after-each-tile"', 'wait = "group"'
        ),
        "unknown_reason": transfer.replace(
            'reason = "dma-illegal-exact-acc32-rectangle"',
            'reason = "invented"',
        ),
        "wrong_tiles_type": re.sub(
            r"tiles = \[.*\]", 'tiles = "invalid"', transfer
        ),
        "nondictionary_tile": replace_entry(transfer, "tiles", 0, "0 : i64"),
    }
    for field, value in (
        ("id", 7),
        ("predecessor", 3),
        ("source_offset", 4),
        ("target_offset", 4),
        ("rows", 65),
        ("row_bytes", 252),
        ("source_pitch", 416),
        ("target_pitch", 324),
        ("instruction_bits", 16),
        ("register_count", 2),
        ("mtype", 0),
    ):
        malformed["bad_tile_" + field] = replace_entry(
            transfer, "tiles", 0, replace_field(first, field, value)
        )
    malformed["negative_source"] = replace_entry(
        transfer, "tiles", 0, replace_field(first, "source_offset", -4)
    )
    malformed["overflow_source"] = replace_entry(
        transfer,
        "tiles",
        0,
        replace_field(first, "source_offset", 9223372036854775804),
    )
    malformed["wrong_register_class"] = replace_entry(
        transfer,
        "tiles",
        0,
        first.replace('register_class = "acc"', 'register_class = "tr"'),
    )
    for field in (
        "id",
        "predecessor",
        "source_offset",
        "target_offset",
        "rows",
        "row_bytes",
        "source_pitch",
        "target_pitch",
        "instruction_bits",
        "register_count",
        "register_class",
        "mtype",
    ):
        malformed["missing_tile_" + field] = replace_entry(
            transfer, "tiles", 0, remove_field(first, field)
        )
    for field in ("completion", "total_bytes", "temp_bytes"):
        malformed["wrong_" + field + "_type"] = re.sub(
            rf"\b{field} = -?\d+ : i64", f'{field} = "invalid"', transfer
        )
    for field in ("source_offset", "row_bytes", "mtype"):
        malformed["wrong_tile_" + field + "_type"] = replace_entry(
            transfer,
            "tiles",
            0,
            re.sub(rf"\b{field} = -?\d+ : i64", f'{field} = "invalid"', first),
        )
    for endpoint in ("source_layout", "target_layout"):
        layout = dictionaries(transfer, endpoint + " =")[0][2]
        malformed["wrong_" + endpoint + "_type"] = replace_dictionary(
            transfer, endpoint, '"invalid"'
        )
        malformed["wrong_" + endpoint + "_shape_type"] = replace_dictionary(
            transfer,
            endpoint,
            re.sub(r"shape = array<i64: [^>]*>", 'shape = "invalid"', layout),
        )
        malformed["bad_" + endpoint + "_shape"] = replace_dictionary(
            transfer,
            endpoint,
            layout.replace("array<i64: 65, 69>", "array<i64: 65, 68>"),
        )
        malformed["bad_" + endpoint + "_element_bits"] = replace_dictionary(
            transfer, endpoint, replace_field(layout, "element_bits", 16)
        )
        malformed["bad_" + endpoint + "_strides"] = replace_dictionary(
            transfer,
            endpoint,
            re.sub(
                r"strides = array<i64: [^>]*>",
                "strides = array<i64: 1, 1>",
                layout,
            ),
        )
    for name, replacement in malformed.items():
        assert replacement != transfer, name
        invalid = chunk_planned[:begin] + replacement + chunk_planned[end:]
        for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            rejected(invalid, gate)

    # Handwritten target operations exercise op verification independently of
    # the planner. Each mutation retains a syntactically valid target module.
    attrs = (
        "source_offset = 0 : i64, target_offset = 0 : i64, "
        "rows = 2 : i64, row_bytes = 20 : i64, "
        "source_pitch = 788 : i64, target_pitch = 64 : i64, "
        "instruction_bits = 32 : i64, register_count = 1 : i64, "
        'mtype = 65602 : i64, register_class = "acc"'
    )
    handwritten = f"""module attributes {{bosc_ame.target = "nr-fpga",
      bosc_ame.dma.abi = "args_aligned64_nonalias"}} {{
  func.func @handwritten(%src: memref<2x5xf32, strided<[197, 1]>>) {{
    %storage = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}}
        : memref<256xi8, #bosc_ame.spm<0>>
    %zero = arith.constant 0 : index
    %typed = memref.view %storage[%zero][]
        : memref<256xi8, #bosc_ame.spm<0>> to memref<64xf32, #bosc_ame.spm<0>>
    %dst = memref.reinterpret_cast %typed to offset: [0],
        sizes: [2, 5], strides: [16, 1]
        : memref<64xf32, #bosc_ame.spm<0>>
          to memref<2x5xf32, strided<[16, 1]>, #bosc_ame.spm<0>>
    bosc_ame.matrix.copy %src, %dst {{{attrs}}}
        : memref<2x5xf32, strided<[197, 1]>>,
          memref<2x5xf32, strided<[16, 1]>, #bosc_ame.spm<0>>
    return
  }}
}}
"""
    valid = invoke(handwritten, "--bosc-ame-spm-dma-check", "--lower-bosc-ame")
    assert valid.returncode == 0, valid.stderr
    for element in ("f16", "bf16"):
        invalid = handwritten.replace("f32", element)
        (work / f"handwritten_unsupported_{element}.mlir").write_text(invalid)
        for gate in ("--bosc-ame-spm-dma-check", "--lower-bosc-ame"):
            rejected(
                invalid,
                gate,
                diagnostic="logical storage through the raw ACC32 memory path",
            )
    for name, replacement in (
        ("rows_capacity", replace_field(attrs, "rows", 65)),
        ("row_capacity", replace_field(attrs, "row_bytes", 260)),
        ("short_word", replace_field(attrs, "row_bytes", 18)),
        ("negative", replace_field(attrs, "source_offset", -4)),
        ("source_column_bound", replace_field(attrs, "source_offset", 4)),
        ("target_column_bound", replace_field(attrs, "target_offset", 4)),
        ("source_pitch", replace_field(attrs, "source_pitch", 792)),
        ("target_pitch", replace_field(attrs, "target_pitch", 68)),
        ("bad_mtype", replace_field(attrs, "mtype", 0)),
        ("bad_width", replace_field(attrs, "instruction_bits", 16)),
        ("bad_register_count", replace_field(attrs, "register_count", 2)),
        (
            "bad_register_class",
            attrs.replace('register_class = "acc"', 'register_class = "tr"'),
        ),
    ):
        invalid = handwritten.replace(attrs, replacement)
        for gate in ("--bosc-ame-spm-dma-check", "--lower-bosc-ame"):
            rejected(invalid, gate)
    raw_handwritten = (
        handwritten.replace("f32", "i8")
        .replace("2x5xi8", "2x20xi8")
        .replace("64xi8", "256xi8")
        .replace("sizes: [2, 5]", "sizes: [2, 20]")
        .replace("strides: [16, 1]", "strides: [64, 1]")
        .replace("strided<[197, 1]>", "strided<[788, 1]>")
        .replace("strided<[16, 1]>", "strided<[64, 1]>")
    )
    raw_valid = invoke(raw_handwritten, "--bosc-ame-spm-dma-check")
    assert raw_valid.returncode == 0, raw_valid.stderr
    for invalid in (
        handwritten.replace(
            'bosc_ame.target = "nr-fpga"', 'bosc_ame.target = "unknown"'
        ),
        handwritten.replace(
            'bosc_ame.dma.abi = "args_aligned64_nonalias"',
            "contract.abi = {args_aligned_bytes = 2 : i64}",
        ),
    ):
        rejected(invalid, "--bosc-ame-spm-dma-check", "--lower-bosc-ame")
    for invalid in (
        raw_handwritten.replace(
            "strided<[788, 1]>", "strided<[788, 1], offset: 2>"
        ),
        raw_handwritten.replace(
            "strided<[64, 1]>", "strided<[64, 1], offset: 2>"
        ).replace("to offset: [0]", "to offset: [2]"),
    ):
        rejected(invalid, "--bosc-ame-spm-dma-check", diagnostic="aligned")

    # The interpreter validates the target operation independently rather
    # than trusting compiler validation or silently accepting oversized/raw
    # unaligned transfers. Start from canonical, parser-supported syntax.
    canonical_handwritten = invoke(handwritten)
    assert canonical_handwritten.returncode == 0, canonical_handwritten.stderr
    for name, field, value in (
        ("simulator_bad_mtype", "mtype", 0),
        ("simulator_bad_width", "instruction_bits", 16),
        ("simulator_large_rows", "rows", 65),
        ("simulator_large_row", "row_bytes", 260),
        ("simulator_word_tail", "row_bytes", 18),
        ("simulator_column_bound", "source_offset", 4),
        ("simulator_source_pitch", "source_pitch", 792),
        ("simulator_target_pitch", "target_pitch", 68),
    ):
        invalid_text = replace_field(canonical_handwritten.stdout, field, value)
        invalid_path = work / (name + ".mlir")
        invalid_path.write_text(invalid_text)
        try:
            Machine(invalid_path, "handwritten").launch(
                {"%arg0": bytearray(808)}
            )
        except SimError as error:
            assert "matrix copy" in str(error) or "raw32" in str(error), error
        else:
            raise AssertionError("simulator accepted " + name)

    # Target checker and interpreter both reject overlap with a live request,
    # even when that request uses a disjoint DDR endpoint and SPM window.
    pending = (
        handwritten.replace(
            "%src: memref<2x5xf32, strided<[197, 1]>>",
            "%src: memref<2x5xf32, strided<[197, 1]>>, %other: memref<64xi8>",
        )
        .replace(
            "    bosc_ame.matrix.copy",
            """    %bytes = arith.constant 64 : index
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %token = bosc_ame.dma.start %other, %dma, %bytes {channel = 1 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<1>
    bosc_ame.matrix.copy""",
        )
        .replace(
            "    return",
            "    bosc_ame.dma.wait %token : !bosc_ame.dma_token<1>\n    return",
        )
    )
    rejected(pending, "--bosc-ame-spm-dma-check", diagnostic="in flight")
    pending_path = work / "pending_dma.mlir"
    canonical_pending = invoke(pending)
    assert canonical_pending.returncode == 0, canonical_pending.stderr
    pending_path.write_text(canonical_pending.stdout)
    try:
        Machine(pending_path, "handwritten").launch(
            {"%arg0": bytearray(808), "%arg1": bytearray(64)}
        )
    except SimError as error:
        assert "matrix copy before DMA wait" in str(error), error
    else:
        raise AssertionError("simulator accepted matrix copy with DMA pending")

    # A legal compute chain distinguishes new liveness rejection from the
    # pre-existing matrix-value user restrictions. In particular TR is kept
    # live for a later MMA, never consumed by an unsupported vector extract.
    matrix_storage = """    %a = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<2x64xi8, #bosc_ame.spm<1>>
    %b_bytes = bosc_ame.spm.view {bank = 1 : i32, offset = 128 : i64}
        : memref<1024xi8, #bosc_ame.spm<1>>
    %b = memref.reinterpret_cast %b_bytes to offset: [0],
        sizes: [64, 16], strides: [1, 64]
        : memref<1024xi8, #bosc_ame.spm<1>>
          to memref<64x16xi8, strided<[1, 64]>, #bosc_ame.spm<1>>
    %c = bosc_ame.spm.view {bank = 1 : i32, offset = 1152 : i64}
        : memref<2x16xi32, #bosc_ame.spm<1>>
"""
    load_a = """    %ta = bosc_ame.tile.load_a %a
        : memref<2x64xi8, #bosc_ame.spm<1>> -> vector<2x64xi8>
"""
    load_b = """    %tb = bosc_ame.tile.load_b %b
        : memref<64x16xi8, strided<[1, 64]>, #bosc_ame.spm<1>>
          -> vector<64x16xi8>
"""
    load_c = """    %tc = bosc_ame.tile.load_acc %c
        : memref<2x16xi32, #bosc_ame.spm<1>> -> vector<2x16xi32>
"""
    mma_store = """    %next = bosc_ame.tile.mma %tc, %ta, %tb
        : vector<2x16xi32>, vector<2x64xi8>, vector<64x16xi8>
          -> vector<2x16xi32>
    bosc_ame.tile.store_acc %next, %c
        : vector<2x16xi32>, memref<2x16xi32, #bosc_ame.spm<1>>
"""
    copy_start = handwritten.index("    bosc_ame.matrix.copy")
    copy_end = handwritten.index("    return", copy_start)
    copy_text = handwritten[copy_start:copy_end]
    for register, before, after in (
        ("tr", load_a, load_b + load_c + mma_store),
        ("acc", load_a + load_b + load_c, mma_store),
    ):
        live = handwritten.replace(
            copy_text, matrix_storage + before + copy_text + after
        )
        path = work / ("live_" + register + ".mlir")
        path.write_text(live)
        for gate in ("--bosc-ame-spm-dma-check", "--lower-bosc-ame"):
            result = rejected(live, gate)
            assert "live" in result.stderr and (
                "TR/ACC" in result.stderr or "accumulator" in result.stderr
            ), result.stderr

    # An accumulator carried by an enclosing loop and captured TR operands
    # remain live even though their origins are outside the copy's block.
    loop = (
        """    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %out = scf.for %i = %zero to %two step %one
        iter_args(%iter = %tc) -> (vector<2x16xi32>) {
"""
        + copy_text
        + """      %next = bosc_ame.tile.mma %iter, %ta, %tb
          : vector<2x16xi32>, vector<2x64xi8>, vector<64x16xi8>
            -> vector<2x16xi32>
      scf.yield %next : vector<2x16xi32>
    }
    bosc_ame.tile.store_acc %out, %c
        : vector<2x16xi32>, memref<2x16xi32, #bosc_ame.spm<1>>
"""
    )
    loop_live = handwritten.replace(
        copy_text, matrix_storage + load_a + load_b + load_c + loop
    )
    (work / "loop_carried_live.mlir").write_text(loop_live)
    for gate in ("--bosc-ame-spm-dma-check", "--lower-bosc-ame"):
        result = rejected(loop_live, gate)
        assert "live" in result.stderr and (
            "TR/ACC" in result.stderr or "accumulator" in result.stderr
        ), result.stderr

    # Completed transfer -> complete AME chain -> policy call -> transfer is
    # legal: no register survives the call, and every compute reconfigures.
    mixed_compute = handwritten.replace(
        "  func.func @handwritten",
        "  func.func private @bosc_ame_ddr_read_begin()\n  func.func @handwritten",
    ).replace(
        copy_text,
        copy_text
        + matrix_storage
        + load_a
        + load_b
        + load_c
        + mma_store
        + "    func.call @bosc_ame_ddr_read_begin() : () -> ()\n"
        + copy_text,
    )
    canonical_mixed = invoke(mixed_compute)
    assert canonical_mixed.returncode == 0, canonical_mixed.stderr
    mixed_emitted_path = work / "mixed_compute.mlir"
    mixed_emitted_path.write_text(canonical_mixed.stdout)
    mixed_machine = Machine(mixed_emitted_path, "handwritten")
    mixed_source = (RAW_PATTERN * 16)[:808]
    mixed_source += bytes(808 - len(mixed_source))
    mixed_guard = bytes([0xE3]) * 64
    mixed_buffers = {"%arg0": bytearray(mixed_source + mixed_guard)}
    mixed_machine.launch(mixed_buffers)
    assert mixed_buffers["%arg0"] == mixed_source + mixed_guard
    # The unrelated 2x64 by 64x16 AME computation operates on poisoned bank1.
    # Model signed i8 multiplication plus i32 wrap independently of the op.
    old_acc = int.from_bytes(bytes([0x5A]) * 4, "little", signed=True)
    expected_acc = ((old_acc + 64 * 90 * 90) % (1 << 32)).to_bytes(4, "little")
    assert mixed_machine.spm[1][1152:1280] == expected_acc * 32
    assert len(mixed_machine.matrix_log) == 2
    assert not mixed_machine.pending and not mixed_machine.live_matrix
    mixed_coverage = Counter()
    for argument, begin, end, action in mixed_machine.accesses:
        assert argument == "%arg0" and action == "read"
        mixed_coverage.update(range(begin, end))
    assert mixed_coverage == Counter(
        {byte: 2 for begin in (0, 788) for byte in range(begin, begin + 20)}
    )
    mixed_lowered = invoke(
        mixed_compute, "--bosc-ame-spm-dma-check", "--lower-bosc-ame"
    )
    assert mixed_lowered.returncode == 0, mixed_lowered.stderr
    mixed_path = work / "mixed_compute.lowered.mlir"
    mixed_path.write_text(mixed_lowered.stdout)
    compile_object("mixed_compute", mixed_path, 2)

    print(
        "PASS: matrix raw-byte transport, routes, exact ranges and rejection gates"
    )


if __name__ == "__main__":
    main()
