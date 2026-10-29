# RUN: %PYTHON %s %t
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Verify planned DMA/RVV boundaries without extending external buffers.

The production planner chooses the segments. The emitter must consume that
plan, and the strict interpreter checks every external byte plus unchanged
guards. Hand-edited plans exercise the verifier and the emitter's own gate.
"""

import re
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def dictionaries(text, marker):
    """Return exact spans of balanced dictionaries following a marker."""
    result = []
    cursor = 0
    while True:
        start = text.find(marker, cursor) if marker else cursor
        if start < 0:
            break
        begin = text.find("{", start + len(marker))
        if begin < 0:
            break
        depth = 1
        end = begin + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        result.append((begin, end, text[begin:end]))
        cursor = end
    return result


def number(text, field):
    found = re.search(rf"\b{field} = (-?\d+) : i64", text)
    assert found, (field, text)
    return int(found[1])


def string(text, field):
    found = re.search(rf'\b{field} = "([^"]*)"', text)
    assert found, (field, text)
    return found[1]


def segments(transfer):
    if string(transfer, "engine") != "segmented":
        return []
    begin = transfer.index("segments = [") + len("segments = [")
    end = transfer.index("]", begin)
    return [entry[2] for entry in dictionaries(transfer[begin:end], "")]


def replace_field(text, field, value):
    return re.sub(
        rf"\b{field} = -?\d+ : i64",
        f"{field} = {value} : i64",
        text,
        count=1,
    )


def remove_field(text, field):
    pattern = rf'\b{field} = (?:-?\d+ : i64|"[^"]*")'
    result, count = re.subn(pattern + r", ", "", text, count=1)
    if count:
        return result
    result, count = re.subn(r", " + pattern, "", text, count=1)
    assert count, (field, text)
    return result


def replace_segment(transfer, index, replacement):
    begin = transfer.index("segments = [") + len("segments = [")
    end = transfer.index("]", begin)
    pieces = segments(transfer)
    if replacement:
        pieces[index] = replacement
    else:
        del pieces[index]
    return transfer[:begin] + ", ".join(pieces) + transfer[end:]


def roundtrip(
    rows,
    columns,
    external_pitch,
    local_pitch,
    offset=0,
    local_offset=0,
    rank_one=False,
):
    """Preserve two boundary copies around one initialized private root."""
    external_count = offset + (rows - 1) * external_pitch + columns
    local_count = local_offset + (rows - 1) * local_pitch + columns
    local_count = (local_count + 15) // 16 * 16
    external_type = (
        f"memref<{rows}x{columns}xi32, "
        f"strided<[{external_pitch}, 1], offset: {offset}>>"
    )
    local_type = (
        f"memref<{rows}x{columns}xi32, "
        f"strided<[{local_pitch}, 1], offset: {local_offset}>>"
    )
    # Single-row tests exercise the rank-one segmented fallback. Rank-two
    # rectangles now prefer exact matrix transport when raw32 is legal.
    sizes = f"[{rows}, {columns}]"
    external_strides = f"[{external_pitch}, 1]"
    local_strides = f"[{local_pitch}, 1]"
    if rows == 1 and rank_one:
        external_type = f"memref<{columns}xi32, strided<[1], offset: {offset}>>"
        local_type = (
            f"memref<{columns}xi32, strided<[1], offset: {local_offset}>>"
        )
        sizes, external_strides, local_strides = f"[{columns}]", "[1]", "[1]"
    source = f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @boundary(%src: memref<{external_count}xi32>,
                      %dst: memref<{external_count}xi32>) {{
    %zero = arith.constant 0 : i32
    %local = memref.alloc() alignment = 64 : memref<{local_count}xi32>
    linalg.fill ins(%zero : i32) outs(%local : memref<{local_count}xi32>)
    %sv = memref.reinterpret_cast %src to offset: [{offset}],
        sizes: {sizes}, strides: {external_strides}
        : memref<{external_count}xi32> to {external_type}
    %dv = memref.reinterpret_cast %dst to offset: [{offset}],
        sizes: {sizes}, strides: {external_strides}
        : memref<{external_count}xi32> to {external_type}
    %lv = memref.reinterpret_cast %local to offset: [{local_offset}],
        sizes: {sizes}, strides: {local_strides}
        : memref<{local_count}xi32> to {local_type}
    memref.copy %sv, %lv : {external_type} to {local_type}
    memref.copy %lv, %dv : {local_type} to {external_type}
    memref.dealloc %local : memref<{local_count}xi32>
    return
  }}
}}
"""
    ranges = [
        ((offset + row * external_pitch) * 4, columns * 4)
        for row in range(rows)
    ]
    return source, external_count * 4, ranges


def view_roundtrip(aligned=False):
    """Account for both the shifted byte pointer and descriptor offset."""
    columns = 128 if aligned else 200
    source_offset, local_offset = (60, 56) if aligned else (8, 4)
    source = f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @boundary(%src: memref<512xi8>, %dst: memref<512xi8>) {{
    %zero = arith.constant 0 : i8
    %s4 = arith.constant 4 : index
    %s8 = arith.constant 8 : index
    %local = memref.alloc() alignment = 64 : memref<512xi8>
    linalg.fill ins(%zero : i8) outs(%local : memref<512xi8>)
    %sv = memref.view %src[%s4][] : memref<512xi8> to memref<256xi8>
    %lv = memref.view %local[%s8][] : memref<512xi8> to memref<256xi8>
    %dv = memref.view %dst[%s4][] : memref<512xi8> to memref<256xi8>
    %sr = memref.reinterpret_cast %sv to offset: [{source_offset}],
        sizes: [{columns}], strides: [1] : memref<256xi8>
        to memref<{columns}xi8, strided<[1], offset: {source_offset}>>
    %lr = memref.reinterpret_cast %lv to offset: [{local_offset}],
        sizes: [{columns}], strides: [1] : memref<256xi8>
        to memref<{columns}xi8, strided<[1], offset: {local_offset}>>
    %dr = memref.reinterpret_cast %dv to offset: [{source_offset}],
        sizes: [{columns}], strides: [1] : memref<256xi8>
        to memref<{columns}xi8, strided<[1], offset: {source_offset}>>
    memref.copy %sr, %lr
        : memref<{columns}xi8, strided<[1], offset: {source_offset}>>
        to memref<{columns}xi8, strided<[1], offset: {local_offset}>>
    memref.copy %lr, %dr
        : memref<{columns}xi8, strided<[1], offset: {local_offset}>>
        to memref<{columns}xi8, strided<[1], offset: {source_offset}>>
    memref.dealloc %local : memref<512xi8>
    return
  }}
}}
"""
    return source, 512, [(4 + source_offset, columns)]


def extui_roundtrip():
    """i2 -1 extends to 3, not an assumed cache-line residue of 63."""
    return """module attributes {
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
} {
  func.func @boundary(%src: memref<*xi8>, %dst: memref<*xi8>) {
    %zero = arith.constant 0 : i8
    %small = arith.constant -1 : i2
    %wide = arith.extui %small : i2 to i64
    %offset = arith.index_cast %wide : i64 to index
    %local = memref.alloc() alignment = 64 : memref<256xi8>
    linalg.fill ins(%zero : i8) outs(%local : memref<256xi8>)
    %sv = memref.reinterpret_cast %src to offset: [%offset],
        sizes: [128], strides: [1] : memref<*xi8>
        to memref<128xi8, strided<[1], offset: ?>>
    %lv = memref.reinterpret_cast %local to offset: [63],
        sizes: [128], strides: [1] : memref<256xi8>
        to memref<128xi8, strided<[1], offset: 63>>
    %dv = memref.reinterpret_cast %dst to offset: [63],
        sizes: [128], strides: [1] : memref<*xi8>
        to memref<128xi8, strided<[1], offset: 63>>
    memref.copy %sv, %lv : memref<128xi8, strided<[1], offset: ?>>
        to memref<128xi8, strided<[1], offset: 63>>
    memref.copy %lv, %dv : memref<128xi8, strided<[1], offset: 63>>
        to memref<128xi8, strided<[1], offset: 63>>
    memref.dealloc %local : memref<256xi8>
    return
  }
}
"""


def halfword_roundtrip(columns):
    """A byte-initialized root isolates the i16 boundary-copy support."""
    return f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @boundary(%src: memref<32xi16>, %dst: memref<32xi16>) {{
    %zero = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %local = memref.alloc() alignment = 64 : memref<64xi8>
    linalg.fill ins(%zero : i8) outs(%local : memref<64xi8>)
    %lv = memref.view %local[%c0][] : memref<64xi8> to memref<32xi16>
    %sv = memref.subview %src[0] [{columns}] [1]
        : memref<32xi16> to memref<{columns}xi16, strided<[1]>>
    %dv = memref.subview %dst[0] [{columns}] [1]
        : memref<32xi16> to memref<{columns}xi16, strided<[1]>>
    %lp = memref.subview %lv[0] [{columns}] [1]
        : memref<32xi16> to memref<{columns}xi16, strided<[1]>>
    memref.copy %sv, %lp : memref<{columns}xi16, strided<[1]>>
        to memref<{columns}xi16, strided<[1]>>
    memref.copy %lp, %dv : memref<{columns}xi16, strided<[1]>>
        to memref<{columns}xi16, strided<[1]>>
    memref.dealloc %local : memref<64xi8>
    return
  }}
}}
"""


def matmul_source(begin, valid):
    """Use the real [N,K] pitch, padded M/N, and one complete K64 block."""
    return f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @matmul(%a: memref<1x768xi8>, %b: memref<1000x768xi8>,
                    %c: memref<1x1000xi32>) {{
    %zi = arith.constant 0 : i8
    %zc = arith.constant 0 : i32
    %aa = memref.alloc() alignment = 64 : memref<32x64xi8>
    %bb = memref.alloc() alignment = 64 : memref<64x64xi8>
    %cc = memref.alloc() alignment = 64 : memref<32x64xi32>
    linalg.fill ins(%zi : i8) outs(%aa : memref<32x64xi8>)
    linalg.fill ins(%zi : i8) outs(%bb : memref<64x64xi8>)
    linalg.fill ins(%zc : i32) outs(%cc : memref<32x64xi32>)
    %ae = memref.subview %a[0, 704] [1, 64] [1, 1]
        : memref<1x768xi8> to memref<1x64xi8, strided<[768, 1], offset: 704>>
    %ap = memref.subview %aa[0, 0] [1, 64] [1, 1]
        : memref<32x64xi8> to memref<1x64xi8, strided<[64, 1]>>
    %be = memref.subview %b[{begin}, 704] [{valid}, 64] [1, 1]
        : memref<1000x768xi8> to memref<{valid}x64xi8, strided<[768, 1], offset: {begin * 768 + 704}>>
    %bp = memref.subview %bb[0, 0] [{valid}, 64] [1, 1]
        : memref<64x64xi8> to memref<{valid}x64xi8, strided<[64, 1]>>
    %ce = memref.subview %c[0, {begin}] [1, {valid}] [1, 1]
        : memref<1x1000xi32> to memref<1x{valid}xi32, strided<[1000, 1], offset: {begin}>>
    %cp = memref.subview %cc[0, 0] [1, {valid}] [1, 1]
        : memref<32x64xi32> to memref<1x{valid}xi32, strided<[64, 1]>>
    memref.copy %ae, %ap
        : memref<1x64xi8, strided<[768, 1], offset: 704>> to memref<1x64xi8, strided<[64, 1]>>
    memref.copy %be, %bp
        : memref<{valid}x64xi8, strided<[768, 1], offset: {begin * 768 + 704}>> to memref<{valid}x64xi8, strided<[64, 1]>>
    memref.copy %ce, %cp
        : memref<1x{valid}xi32, strided<[1000, 1], offset: {begin}>> to memref<1x{valid}xi32, strided<[64, 1]>>
    %bt = memref.transpose %bb (d0, d1) -> (d1, d0)
        : memref<64x64xi8> to memref<64x64xi8, strided<[1, 64]>>
    linalg.matmul {{cast = #linalg.type_fn<cast_signed>}}
        ins(%aa, %bt : memref<32x64xi8>, memref<64x64xi8, strided<[1, 64]>>)
        outs(%cc : memref<32x64xi32>)
    memref.copy %cp, %ce
        : memref<1x{valid}xi32, strided<[64, 1]>> to memref<1x{valid}xi32, strided<[1000, 1], offset: {begin}>>
    memref.dealloc %aa : memref<32x64xi8>
    memref.dealloc %bb : memref<64x64xi8>
    memref.dealloc %cc : memref<32x64xi32>
    return
  }}
}}
"""


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
        target = invoke(
            emitted.stdout, "--bosc-ame-spm-dma-check", "--lower-bosc-ame"
        )
        assert target.returncode == 0, (name, target.stderr)
        path = work / f"{name}.emitted.mlir"
        path.write_text(emitted.stdout)
        return planned.stdout, path

    def simulate_roundtrip(name, source, count, ranges):
        planned, emitted = compile_case(name, source)
        guard = bytes([0xD7]) * 64
        source_bytes = bytes((index * 17 + 5) % 256 for index in range(count))
        buffers = {
            "%arg0": bytearray(source_bytes + guard),
            "%arg1": bytearray(bytes([0xCE]) * count + guard),
        }
        expected = bytearray(buffers["%arg1"])
        allowed = set()
        for begin, length in ranges:
            expected[begin : begin + length] = source_bytes[
                begin : begin + length
            ]
            allowed.update(range(begin, begin + length))
        machine = Machine(emitted, "boundary")
        machine.launch(buffers)
        assert buffers["%arg0"] == source_bytes + guard, name
        assert buffers["%arg1"] == expected, name
        coverage = {"%arg0": Counter(), "%arg1": Counter()}
        for argument, begin, end, action in machine.accesses:
            assert action == ("read" if argument == "%arg0" else "write")
            assert set(range(begin, end)) <= allowed, (name, begin, end)
            coverage[argument].update(range(begin, end))
        assert all(
            set(counts) == allowed and set(counts.values()) == {1}
            for counts in coverage.values()
        ), (name, coverage)
        assert not machine.pending, name
        for _, _, src, dst, length, _ in machine.dma_log:
            assert src % 64 == dst % 64 == length % 64 == 0
        transfers = [
            entry[2] for entry in dictionaries(planned, "plan.transfer =")
        ]
        assert len(transfers) == 2, (name, transfers)
        assert {string(transfer, "from") for transfer in transfers} == {
            "ddr",
            "spm",
        }
        return planned, transfers, machine

    # No row 1 exists: the unused DDR pitch of 1000 i32 is not a DMA address.
    _, transfers, machine = simulate_roundtrip(
        "single_row_unused_pitch", *roundtrip(1, 64, 1000, 64)
    )
    assert all(string(transfer, "engine") == "dma" for transfer in transfers)
    assert len(machine.dma_log) == 2
    assert all(entry[4] == 256 for entry in machine.dma_log)

    for aligned in (False, True):
        name = "view_aligned_sum" if aligned else "view_matching_residues"
        _, transfers, machine = simulate_roundtrip(
            name, *view_roundtrip(aligned)
        )
        for transfer in transfers:
            actual = segments(transfer)
            if aligned:
                # The planner may conservatively keep the segmented envelope,
                # provided that the combined actual address authorizes DMA.
                assert not actual or all(
                    string(segment, "engine") == "dma" for segment in actual
                )
            else:
                assert [string(segment, "engine") for segment in actual] == [
                    "rvv",
                    "dma",
                    "rvv",
                ]
                assert [number(segment, "bytes") for segment in actual] == [
                    52,
                    128,
                    20,
                ]
        assert len(machine.dma_log) == 2
        assert all(entry[2:5] == (64, 64, 128) for entry in machine.dma_log)

    extui_planned, extui_emitted = compile_case(
        "narrow_unsigned_extension", extui_roundtrip()
    )
    extui_transfers = [
        entry[2] for entry in dictionaries(extui_planned, "plan.transfer =")
    ]
    assert len(extui_transfers) == 2
    assert all(
        string(segment, "engine") == "rvv"
        for segment in segments(extui_transfers[0])
    )
    assert (
        sum(
            number(segment, "bytes") for segment in segments(extui_transfers[0])
        )
        == 128
    )
    source_bytes = bytes((index * 11 + 3) % 256 for index in range(256))
    buffers = {
        "%arg0": bytearray(source_bytes),
        "%arg1": bytearray([0xD9] * 256),
    }
    expected = bytearray(buffers["%arg1"])
    expected[63:191] = source_bytes[3:131]
    machine = Machine(extui_emitted, "boundary")
    machine.launch(buffers)
    assert buffers["%arg0"] == source_bytes
    assert buffers["%arg1"] == expected
    assert not machine.pending
    coverage = {"%arg0": Counter(), "%arg1": Counter()}
    for argument, begin, end, action in machine.accesses:
        assert action == ("read" if argument == "%arg0" else "write")
        coverage[argument].update(range(begin, end))
    assert coverage["%arg0"] == Counter(range(3, 131))
    assert coverage["%arg1"] == Counter(range(63, 191))
    assert all(entry[0] != "%arg0" for entry in machine.dma_log)

    halfword_planned, _ = compile_case("halfword_dma", halfword_roundtrip(32))
    assert all(
        string(entry[2], "engine") == "dma"
        for entry in dictionaries(halfword_planned, "plan.transfer =")
    )
    halfword = invoke(
        halfword_roundtrip(31),
        "--kernel-contract-annotate",
        "--bosc-ame-plan=plan-version=1",
    )
    assert halfword.returncode != 0
    assert "no exact 16-bit RVV boundary-copy recipe" in halfword.stderr

    mixed, transfers, machine = simulate_roundtrip(
        "4000_byte_boundary", *roundtrip(1, 1000, 1000, 1024, rank_one=True)
    )
    emitted_text = (work / "4000_byte_boundary.emitted.mlir").read_text()
    # The task is complete only after its final RVV suffix has run. Recording
    # it on the preceding DMA wait would expose an incomplete logical copy.
    for task in (1, 2):
        completion = re.search(
            rf"contract.task = \{{[^}}]*\bid = {task} : i64", emitted_text
        )
        assert completion
        suffix = emitted_text[: completion.start()]
        assert suffix.rfind("vector.store") > suffix.rfind("bosc_ame.dma.wait")
    assert all(
        [string(segment, "engine") for segment in segments(transfer)]
        == ["dma", "rvv"]
        for transfer in transfers
    )
    assert all(
        [number(segment, "bytes") for segment in segments(transfer)]
        == [3968, 32]
        for transfer in transfers
    )
    assert len(machine.dma_log) == 2

    residue_cases = {}
    for name, arguments, engines, sizes in (
        ("short_32_bytes", (1, 8, 8, 16), ["rvv"], [32]),
        (
            "matching_residues",
            (1, 50, 50, 50, 1, 1),
            ["rvv", "dma", "rvv"],
            [60, 128, 12],
        ),
        (
            "different_residues",
            (1, 50, 50, 50, 1, 2),
            ["rvv", "rvv"],
            [192, 8],
        ),
        ("multirow_pitch_68", (2, 16, 17, 16), [], []),
    ):
        planned, transfers, _ = simulate_roundtrip(
            name, *roundtrip(*arguments, rank_one=True)
        )
        residue_cases[name] = planned
        for transfer in transfers:
            if name == "multirow_pitch_68":
                assert string(transfer, "engine") == "matrix"
                assert string(transfer, "wait") == "after-each-tile"
                continue
            actual = segments(transfer)
            assert [
                string(segment, "engine") for segment in actual
            ] == engines, (name, transfer)
            assert [number(segment, "bytes") for segment in actual] == sizes, (
                name,
                transfer,
            )
            assert string(transfer, "wait") == "after-each-segment"
            assert number(transfer, "completion") == len(actual) - 1
            for index, segment in enumerate(actual):
                assert number(segment, "id") == index
                assert number(segment, "predecessor") == index - 1
                if string(segment, "engine") == "rvv":
                    vector_bytes = number(segment, "vector_bytes")
                    assert 0 < vector_bytes <= 64
                    assert number(segment, "bytes") % vector_bytes == 0

    # Mutate only the first segmented transfer. Both verifier and emitter must
    # reject incomplete coverage, overlap, wrong dependencies and illegal DMA.
    begin, end, transfer = dictionaries(mixed, "plan.transfer =")[0]
    first, tail = segments(transfer)
    malformed = {
        "missing_tail": replace_segment(transfer, 1, ""),
        "overlap": replace_segment(
            transfer, 1, replace_field(tail, "source_offset", 0)
        ),
        "destination_gap": replace_segment(
            transfer, 1, replace_field(tail, "target_offset", 3972)
        ),
        "wrong_id": replace_segment(transfer, 1, replace_field(tail, "id", 0)),
        "wrong_predecessor": replace_segment(
            transfer, 1, replace_field(tail, "predecessor", -1)
        ),
        "wrong_completion": replace_field(transfer, "completion", 0),
        "bad_dma_length": replace_segment(
            transfer, 0, replace_field(first, "bytes", 3964)
        ),
        "bad_dma_source": replace_segment(
            transfer, 0, replace_field(first, "source_offset", 4)
        ),
        "bad_dma_destination": replace_segment(
            transfer, 0, replace_field(first, "target_offset", 4)
        ),
        "negative_offset": replace_segment(
            transfer, 1, replace_field(tail, "source_offset", -4)
        ),
        "overflow_offset": replace_segment(
            transfer,
            1,
            replace_field(tail, "source_offset", 9223372036854775804),
        ),
        "zero_segment": replace_segment(
            transfer, 1, replace_field(tail, "bytes", 0)
        ),
        "oversize_vector": replace_segment(
            transfer, 1, replace_field(tail, "vector_bytes", 128)
        ),
        "fractional_element": replace_segment(
            transfer, 1, replace_field(tail, "vector_bytes", 3)
        ),
        "unknown_engine": replace_segment(
            transfer, 1, tail.replace('engine = "rvv"', 'engine = "scalar"')
        ),
        "missing_completion": remove_field(transfer, "completion"),
        "missing_segments": transfer.replace(
            "segments = [" + ", ".join(segments(transfer)) + "], ", ""
        ),
        "wrong_completion_type": transfer.replace(
            "completion = 1 : i64", 'completion = "1"'
        ),
        "wrong_segments_type": transfer.replace(
            "segments = [" + ", ".join(segments(transfer)) + "]",
            'segments = "invalid"',
        ),
        "nondictionary_segment": replace_segment(transfer, 0, "0 : i64"),
    }
    for field in (
        "id",
        "predecessor",
        "source_offset",
        "target_offset",
        "bytes",
        "engine",
        "reason",
        "vector_bytes",
    ):
        malformed["missing_segment_" + field] = replace_segment(
            transfer, 1, remove_field(tail, field)
        )
    for field in ("source_offset", "vector_bytes"):
        malformed["wrong_segment_" + field + "_type"] = replace_segment(
            transfer,
            1,
            re.sub(rf"\b{field} = \d+ : i64", f'{field} = "invalid"', tail),
        )
    malformed["wrong_segment_engine_type"] = replace_segment(
        transfer, 1, tail.replace('engine = "rvv"', "engine = 0 : i64")
    )
    parts = segments(transfer)
    malformed["reordered_segments"] = replace_segment(
        replace_segment(transfer, 0, parts[1]), 1, parts[0]
    )
    for name, replacement in malformed.items():
        assert replacement != transfer, name
        invalid = mixed[:begin] + replacement + mixed[end:]
        for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            rejected = invoke(invalid, gate)
            assert rejected.returncode != 0, (name, gate)
            assert "error:" in rejected.stderr, (name, gate, rejected.stderr)

    # The verifier must independently reject an injected i16 RVV plan, even
    # though the original whole-cache-line i16 DMA route is supported.
    begin, end, original = dictionaries(halfword_planned, "plan.transfer =")[0]
    replacement = original.replace(
        'engine = "dma"', 'completion = 0 : i64, engine = "segmented"'
    ).replace('wait = "after-each-chunk"', 'wait = "after-each-segment"')
    replacement = replacement[:-1] + (
        ', segments = [{bytes = 64 : i64, engine = "rvv", id = 0 : i64, '
        'predecessor = -1 : i64, reason = "injected-halfword-rvv", '
        "source_offset = 0 : i64, target_offset = 0 : i64, "
        "vector_bytes = 64 : i64}]}"
    )
    narrowed = halfword_planned[:begin] + replacement + halfword_planned[end:]
    for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        rejected = invoke(narrowed, gate)
        assert rejected.returncode != 0
        assert "naturally aligned exact element vectors" in rejected.stderr

    # A cache-line-sized row is still illegal DMA when the actual source or
    # target address has a different residue. Cover both transfer directions.
    for name in ("different_residues",):
        planned = residue_cases[name]
        for direction, (begin, end, transfer) in enumerate(
            dictionaries(planned, "plan.transfer =")
        ):
            for index, segment in enumerate(segments(transfer)):
                if string(segment, "engine") != "rvv":
                    continue
                if number(segment, "bytes") % 64:
                    continue
                illegal = segment.replace('engine = "rvv"', 'engine = "dma"')
                illegal = re.sub(r", vector_bytes = \d+ : i64", "", illegal)
                replacement = replace_segment(transfer, index, illegal)
                invalid = planned[:begin] + replacement + planned[end:]
                for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
                    rejected = invoke(invalid, gate)
                    assert rejected.returncode != 0, (name, direction, gate)
                    assert "error:" in rejected.stderr

    # Full N64 and N40 tail both preserve pre-existing C. The external B is
    # [N,K], its parent pitch is 768 bytes, and only one K64 slice is read.
    for begin, valid in ((960, 40), (896, 64)):
        name = f"matmul_n{valid}"
        planned, emitted = compile_case(name, matmul_source(begin, valid))
        c_transfers = [
            entry[2]
            for entry in dictionaries(planned, "plan.transfer =")
            if number(entry[2], "bytes") == valid * 4
            and number(entry[2], "count") == 1
        ]
        assert len(c_transfers) == 2, (name, c_transfers)
        if valid == 40:
            assert all(
                string(transfer, "engine") == "matrix"
                and number(transfer, "total_bytes") == 160
                for transfer in c_transfers
            )
        else:
            assert all(
                string(transfer, "engine") == "dma" for transfer in c_transfers
            )
        a = bytes(128 if k % 2 == 0 else 127 for k in range(768))
        b = bytes(
            (128, 255, 127)[row % 3] for row in range(1000) for _ in range(768)
        )
        initial = [2147483400 + column % 127 for column in range(1000)]
        c = b"".join(struct.pack("<i", value) for value in initial)
        guard = bytes([0xC7]) * 64
        buffers = {
            "%arg0": bytearray(a + guard),
            "%arg1": bytearray(b + guard),
            "%arg2": bytearray(c + guard),
        }
        expected = bytearray(c + guard)
        for column in range(begin, begin + valid):
            dot = sum(
                int.from_bytes(a[k : k + 1], "little", signed=True)
                * int.from_bytes(
                    b[column * 768 + k : column * 768 + k + 1],
                    "little",
                    signed=True,
                )
                for k in range(704, 768)
            )
            value = (initial[column] + dot + (1 << 31)) % (1 << 32) - (1 << 31)
            struct.pack_into("<i", expected, column * 4, value)
        machine = Machine(emitted, "matmul")
        machine.launch(buffers)
        assert buffers["%arg0"] == a + guard
        assert buffers["%arg1"] == b + guard
        assert buffers["%arg2"] == expected, name
        assert not machine.pending
        for argument, start, end, action in machine.accesses:
            if argument == "%arg0":
                assert action == "read" and 704 <= start < end <= 768
            elif argument == "%arg1":
                row, remainder = divmod(start, 768)
                assert action == "read" and begin <= row < begin + valid
                assert remainder >= 704 and end <= row * 768 + 768
            else:
                assert argument == "%arg2"
                assert begin * 4 <= start < end <= (begin + valid) * 4

    print(
        "PASS boundary transfer plans: unused single-row pitch, exact DMA/RVV segments, "
        "view shifts, narrow unsigned offsets, halfword route limits, residues, "
        "multirow endpoints, malformed-plan gates and padded matmul N tails"
    )


if __name__ == "__main__":
    main()
