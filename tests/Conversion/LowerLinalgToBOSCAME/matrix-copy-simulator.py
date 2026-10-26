# RUN: %PYTHON %s
"""Check strict internal ACC32 copying without requiring a compiler build."""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import WIDTHS, Machine, SimError  # noqa: E402
from boscame_sim import op_name  # noqa: E402

RAW_BITS = (
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
)
PATTERN = b"".join(bits.to_bytes(4, "little") for bits in RAW_BITS)
PATTERN += b"".join(
    bits.to_bytes(8, "little")
    for bits in (
        0x0000000000000000,
        0x8000000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8123456789ABC,
        0xFFF8543210FEDCBA,
        0x7FF0123456789ABC,
        0xFFF0543210FEDCBA,
    )
)


def module(
    element="f32",
    bank=0,
    target_base=2048,
    rows=3,
    row_bytes=24,
    source_pitch=40,
    target_pitch=56,
    axis=1,
    window_bytes=512,
):
    width = WIDTHS[element]
    shape = [rows, row_bytes // width]
    if axis == 0:
        shape.reverse()
    declarations = []
    types = []
    for name, selected_bank, base, shift, pitch in (
        ("source", 0, 64, 8, source_pitch),
        ("target", bank, target_base, 16, target_pitch),
    ):
        space = f"#bosc_ame.spm<{selected_bank}>"
        root_type = f"memref<{window_bytes}xi8, {space}>"
        flat_type = f"memref<{(window_bytes - 32) // width}x{element}, {space}>"
        strides = [pitch // width, 1]
        if axis == 0:
            strides.reverse()
        view_type = (
            f"memref<{shape[0]}x{shape[1]}x{element}, "
            f"strided<[{strides[0]}, {strides[1]}], offset: {8 // width}>, "
            f"{space}>"
        )
        types.append(view_type)
        declarations.append(
            f"    %{name}_root = bosc_ame.spm.view {{bank = {selected_bank} : i32, offset = {base} : i64}} : {root_type}\n"
            f"    %{name}_shift = arith.constant {shift} : index\n"
            f"    %{name}_flat = memref.view %{name}_root[%{name}_shift][] : {root_type} to {flat_type}\n"
            f"    %{name} = memref.reinterpret_cast %{name}_flat to offset: [{8 // width}], sizes: [{shape[0]}, {shape[1]}], strides: [{strides[0]}, {strides[1]}] : {flat_type} to {view_type}"
        )
    return (
        'module attributes {bosc_ame.target = "nr-fpga"} {\n'
        "  func.func @copy() {\n"
        + "\n".join(declarations)
        + f'\n    bosc_ame.matrix.copy %source, %target {{source_offset = 0 : i64, target_offset = 0 : i64, rows = {rows} : i64, row_bytes = {row_bytes} : i64, source_pitch = {source_pitch} : i64, target_pitch = {target_pitch} : i64, instruction_bits = 32 : i64, register_count = 1 : i64, mtype = 65602 : i64, register_class = "acc"}} : {types[0]}, {types[1]}\n'
        "    return\n  }\n}\n"
    )


def change(source, field, value):
    changed, count = re.subn(
        rf"\b{field} = -?\d+ : i64", f"{field} = {value} : i64", source
    )
    assert count == 1, (field, count)
    return changed


def prepare(path, source):
    path.write_text(source)
    machine = Machine(path, "copy")
    operation = None
    for op in machine.function.blocks[0]:
        if op_name(op.text) == "bosc_ame.matrix.copy":
            operation = op
            break
        machine.run(op)
    assert operation is not None
    for bank, data in enumerate(machine.spm):
        data[:] = bytes([0xD3 - bank]) * len(data)
    source_buffer = machine.values["%source"]
    index = 0
    for row in range(source_buffer.shape[0]):
        for column in range(source_buffer.shape[1]):
            address = source_buffer.address([row, column])
            source_buffer.data[address : address + source_buffer.width] = (
                PATTERN * 2
            )[index % len(PATTERN) : index % len(PATTERN) + source_buffer.width]
            index += source_buffer.width
    return machine, operation


def positive(path, source):
    machine, operation = prepare(path, source)
    expected = [bytearray(bank) for bank in machine.spm]
    attrs = {
        name: int(re.search(rf"\b{name} = (-?\d+) : i64", operation.text)[1])
        for name in (
            "source_offset",
            "target_offset",
            "rows",
            "row_bytes",
            "source_pitch",
            "target_pitch",
        )
    }
    src, dst = (machine.values["%" + name] for name in ("source", "target"))
    assert src.spm_root is not dst.spm_root
    for row in range(attrs["rows"]):
        start = (
            src.address([0, 0])
            + attrs["source_offset"]
            + row * attrs["source_pitch"]
        )
        end = start + attrs["row_bytes"]
        target = (
            dst.address([0, 0])
            + attrs["target_offset"]
            + row * attrs["target_pitch"]
        )
        expected[dst.bank][target : target + attrs["row_bytes"]] = src.data[
            start:end
        ]
    machine.run(operation)
    assert machine.spm == expected, "copy changed source, padding or guards"
    assert len(machine.matrix_log) == 1
    assert machine.accesses == [] and machine.dma_log == []
    assert not machine.pending and not machine.live_matrix


def rejected(path, source, diagnostic, mutate=None):
    machine, operation = prepare(path, source)
    if mutate:
        mutate(machine)
    before = [bytes(bank) for bank in machine.spm]
    try:
        machine.run(operation)
    except SimError as error:
        assert diagnostic in str(error), (diagnostic, str(error))
    else:
        raise AssertionError("invalid internal matrix copy accepted")
    assert [bytes(bank) for bank in machine.spm] == before
    assert (
        not machine.matrix_log and not machine.accesses and not machine.dma_log
    )


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "copy.mlir"
        for element in ("i8", "i16", "i32", "i64", "f32", "f64"):
            for bank in (0, 1):
                for axis in (0, 1):
                    positive(
                        path,
                        module(
                            element=element,
                            bank=bank,
                            axis=axis,
                            rows=3,
                            row_bytes=128,
                            source_pitch=144,
                            target_pitch=160,
                        ),
                    )
        positive(
            path,
            module(
                rows=64,
                row_bytes=256,
                source_pitch=256,
                target_pitch=256,
                target_base=32768,
                window_bytes=16448,
            ),
        )
        partial = module(rows=4, row_bytes=32)
        for field, value in (
            ("source_offset", 48),
            ("target_offset", 64),
            ("rows", 2),
            ("row_bytes", 16),
        ):
            partial = change(partial, field, value)
        positive(path, partial)
        # One physical row permits an unused short pitch; byte origins still
        # describe columns rather than being divided by the unused pitch.
        positive(path, module(rows=1, source_pitch=4, target_pitch=8))
        # No write can destroy a future tile's source: prove whole endpoint
        # envelopes, even when this bundle's actual rows do not overlap.
        overlap = change(module(target_base=128), "rows", 1)
        rejected(path, overlap, "nonoverlapping complete SPM")
        # Distinct physical subranges of one declared root remain unproven.
        alias = (
            module()
            .replace(
                "%target_root[%target_shift]", "%source_root[%target_shift]"
            )
            .replace(
                "%target_shift = arith.constant 16",
                "%target_shift = arith.constant 256",
            )
        )
        rejected(path, alias, "nonoverlapping complete SPM")
        # Preserve the same declared identity through subview and cast too.
        subviews = """module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @copy() {
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<8x8xi32, #bosc_ame.spm<0>>
    %source = memref.subview %root[0, 0] [2, 4] [1, 1] : memref<8x8xi32, #bosc_ame.spm<0>> to memref<2x4xi32, strided<[8, 1]>, #bosc_ame.spm<0>>
    %target_view = memref.subview %root[4, 0] [2, 4] [1, 1] : memref<8x8xi32, #bosc_ame.spm<0>> to memref<2x4xi32, strided<[8, 1], offset: 32>, #bosc_ame.spm<0>>
    %target = memref.cast %target_view : memref<2x4xi32, strided<[8, 1], offset: 32>, #bosc_ame.spm<0>> to memref<2x4xi32, strided<[8, 1], offset: 32>, #bosc_ame.spm<0>>
    bosc_ame.matrix.copy %source, %target {source_offset = 0 : i64, target_offset = 0 : i64, rows = 2 : i64, row_bytes = 16 : i64, source_pitch = 32 : i64, target_pitch = 32 : i64, instruction_bits = 32 : i64, register_count = 1 : i64, mtype = 65602 : i64, register_class = "acc"} : memref<2x4xi32, strided<[8, 1]>, #bosc_ame.spm<0>>, memref<2x4xi32, strided<[8, 1], offset: 32>, #bosc_ame.spm<0>>
    return
  }
}
"""
        rejected(path, subviews, "nonoverlapping complete SPM")
        dynamic_endpoint = module().replace(
            "strided<[14, 1], offset: 2>",
            "strided<[14, 1], offset: ?>",
        )
        rejected(path, dynamic_endpoint, "static endpoint ranges")
        # An early legal tile cannot justify an endpoint escaping its window.
        escape = change(module(), "rows", 1).replace(
            "sizes: [3, 6]", "sizes: [13, 6]"
        )
        rejected(path, escape, "complete endpoint ranges")
        rejected(
            path,
            module(target_base=131008),
            "backing storage",
        )
        for field, value, diagnostic in (
            ("source_offset", -4, "geometry"),
            ("target_offset", 2, "geometry"),
            ("source_offset", 4, "valid endpoint rows"),
            ("source_pitch", 44, "valid endpoint rows"),
            ("target_pitch", 60, "valid endpoint rows"),
            ("rows", 65, "configuration"),
            ("row_bytes", 260, "configuration"),
            ("row_bytes", 22, "configuration"),
            ("instruction_bits", 16, "configuration"),
            ("register_count", 2, "configuration"),
            ("mtype", 0, "configuration"),
        ):
            rejected(path, change(module(), field, value), diagnostic)
        rejected(
            path,
            module().replace('register_class = "acc"', 'register_class = "tr"'),
            "configuration",
        )
        rejected(
            path,
            module().replace(
                'bosc_ame.target = "nr-fpga"', 'bosc_ame.target = "other"'
            ),
            "target profile",
        )
        rejected(
            path,
            module(),
            "before DMA wait",
            lambda m: m.pending.update({0: None}),
        )
        rejected(
            path,
            module(),
            "matrix live range",
            lambda m: setattr(m, "live_matrix", True),
        )
        rejected(
            path,
            module(),
            "declared SPM windows",
            lambda m: setattr(m.values["%target"], "spm_root", None),
        )
        for attribute, value, diagnostic in (
            ("bank", 2, "valid bank"),
            ("data", bytearray(131072), "valid bank"),
            ("element", "index", "matching static rank-two"),
            ("shape", [3, 5], "matching static rank-two"),
            ("strides", [10, 2], "contiguous axis"),
            ("strides", [5, 1], "valid endpoint rows"),
            ("base", 2050, "e32 aligned"),
            ("offset", -1, "declared SPM windows"),
        ):
            rejected(
                path,
                module(),
                diagnostic,
                lambda m, a=attribute, v=value: setattr(
                    m.values["%target"], a, v
                ),
            )
    print(
        "PASS internal SPM ACC32 raw-byte copies, independent windows and strict rejection gates"
    )


if __name__ == "__main__":
    main()
