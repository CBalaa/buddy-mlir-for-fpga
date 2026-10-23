# RUN: %PYTHON %s %t buddy-opt
"""Fixed SCF windows, in-place sums, independent pitches and boundary task IDs."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402

PASS = (
    "--linalg-to-boscame=buffer-placement=true enable-spm-dma=true "
    "dma-abi=args_aligned64_nonalias store-policy=spm-dma plan-report=true"
)


def add(left, right):
    return f"""
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>], iterator_types = ["parallel"]}}
      ins({left}, {right} : memref<16xi32>, memref<16xi32>)
      outs({left} : memref<16xi32>) {{
    ^bb0(%x: i32, %y: i32, %unused: i32):
      %value = arith.addi %x, %y : i32
      linalg.yield %value : i32
    }}
    """


def scoped_source():
    return (
        """
module {
  func.func @scoped(%input: memref<256xi32>, %output: memref<256xi32>,
                    %count: index, %take: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %sum = memref.alloc() alignment = 64 : memref<16xi32>
    %src = memref.subview %input[16] [16] [1]
      : memref<256xi32> to memref<16xi32, strided<[1], offset: 16>>
    memref.copy %src, %sum {bosc_ame.memory = {kind = "load",
      masked = false, valid_shape = array<i64: 16>}}
      : memref<16xi32, strided<[1], offset: 16>> to memref<16xi32>
    scf.for %outer = %c0 to %count step %c1 {
      scf.if %take {
        %a = memref.alloc() alignment = 64 : memref<16xi32>
        linalg.fill ins(%one : i32) outs(%a : memref<16xi32>)
        scf.for %inner = %c0 to %c2 step %c1 {
          %b = memref.alloc() alignment = 64 : memref<16xi32>
          linalg.fill ins(%two : i32) outs(%b : memref<16xi32>)
    """
        + add("%a", "%b")
        + """
        }
    """
        + add("%sum", "%a")
        + """
      } else {
        %other = memref.alloc() alignment = 64 : memref<16xi32>
        linalg.fill ins(%two : i32) outs(%other : memref<16xi32>)
    """
        + add("%sum", "%other")
        + """
      }
    }
    %after = memref.alloc() alignment = 64 : memref<16xi32>
    linalg.fill ins(%one : i32) outs(%after : memref<16xi32>)
    """
        + add("%sum", "%after")
        + """
    %dst = memref.subview %output[32] [16] [1]
      : memref<256xi32> to memref<16xi32, strided<[1], offset: 32>>
    memref.copy %sum, %dst {bosc_ame.memory = {kind = "store",
      destination = "external_ddr", masked = false, valid_shape = array<i64: 16>}}
      : memref<16xi32> to memref<16xi32, strided<[1], offset: 32>>
    return
  }
}
"""
    )


def boundary_source(column_major):
    # The same physical three rows, described with either logical axis order.
    shape = "16x3" if column_major else "3x16"
    sizes = "16, 3" if column_major else "3, 16"
    external = "1, 96" if column_major else "96, 1"
    output = "1, 112" if column_major else "112, 1"
    parent = "64x6" if column_major else "6x64"
    strides = "1, 64" if column_major else "64, 1"
    starts = "16, 2" if column_major else "2, 16"
    return f"""
module attributes {{bosc_ame.compute_graph_version = 1 : i32}} {{
  func.func @boundary(%src: memref<1024xi32> {{
      bosc_ame.logical_value_id = 10 : i64,
      bosc_ame.logical_value_users = array<i64: 0>}},
      %dst: memref<1024xi32>, %row: index)
      attributes {{bosc_ame.compute_graph_complete = true,
                   bosc_ame.compute_graph_node_count = 2 : i64}} {{
    %c96 = arith.constant 96 : index
    %c112 = arith.constant 112 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %a = arith.muli %row, %c96 : index
    %b = arith.muli %row, %c112 : index
    %in_offset = arith.addi %a, %c16 : index
    %out_offset = arith.addi %b, %c32 : index
    %raw = bosc_ame.spm.view {{bank = 1 : i32, offset = 128 : i64}}
      : memref<384xi32, #bosc_ame.spm<1>>
    %parent = memref.reinterpret_cast %raw to offset: [0],
      sizes: [{"64, 6" if column_major else "6, 64"}], strides: [{strides}]
      : memref<384xi32, #bosc_ame.spm<1>>
      to memref<{parent}xi32, strided<[{strides}]>, #bosc_ame.spm<1>>
    %valid = memref.subview %parent[{starts}] [{sizes}] [1, 1]
      : memref<{parent}xi32, strided<[{strides}]>, #bosc_ame.spm<1>>
      to memref<{shape}xi32, strided<[{strides}], offset: 144>, #bosc_ame.spm<1>>
    %input = memref.reinterpret_cast %src to offset: [%in_offset],
      sizes: [{sizes}], strides: [{external}]
      : memref<1024xi32> to memref<{shape}xi32, strided<[{external}], offset: ?>>
    memref.copy %input, %valid {{
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_kind = "load",
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_output_values = array<i64: 11>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = false,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {{kind = "load", masked = true,
                          valid_shape = array<i64: {sizes}>}}}}
      : memref<{shape}xi32, strided<[{external}], offset: ?>>
      to memref<{shape}xi32, strided<[{strides}], offset: 144>, #bosc_ame.spm<1>>
    %output = memref.reinterpret_cast %dst to offset: [%out_offset],
      sizes: [{sizes}], strides: [{output}]
      : memref<1024xi32> to memref<{shape}xi32, strided<[{output}], offset: ?>>
    memref.copy %valid, %output {{
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_kind = "store",
      bosc_ame.compute_input_values = array<i64: 11>,
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = false,
      bosc_ame.varying_program_axes = array<i64>,
      bosc_ame.memory = {{kind = "store", masked = true,
                          valid_shape = array<i64: {sizes}>}}}}
      : memref<{shape}xi32, strided<[{strides}], offset: 144>, #bosc_ame.spm<1>>
      to memref<{shape}xi32, strided<[{output}], offset: ?>>
    return
  }}
}}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = sys.argv[2]

    def lower(name, source):
        path = work / (name + ".mlir")
        input_path = work / (name + ".input.mlir")
        input_path.write_text(source)
        result = subprocess.run(
            [opt, str(input_path), PASS], text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
        path.write_text(result.stdout)
        subprocess.run(
            [opt, str(path), "--bosc-ame-spm-dma-check", "-o", "/dev/null"],
            check=True,
            capture_output=True,
        )
        return path, result.stdout, result.stderr

    path, text, report = lower("scoped", scoped_source())
    assert "resident=5 ddr=0 applied=true" in report, report
    assert "scope=serial-scf fixed-windows=true" in report
    roots = [
        tuple(map(int, match))
        for match in re.findall(
            r"root\[\d+\]: kind=SpmResident bytes=(\d+) bank=(\d+) offset=(\d+)",
            report,
        )
    ]
    assert len(roots) == 5, roots
    for i, (size, bank, offset) in enumerate(roots):
        for j, (other_size, other_bank, other_offset) in enumerate(roots[:i]):
            overlap = bank == other_bank and (
                offset < other_offset + other_size
                and other_offset < offset + size
            )
            # Only the two roots confined to opposite arms may overlap. In
            # particular the outer sum and inner-loop scratch stay distinct.
            assert overlap == ((j, i) == (1, 3)), roots
    assert "memref.alloc" not in text and text.count("bosc_ame.dma.start") == 2
    assert "scope=lexical status=incomplete" in report
    machine = Machine(path, "scoped")
    values = np.arange(256, dtype=np.int64) * 131 + (1 << 31) - 100
    values = values.astype("<i4")
    for count, take in [(0, True), (1, False), (3, True), (0, False)]:
        output = np.full(256, -937, dtype="<i4")
        expected = output.copy()
        expected[32:48] = (
            values[16:32].astype(np.int64) + count * (5 if take else 2) + 1
        ).astype("<i4")
        buffers = {
            "%arg0": bytearray(values.tobytes()),
            "%arg1": bytearray(output.tobytes()),
        }
        first = len(machine.accesses)
        machine.launch(buffers, {"%arg2": count, "%arg3": take})
        np.testing.assert_array_equal(
            np.frombuffer(buffers["%arg1"], dtype="<i4"), expected
        )
        assert buffers["%arg0"] == values.tobytes()
        assert machine.accesses[first:] == [
            ("%arg0", 64, 128, "read"),
            ("%arg1", 128, 192, "write"),
        ]

    for column_major in [False, True]:
        path, text, report = lower(
            "boundary-" + str(column_major), boundary_source(column_major)
        )
        assert (
            "transfers=2 dma-packets=6 read-bytes=192 write-bytes=192" in report
        )
        starts = [
            line for line in text.splitlines() if "bosc_ame.dma.start" in line
        ]
        assert len(starts) == 6
        for index, line in enumerate(starts):
            task = re.search(r"bosc_ame.compute_id = (\d+)", line)
            assert (int(task[1]) if task else None) == {2: 0, 5: 1}.get(index)
        assert "bosc_ame.compute_graph_complete = true" in text
        machine = Machine(path, "boundary")
        for row in [2, 1]:
            output = np.full(1024, -317, dtype="<i4")
            values = (np.arange(1024, dtype=np.int32) * 97 - 1357).astype("<i4")
            expected = output.copy()
            expected_spm = bytearray(machine.spm[1])
            for i in range(3):
                a, b = 16 + (row + i) * 96, 32 + (row + i) * 112
                expected[b : b + 16] = values[a : a + 16]
                start = 128 + (144 + i * 64) * 4
                expected_spm[start : start + 64] = values[a : a + 16].tobytes()
            buffers = {
                "%arg0": bytearray(values.tobytes()),
                "%arg1": bytearray(output.tobytes()),
            }
            machine.launch(buffers, {"%arg2": row})
            np.testing.assert_array_equal(
                np.frombuffer(buffers["%arg1"], dtype="<i4"), expected
            )
            assert buffers["%arg0"] == values.tobytes()
            assert machine.spm[1] == expected_spm

    print(
        "PASS serial SCF placement: private windows, branch reuse, zero trips, pitches and task IDs"
    )


if __name__ == "__main__":
    main()
