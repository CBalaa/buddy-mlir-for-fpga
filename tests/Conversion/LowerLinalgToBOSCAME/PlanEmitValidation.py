# RUN: %PYTHON %s %t
"""Verify concrete plans and simulate emitted direct and pitched-DMA accesses."""

import math
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    examples = ROOT / "tests/Conversion/LowerLinalgToBOSCAME/Inputs/contracts"
    direct = (examples / "matmul128.planned.mlir").read_text()
    mixed = (examples / "kernel.planned.mlir").read_text()
    norm = (examples / "layernorm.planned.mlir").read_text()

    def invoke(name, source, emit=False):
        path = work / f"{name}.mlir"
        path.write_text(source)
        return subprocess.run(
            [
                str(buddy_opt_path()),
                str(path),
                "--bosc-ame-emit" if emit else "--bosc-ame-plan-verify",
            ],
            text=True,
            capture_output=True,
        )

    transposed_dma = """
module attributes {
  contract.version = 1 : i32,
  contract.producer = "plan-emit-test",
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true},
  plan.version = 1 : i32, plan.target_profile = "nr-fpga"
} {
  func.func @transpose_dma(
      %source: memref<64x64xi8, strided<[1, 64]>> {contract.value_id = 0 : i64})
      attributes {
        contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
          complete = true, scope_count = 1 : i64, source_lineage_complete = false},
        plan.budget = {spm_banks = 2 : i64, bank_capacity = 131072 : i64,
          high_water = array<i64: 4096, 0>}
      } {
    %target = memref.alloc() alignment = 64 {
      contract.buffer = {role = "tile", kind = "temporary", logical_shape = array<i64: 64, 64>},
      plan.buffer = {level = "spm", bank = 0 : i64, offset = 0 : i64,
        capacity = 4096 : i64, role = "input"}} : memref<64x64xi8>
    memref.copy %source, %target {
      contract.task = {id = 0 : i64, kind = "memory-load", scope = 0 : i64,
        stage = 0 : i64, predecessors = array<i64>, input_values = array<i64: 0>,
        output_values = array<i64: 1>, barrier_after = true, rematerializable = false},
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}},
      plan.transfer = {role = "input", engine = "dma", channel = 0 : i64,
        from = "ddr", to = "spm", wait = "after-each-chunk",
        chunks = {count = 64 : i64, bytes = 64 : i64,
          ddr_stride = 64 : i64, spm_stride = 64 : i64}}}
      : memref<64x64xi8, strided<[1, 64]>> to memref<64x64xi8>
    return
  }
}
"""
    row_major_dma = transposed_dma.replace("[1, 64]", "[64, 1]")

    rejected = [
        ("implicit_dma_transpose", transposed_dma),
        (
            "prematerialized_memory",
            row_major_dma.replace(
                "memref<64x64xi8, strided<[64, 1]>>",
                "memref<64x64xi8, strided<[64, 1]>, 1>",
            ).replace("memref<64x64xi8>", "memref<64x64xi8, 1>"),
        ),
        (
            "small_capacity",
            mixed.replace("capacity = 4096 : i64", "capacity = 4032 : i64", 1),
        ),
        (
            "bad_bank",
            mixed.replace(
                'channel = 1 : i64, from = "ddr"',
                'channel = 0 : i64, from = "ddr"',
                1,
            ),
        ),
        (
            "wrong_pitch",
            mixed.replace(
                "ddr_stride = 3072 : i64", "ddr_stride = 512 : i64", 1
            ),
        ),
        (
            "unaligned_bytes",
            norm.replace("bytes = 3072 : i64", "bytes = 3073 : i64", 1),
        ),
        (
            "wrong_direction",
            norm.replace(
                'from = "spm", to = "ddr"', 'from = "ddr", to = "spm"', 1
            ),
        ),
        (
            "weak_abi",
            direct.replace("args_aligned_bytes = 64", "args_aligned_bytes = 1"),
        ),
        (
            "nonmultiple_abi",
            direct.replace(
                "args_aligned_bytes = 64", "args_aligned_bytes = 96"
            ),
        ),
        (
            "missing_source_count",
            direct.replace("plan.source_task_count = 5 : i64,", ""),
        ),
        (
            "mapping_coverage",
            direct.replace(
                "plan.source_task_count = 5", "plan.source_task_count = 6"
            ),
        ),
        (
            "duplicate_source_mapping",
            direct.replace("old_id = 4 : i64", "old_id = 3 : i64"),
        ),
        (
            "invalid_source_target",
            direct.replace("new_id = 0 : i64", "new_id = 9 : i64", 1),
        ),
        (
            "unknown_plan",
            direct.replace("plan.route =", "plan.unsupported ="),
        ),
        (
            "nested_plan",
            direct.replace(
                "plan.target_profile =",
                "hidden = {plan.unsupported = true}, plan.target_profile =",
            ),
        ),
        (
            "wrong_plan_carrier",
            direct.replace("plan.route =", "plan.transfer ="),
        ),
        (
            "wrong_field_type",
            direct.replace("m = 64 : i64", "m = 64 : i32", 1),
        ),
        ("direct_pitch", direct.replace("[1024, 1]", "[1025, 1]")),
        (
            "subtile_alignment",
            direct.replace("k = 64 : i64}", "k = 32 : i64}").replace(
                "k = 2 : i64}", "k = 4 : i64}"
            ),
        ),
        ("missing_route", direct.replace("plan.route =", "discarded_route =")),
        (
            "wrong_route_binding",
            direct.replace(
                "b = {operand = 1 : i64", "b = {operand = 0 : i64", 1
            ),
        ),
        (
            "wrong_vector_extent",
            norm.replace(
                'elements = 1024 : i64, tail = "none"',
                'elements = 768 : i64, tail = "none"',
                1,
            ),
        ),
        (
            "wrong_reduction",
            norm.replace(
                "arith.addf %value, %acc : f32",
                "arith.addf %value, %value : f32",
                1,
            ),
        ),
        (
            "fast_reduction",
            norm.replace(
                "arith.addf %value, %acc : f32",
                "arith.addf %value, %acc fastmath<reassoc> : f32",
                1,
            ),
        ),
        ("missing_fill", norm.replace("plan.fill =", "discarded_fill =", 1)),
        (
            "physical_alias",
            norm.replace(
                'offset = 8192 : i64, capacity = 4096 : i64, role = "squares"',
                'offset = 4096 : i64, capacity = 4096 : i64, role = "squares"',
            ),
        ),
        (
            "unsupported_vector_payload",
            norm.replace(
                "arith.subf %lhs, %rhs", "arith.maximumf %lhs, %rhs", 1
            ),
        ),
        (
            "wrong_common_lane_count",
            norm.replace("lanes = 16 : i64", "lanes = 64 : i64", 1),
        ),
        (
            "zero_common_lane_count",
            norm.replace("lanes = 16 : i64", "lanes = 0 : i64", 1),
        ),
        (
            "wrong_common_lane_type",
            norm.replace("lanes = 16 : i64", "lanes = 16 : i32", 1),
        ),
        (
            "scalar_capture",
            norm.replace("arith.subf %lhs, %rhs", "arith.subf %lhs, %zero", 1),
        ),
    ]
    for name, source in rejected:
        result = invoke(name, source)
        assert result.returncode != 0, name
        assert "error:" in result.stderr, (name, result.stderr)
        assert "LLVM ERROR" not in result.stderr, (name, result.stderr)
        assert "PLEASE submit" not in result.stderr, (name, result.stderr)

    # A single chunk has no second start address. Its unused strides need not
    # satisfy the 64-byte pitch rule, while byte/element units still agree.
    single = norm.replace(
        "ddr_stride = 3072 : i64, spm_stride = 3072 : i64",
        "ddr_stride = 4 : i64, spm_stride = 4 : i64",
        1,
    )
    result = invoke("single_chunk_unused_pitch", single, emit=True)
    assert result.returncode == 0, result.stderr
    result = invoke("same_dma_axis", row_major_dma, emit=True)
    assert result.returncode == 0, result.stderr

    emitted = {}
    for name, source, tasks in (
        ("direct", direct, 1),
        ("mixed", mixed, 10),
        ("norm", norm, 21),
    ):
        result = invoke(name, source, emit=True)
        assert result.returncode == 0, result.stderr
        assert "plan." not in result.stdout
        assert "linalg." not in result.stdout
        assert result.stdout.count("contract.task =") == tasks
        if name in {"direct", "mixed"}:
            count = 5 if name == "direct" else 12
            assert (
                f"bosc_ame.source_task_count = {count} : i64" in result.stdout
            )
            assert "bosc_ame.task_map =" in result.stdout
        path = work / f"{name}.emitted.mlir"
        path.write_text(result.stdout)
        emitted[name] = path

    # Task IDs are function-local. Two kernels may both use physical task 0.
    function = direct[direct.index("  func.func") : direct.rfind("}")]
    multi = (
        direct[: direct.rfind("}")]
        + function.replace("@matmul128(", "@matmul128_second(")
        + "}\n"
    )
    result = invoke("multiple_functions", multi, emit=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("contract.task =") == 2
    assert result.stdout.count("bosc_ame.source_task_count = 5 : i64") == 2

    # Nonzero signed accumulator near wrap, last tile of the grid, and guards.
    previous = 0x7FFFFFF0
    a = bytearray([128]) * (128 * 128) + bytearray([0xDB]) * 64
    b = bytearray([255]) * (1024 * 128) + bytearray([0xBC]) * 64
    c = (
        bytearray(struct.pack("<i", previous) * (128 * 1024))
        + bytearray([0xCE]) * 64
    )
    expected = bytearray(c)
    updated = (previous + 128 * 128 + (1 << 31)) % (1 << 32) - (1 << 31)
    for row in range(64, 128):
        begin = (row * 1024 + 960) * 4
        expected[begin : begin + 256] = struct.pack("<i", updated) * 64
    saved_a, saved_b = bytes(a), bytes(b)
    machine = Machine(emitted["direct"], "matmul128")
    machine.launch(
        {"%arg0": a, "%arg1": b, "%arg2": c}, {"%arg3": 1, "%arg4": 15}
    )
    assert c == expected and bytes(a) == saved_a and bytes(b) == saved_b
    assert not machine.dma_log

    # The mixed plan carries C through six K blocks, reuses polluted SPM,
    # and reads B directly from DDR. Run the tail before the preceding full tile.
    rows, columns, depth = 196, 768, 768
    a = bytearray([128]) * (rows * depth) + bytearray([0xDB]) * 64
    b = bytearray([255]) * (columns * depth) + bytearray([0xBC]) * 64
    c = (
        bytearray(struct.pack("<i", previous) * (rows * columns))
        + bytearray([0xCE]) * 64
    )
    expected = bytearray(c)
    saved_a, saved_b = bytes(a), bytes(b)
    machine = Machine(emitted["mixed"], "triton_matmul_i8_196x768x768")
    updated = (previous + 128 * depth + (1 << 31)) % (1 << 32) - (1 << 31)
    for pid_m in (6, 5):
        machine.launch(
            {"%arg0": a, "%arg1": b, "%arg2": c},
            {"%arg6": pid_m, "%arg7": 5},
        )
        for row_index in range(pid_m * 32, min((pid_m + 1) * 32, rows)):
            begin = (row_index * columns + 640) * 4
            expected[begin : begin + 512] = struct.pack("<i", updated) * 128
        assert c == expected
    assert bytes(a) == saved_a and bytes(b) == saved_b
    assert len(machine.dma_log) == 288
    assert not any("%arg1" in entry[:2] for entry in machine.dma_log)

    # Row 1 exercises f32 descriptor offsets in both DMA directions. The DDR
    # offset is 768 elements, while the packet length is 3072 bytes.
    row = [float(i % 17 - 8) for i in range(768)]
    x = bytearray(struct.pack("<768f", *row) * 197) + bytearray([0xAB]) * 64
    weight = bytearray(struct.pack("<f", 1.25) * 768)
    bias = bytearray(struct.pack("<f", -0.5) * 768)
    output = bytearray([0xCD]) * (197 * 768 * 4 + 64)
    original_x = bytes(x)
    machine = Machine(emitted["norm"], "layernorm_197x768")
    machine.launch(
        {"%arg0": x, "%arg1": weight, "%arg2": bias, "%arg3": output},
        {"%arg4": 1},
    )
    mean = sum(row) / 768
    variance = sum((value - mean) ** 2 for value in row) / 768
    expected = [
        (value - mean) / math.sqrt(variance + 1.0e-12) * 1.25 - 0.5
        for value in row
    ]
    actual = struct.unpack_from("<768f", output, 3072)
    assert all(
        abs(a - b) <= 2e-5 + 2e-5 * abs(b) for a, b in zip(actual, expected)
    )
    assert output[:3072] == bytearray([0xCD]) * 3072
    assert output[6144:] == bytearray([0xCD]) * (len(output) - 6144)
    assert bytes(x) == original_x
    assert len(machine.dma_log) == 4
    assert all(entry[4] == 3072 for entry in machine.dma_log)
    assert machine.dma_log[0][2] == 3072
    assert machine.dma_log[-1][3] == 3072
    print(
        "PASS concrete planned emission: rejection boundaries, task anchors, signed wrap, DMA units, LayerNorm"
    )


if __name__ == "__main__":
    main()
