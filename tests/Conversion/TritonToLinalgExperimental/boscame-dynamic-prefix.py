# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Execute guarded dynamic-prefix DMA at zero, partial, and full extents."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path, experimental_pipeline  # noqa: E402


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    fixture = Path(__file__).with_name("dynamic-prefix-dma.mlir").read_text()
    module_start = fixture.index("\nmodule {") + 1
    input_path = work / "dynamic-prefix.ttir"
    input_path.write_text(fixture[module_start:])

    optimized = work / "dynamic-prefix.mlir"
    result = subprocess.run(
        [
            buddy_opt_path(),
            str(input_path),
            *experimental_pipeline(),
            "-o",
            str(optimized),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lowered = optimized.read_text()
    assert "bosc_ame.compute_graph_complete = true" in lowered
    assert "bosc_ame.compute_graph_node_count = 3 : i64" in lowered
    assert sorted(
        map(int, re.findall(r"bosc_ame.compute_id = (\d+)", lowered))
    ) == [
        0,
        1,
        2,
    ]
    # Guarded load/store task IDs belong to the final scf.if, which completes
    # the memory event even when all packet guards are false at extent zero.
    for task_id in (1, 2):
        assert re.search(
            rf"^\s*\}} \{{[^\n]*bosc_ame.compute_id = {task_id} : i64",
            lowered,
            re.M,
        ), lowered

    loops = work / "dynamic-prefix.loops.mlir"
    subprocess.run(
        [
            buddy_opt_path(),
            str(optimized),
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            str(loops),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    machine = Machine(loops, "dynamic_prefix_aligned")

    values = np.arange(32, dtype=np.float32) + 1.25
    source_guard = bytes([0xA5]) * 256
    destination_guard = bytes([0x5A]) * 256
    source = bytearray(values.tobytes() + source_guard)
    initial = np.full(32, -91.5, dtype=np.float32)
    destination = bytearray(initial.tobytes() + destination_guard)
    buffers = {"%arg0": source, "%arg1": destination}

    for limit, valid_elements in (
        (-1, 0),
        (0, 0),
        (16, 16),
        (32, 32),
        (48, 32),
    ):
        destination[:] = initial.tobytes() + destination_guard
        first_access = len(machine.accesses)
        machine.launch(buffers, {"%arg2": limit})

        valid_bytes = valid_elements * values.itemsize
        actual = np.frombuffer(destination, dtype=np.float32, count=32)
        np.testing.assert_array_equal(
            actual[:valid_elements], values[:valid_elements]
        )
        np.testing.assert_array_equal(
            actual[valid_elements:], initial[valid_elements:]
        )
        assert source == values.tobytes() + source_guard
        assert destination[-len(destination_guard) :] == destination_guard

        accesses = machine.accesses[first_access:]
        source_accesses = [entry for entry in accesses if entry[0] == "%arg0"]
        destination_accesses = [
            entry for entry in accesses if entry[0] == "%arg1"
        ]
        packet_count = valid_elements // 16
        assert len(source_accesses) == packet_count, source_accesses
        assert len(destination_accesses) == packet_count, destination_accesses
        assert len(accesses) == 2 * packet_count, accesses
        assert all(
            kind == "read" and 0 <= start < end <= valid_bytes
            for _, start, end, kind in source_accesses
        )
        assert all(
            kind == "write" and 0 <= start < end <= valid_bytes
            for _, start, end, kind in destination_accesses
        )

    # CHECK: PASS dynamic prefix: clamped zero, partial, and maximum extents preserve excluded DDR bytes
    print(
        "PASS dynamic prefix: clamped zero, partial, and maximum extents preserve excluded DDR bytes"
    )


if __name__ == "__main__":
    main()
