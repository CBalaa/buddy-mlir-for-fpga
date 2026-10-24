# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Large write/read chains retain complete content-order proofs without truncation."""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from boscame_check import buddy_opt_path  # noqa: E402


def check_bounds(work, name, operations, expected_edges):
    source = """module {
  func.func @bounded(%root: memref<16xf32>, %value: f32) {
    %c0 = arith.constant 0 : index
"""
    source += "\n".join(operations)
    source += "\n    return\n  }\n}\n"
    path = work / f"{name}.mlir"
    path.write_text(source)
    result = subprocess.run(
        [
            buddy_opt_path(),
            str(path),
            "--kernel-contract-annotate",
            "--kernel-contract-verify",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "contract.compute_graph = {complete = true" in result.stdout
    tasks = re.findall(r"contract\.task = \{([^{}]*)\}", result.stdout)
    assert len(tasks) == len(operations), (len(tasks), len(operations))
    edges = []
    for index, task in enumerate(tasks):
        assert f"id = {index} : i64" in task, task
        predecessors = re.search(
            r"predecessors = array<i64(?:: ([^>]+))?>", task
        )
        actual = (
            [int(value) for value in predecessors[1].split(",")]
            if predecessors[1]
            else []
        )
        # A WAW chain orders all stores transitively. Read/read needs no edge.
        assert actual == ([index - 1] if expected_edges and index else []), task
        edges.extend((predecessor, index) for predecessor in actual)
    assert len(edges) == expected_edges, (len(edges), expected_edges)


work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)

# The retired pairwise report truncated more than 8192 conflicts. The neutral
# graph keeps the equivalent transitive WAW chain and verifies every task.
check_bounds(
    work,
    "stores",
    ["    memref.store %value, %root[%c0] : memref<16xf32>"] * 130,
    129,
)
print("complete memory dependency stress: 130 stores, 129 WAW edges")

# Keep the original read-only stress size (66430 pairs in the old analysis).
# Every read must retain its captured version without false read/read hazards.
check_bounds(
    work,
    "reads",
    [
        f"    %read{index} = memref.load %root[%c0] : memref<16xf32>"
        for index in range(365)
    ],
    0,
)
print("complete memory dependency stress: 365 reads, zero read/read edges")

# CHECK: complete memory dependency stress: 130 stores, 129 WAW edges
# CHECK: complete memory dependency stress: 365 reads, zero read/read edges
