# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Bound memory-order analysis even when it produces no dependency edges."""

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
        [buddy_opt_path(), str(path), "--linalg-to-boscame=plan-report=true"],
        capture_output=True,
        text=True,
        check=True,
    )
    header = re.search(
        r"memory-dependencies: count=(\d+) scope=lexical status=(\w+)",
        result.stderr,
    )
    assert header, result.stderr
    assert header.groups() == (str(expected_edges), "bounded"), header.group(0)
    edges = re.findall(
        r"memory-dependency\[(\d+)\]: function=@bounded "
        r"accesses=\[(\d+),(\d+)\] kind=(\S+)",
        result.stderr,
    )
    assert len(edges) == expected_edges, (len(edges), expected_edges)
    for index, (edge_index, before, after, kind) in enumerate(edges):
        assert int(edge_index) == index
        assert 0 <= int(before) < int(after) < len(operations), (before, after)
        assert kind == "WAW", kind


work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)

# 130 stores produce more than 8192 conflicts: retain a deterministic bounded
# prefix and expose that full coverage has not been established.
check_bounds(
    work,
    "stores",
    ["    memref.store %value, %root[%c0] : memref<16xf32>"] * 130,
    8192,
)
print("memory dependency edge limit: bounded at 8192")

# 365 reads have 66430 pairs. Read/read pruning must not bypass the analysis
# work budget merely because the retained dependency set remains empty.
check_bounds(
    work,
    "reads",
    [
        f"    %read{index} = memref.load %root[%c0] : memref<16xf32>"
        for index in range(365)
    ],
    0,
)
print("memory dependency comparison limit: bounded with zero edges")

# CHECK: memory dependency edge limit: bounded at 8192
# CHECK: memory dependency comparison limit: bounded with zero edges
