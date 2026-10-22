# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Accumulator fusion contracts metadata and rejects corrupt complete graphs."""

import re
import runpy
import struct
import sys
from pathlib import Path

fusion = runpy.run_path(str(Path(__file__).with_name("direct-ame-fusion.py")))


def ids(values):
    return (
        "array<i64"
        + (": " + ", ".join(map(str, values)) if values else "")
        + ">"
    )


def task(number, kind, inputs, outputs, predecessors=(), barrier=False):
    return ", ".join(
        [
            f"bosc_ame.compute_id = {number} : i64",
            f'bosc_ame.compute_kind = "{kind}"',
            "bosc_ame.compute_stage = 0 : i64",
            f"bosc_ame.compute_input_values = {ids(inputs)}",
            f"bosc_ame.compute_output_values = {ids(outputs)}",
            f"bosc_ame.compute_predecessors = {ids(predecessors)}",
            f"bosc_ame.compute_barrier_after = {str(barrier).lower()}",
            "bosc_ame.rematerializable = false",
            "bosc_ame.varying_program_axes = array<i64: 0, 1>",
        ]
    )


def source(neighbors=False):
    text = fusion["BUFFER_SOURCE"]
    defaults = {
        "ALIASES": "",
        "PRODUCT_BEFORE_LOAD": "",
        "PRODUCT_AFTER_LOAD": "%product = memref.alloc() alignment = 64 : memref<64x64xi32>",
        "LOAD_GAP": "",
        "STORE_GAP": "",
        "BRIDGES": "",
        "STORE_SOURCE": "%bridge",
        "CONSUMERS": "",
    }
    for key, value in defaults.items():
        text = text.replace(f"@{key}@", value)
    shift = int(neighbors)
    load, fill, matmul, add, store = range(shift, shift + 5)
    text = text.replace(
        "module attributes {",
        "module attributes {bosc_ame.compute_graph_version = 1 : i32, ",
    ).replace(
        "%program: i32) {",
        "%program: i32) attributes {bosc_ame.compute_graph_complete = true, "
        f"bosc_ame.compute_graph_node_count = {7 if neighbors else 5} : i64}} {{",
    )
    annotations = {
        "memref.copy %destination, %snapshot {": task(
            load, "memory-load", [4], [6], barrier=True
        ),
        "linalg.fill ": task(fill, "fill", [], [9]),
        "linalg.matmul {": task(
            matmul,
            "ame-matmul",
            [20 if neighbors else 1, 3, 9],
            [10],
            [0, fill] if neighbors else [fill],
            barrier=True,
        )
        + ", bosc_ame.logical_value_ids = array<i64: 10>, "
        f"bosc_ame.logical_value_users = [{ids([add])}]",
        "linalg.generic {": task(
            add, "elementwise", [7, 10], [11], [load, matmul]
        ),
        "memref.copy %bridge, %destination {": task(
            store, "memory-store", [12], [4], [add], barrier=True
        ),
    }
    for spelling, metadata in annotations.items():
        replacement = (
            spelling + metadata + ", "
            if spelling.endswith("{")
            else spelling + "{" + metadata + "} "
        )
        text = text.replace(spelling, replacement, 1)
    users = [load, 6] if neighbors else [load]
    text = text.replace(
        "strides: [1024, 1] : memref<65536xi32>",
        "strides: [1024, 1] {bosc_ame.logical_value_ids = array<i64: 4>, "
        f"bosc_ame.logical_value_users = [{ids(users)}]}} : memref<65536xi32>",
    )
    if neighbors:
        text = text.replace(
            "%program: i32)", "%observed: memref<64x64xi32>, %program: i32)"
        )
        upstream = (
            "    %one = arith.constant 1 : i8\n"
            "    linalg.fill {"
            + task(0, "fill", [], [20], barrier=True)
            + "} ins(%one : i8) outs(%a : memref<64x128xi8, strided<[128, 1]>>)\n"
        )
        text = text.replace(
            "    %snapshot_base =", upstream + "    %snapshot_base ="
        )
        downstream = (
            "    memref.copy %destination, %observed {"
            + task(6, "memory-store", [4], [21], [store])
            + "} : memref<64x64xi32, strided<[1024, 1], offset: ?>> "
            "to memref<64x64xi32>\n"
        )
        text = text.replace("    return", downstream + "    return")
    return text


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def lower(name, text):
        return fusion["lower_source"](work, name, text, plan_report=True)

    plain = source()
    result = lower("fused_graph", plain)
    fusion["check_success"](result, fused=True)
    assert (
        "status=complete tasks=1 dependencies=0 logical-values=4"
        in result.stderr
    )
    assert (
        "task[2]: op=linalg.matmul kind=ame-matmul stage=0 predecessors=[] "
        "inputs=[1,3,4] outputs=[13]"
    ) in result.stderr, result.stderr
    assert (
        "logical-value[4]: producer=function-argument consumers=[2]"
        in result.stderr
    )
    assert "logical-value[13]: producer=2 consumers=[]" in result.stderr
    assert "bosc_ame.compute_graph_node_count = 1 : i64" in result.stdout
    assert "bosc_ame.logical_value_users = [array<i64: 2>]" in result.stdout

    # The graph annotation must preserve signed i8 accumulation and i32 wrap.
    lowered = work / "fused_graph.lowered.mlir"
    lowered.write_text(result.stdout)
    previous = 0x7FFFFFF0
    output = bytearray(struct.pack("<i", previous) * 65536)
    machine = fusion["Machine"](lowered, "gemm")
    machine.launch(
        {
            "%arg0": bytearray([128] * 8192),
            "%arg1": bytearray([255] * 8192),
            "%arg2": output,
        },
        {"%arg3": 0},
    )
    expected = (previous + 128 * 128 + (1 << 31)) % (1 << 32) - (1 << 31)
    for i in range(64):
        for j in range(1024):
            actual = struct.unpack_from("<i", output, (i * 1024 + j) * 4)[0]
            assert actual == (expected if j < 64 else previous)
    assert machine.dma_log == []

    chained = lower("neighbor_graph", source(neighbors=True))
    fusion["check_success"](chained, fused=True)
    assert "status=complete tasks=3 dependencies=2" in chained.stderr, (
        chained.stderr
    )
    assert (
        "task[3]: op=linalg.matmul kind=ame-matmul stage=1 predecessors=[0] "
        "inputs=[20,3,4] outputs=[22]"
    ) in chained.stderr, chained.stderr
    assert "stage=2 predecessors=[3] inputs=[22] outputs=[21]" in chained.stderr
    assert "logical-value[22]: producer=3 consumers=[6]" in chained.stderr
    assert "bosc_ame.logical_value_users = [array<i64: 3>]" in chained.stdout
    assert "array<i64: 3, 6>" not in chained.stdout

    # An earlier writer of C is a predecessor of the removed load, so its
    # dependency must be inherited by the matmul as well.
    c_writer = (
        source(neighbors=True)
        .replace(
            task(0, "fill", [], [20], barrier=True),
            task(0, "fill", [], [4], barrier=True),
        )
        .replace(
            task(1, "memory-load", [4], [6], barrier=True),
            task(1, "memory-load", [4], [6], [0], barrier=True),
        )
    )
    c_writer = (
        c_writer.replace(
            "compute_input_values = array<i64: 20, 3, 9>",
            "compute_input_values = array<i64: 1, 3, 9>",
        )
        .replace(
            "compute_predecessors = array<i64: 0, 2>",
            "compute_predecessors = array<i64: 2>",
        )
        .replace(
            "%one = arith.constant 1 : i8", "%one = arith.constant 1 : i32"
        )
        .replace(
            "ins(%one : i8) outs(%a : memref<64x128xi8, strided<[128, 1]>>)",
            "ins(%one : i32) outs(%destination : memref<64x64xi32, strided<[1024, 1], offset: ?>>)",
        )
    )
    inherited = lower("c_writer_graph", c_writer)
    fusion["check_success"](inherited, fused=True)
    assert "status=complete tasks=3 dependencies=2" in inherited.stderr
    assert "stage=1 predecessors=[0] inputs=[1,3,4]" in inherited.stderr
    assert "logical-value[4]: producer=0 consumers=[3]" in inherited.stderr

    cases = {
        "dangling": (
            plain.replace(
                "bosc_ame.compute_predecessors = array<i64: 1>",
                "bosc_ame.compute_predecessors = array<i64: 99>",
            ),
            "unresolved BOSCAME compute predecessor 99 for task 2",
        ),
        "duplicate": (
            plain.replace(
                "bosc_ame.compute_id = 1 :", "bosc_ame.compute_id = 0 :"
            ),
            "duplicate BOSCAME compute task ID 0",
        ),
        "count": (
            plain.replace(
                "compute_graph_node_count = 5", "compute_graph_node_count = 6"
            ),
            "BOSCAME compute graph node count mismatch: expected 6, found 5",
        ),
        "cycle": (
            plain.replace(
                "bosc_ame.compute_predecessors = array<i64: 1>",
                "bosc_ame.compute_predecessors = array<i64: 2>",
            ),
            "invalid BOSCAME compute predecessor order: 2 -> 2",
        ),
    }
    for name, (text, diagnostic) in cases.items():
        broken = lower(name, text)
        assert broken.returncode != 0, name
        assert diagnostic in broken.stderr, broken.stderr
        assert "BOSCAME plan report:" not in broken.stderr

    incomplete = lower(
        "incomplete_graph",
        plain.replace(
            "compute_graph_complete = true", "compute_graph_complete = false"
        ),
    )
    fusion["check_success"](incomplete, fused=True)
    assert "status=incomplete tasks=1 dependencies=0" in incomplete.stderr
    assert not re.search(r"dependency:.*resolved=false", incomplete.stderr)
    # CHECK: PASS accumulator compute graph: contraction, boundary values and integrity
    print(
        "PASS accumulator compute graph: contraction, boundary values and integrity"
    )


if __name__ == "__main__":
    main()
