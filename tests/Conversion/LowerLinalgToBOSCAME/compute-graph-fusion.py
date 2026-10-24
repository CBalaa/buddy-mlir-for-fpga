# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Accumulator fusion contracts metadata and rejects corrupt complete graphs."""

import re
import runpy
import struct
import subprocess
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
    original = fusion["lower_source"](
        work, "legacy_pitched_graph", source(), plan_report=True
    )
    assert original.returncode != 0
    assert "no initialized content version" in original.stderr, original.stderr

    # Neutral graph integrity is checked independently from legacy metadata.
    # Legacy task/value IDs are derived facts and are rebuilt by the adapter.
    buffer = runpy.run_path(
        str(Path(__file__).with_name("buffer-accumulator-fusion.py"))
    )
    standard = (
        buffer["SOURCE"]
        .replace("@GAP@", "")
        .replace(", bosc_ame.lower_to_ame", "")
        .replace(
            "module {",
            "module attributes {contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}} {",
        )
        .replace(
            "linalg.matmul {cast",
            "linalg.matmul {contract.provenance = {a_source_load_origin_ids = array<i64: 42>, a_source_lineage_complete = true}, cast",
        )
    )
    opt = str(fusion["buddy_opt_path"]())

    def invoke(text, *passes):
        return subprocess.run(
            [opt, *passes], input=text, text=True, capture_output=True
        )

    annotated = invoke(standard, "--kernel-contract-annotate")
    assert annotated.returncode == 0, annotated.stderr
    assert annotated.stdout.count("contract.task =") == 3
    planned = invoke(
        annotated.stdout, "--bosc-ame-plan", "--bosc-ame-plan-verify"
    )
    assert planned.returncode == 0, planned.stderr
    assert planned.stdout.count("contract.task =") == 1
    assert "plan.source_task_count = 3 : i64" in planned.stdout
    assert 'action = "eliminated", old_id = 0 : i64' in planned.stdout
    assert (
        'action = "retained", new_id = 0 : i64, old_id = 1 : i64'
        in planned.stdout
    )
    assert 'action = "eliminated", old_id = 2 : i64' in planned.stdout
    assert "input_values = array<i64: 0, 1, 2>" in planned.stdout
    assert "source_load_origin_ids = array<i64: 42>" in planned.stdout
    emitted = invoke(planned.stdout, "--bosc-ame-emit")
    assert emitted.returncode == 0, emitted.stderr
    path = work / "neutral-fused.mlir"
    path.write_text(emitted.stdout)
    previous = 0x7FFFFFF0
    output = bytearray(struct.pack("<i", previous) * 256)
    machine = fusion["Machine"](path, "f")
    machine.launch(
        {
            "%arg0": bytearray([128] * 1024),
            "%arg1": bytearray([255] * 1024),
            "%arg2": output,
        }
    )
    expected = (previous + 64 * 128 + (1 << 31)) % (1 << 32) - (1 << 31)
    assert output == struct.pack("<i", expected) * 256
    assert not machine.dma_log

    # Schema checks apply before graph contraction and in incomplete graphs.
    corruptions = {
        "duplicate_task": re.sub(
            r"\bid = 1 : i64", "id = 0 : i64", annotated.stdout, count=1
        ),
        "dangling_predecessor": annotated.stdout.replace(
            "predecessors = array<i64: 0>", "predecessors = array<i64: 99>", 1
        ),
        "wrong_input_version": annotated.stdout.replace(
            "input_values = array<i64>", "input_values = array<i64: 999>", 1
        ),
        "self_dependency": annotated.stdout.replace(
            "predecessors = array<i64>", "predecessors = array<i64: 0>", 1
        ),
    }
    for name, broken in corruptions.items():
        assert broken != annotated.stdout, name
        checked = invoke(broken, "--kernel-contract-verify")
        assert checked.returncode != 0, name
        assert "kernel contract" in checked.stderr, checked.stderr
        incomplete = broken.replace(
            "complete = true",
            'complete = false, limitation = "unmodeled boundary"',
            1,
        )
        checked = invoke(incomplete, "--kernel-contract-verify")
        assert checked.returncode != 0, name
        assert "kernel contract" in checked.stderr, checked.stderr
    # CHECK: PASS accumulator compute graph: contraction, boundary values and integrity
    print(
        "PASS accumulator compute graph: contraction, boundary values and integrity"
    )


if __name__ == "__main__":
    main()
