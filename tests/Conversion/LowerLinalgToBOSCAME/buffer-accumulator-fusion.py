# RUN: %PYTHON %s %t
"""Private i32 product fusion is independent of the storage route."""

import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402

SOURCE = """module {
  func.func @f(%a: memref<16x64xi8>, %b: memref<64x16xi8, strided<[1,64]>>, %c: memref<16x16xi32>) {
    %p = memref.alloc() alignment = 64 : memref<16x16xi32>
    %z = arith.constant 0 : i32
    %i = arith.constant 0 : index
    linalg.fill ins(%z : i32) outs(%p : memref<16x16xi32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<16x64xi8>, memref<64x16xi8, strided<[1,64]>>)
      outs(%p : memref<16x16xi32>)
    @GAP@
    linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%c, %p : memref<16x16xi32>, memref<16x16xi32>) outs(%c : memref<16x16xi32>) {
    ^bb0(%c0: i32, %p0: i32, %out: i32):
      %sum = arith.addi %c0, %p0 : i32
      linalg.yield %sum : i32
    }
    memref.dealloc %p : memref<16x16xi32>
    return
  }
}
"""


def ids(values):
    return (
        "array<i64"
        + (": " + ", ".join(map(str, values)) if values else "")
        + ">"
    )


def task(number, inputs, outputs, predecessors, stage, axes=(), barrier=False):
    return ", ".join(
        [
            f"bosc_ame.compute_id = {number} : i64",
            'bosc_ame.compute_kind = "ame-matmul"'
            if number == 1
            else 'bosc_ame.compute_kind = "elementwise"',
            f"bosc_ame.compute_input_values = {ids(inputs)}",
            f"bosc_ame.compute_output_values = {ids(outputs)}",
            f"bosc_ame.compute_predecessors = {ids(predecessors)}",
            f"bosc_ame.compute_stage = {stage} : i64",
            f"bosc_ame.varying_program_axes = {ids(axes)}",
            f"bosc_ame.compute_barrier_after = {str(barrier).lower()}",
            "bosc_ame.rematerializable = true",
            f"bosc_ame.logical_value_ids = {ids(outputs)}",
        ]
    )


def graph_source():
    source = SOURCE.replace("@GAP@", "").replace(
        "module {",
        "module attributes {bosc_ame.compute_graph_version = 1 : i32} {",
    )
    for value, number, users in (
        ("a", 10, [1]),
        ("b", 11, [1]),
        ("c", 12, [2]),
    ):
        shape = {
            "a": "16x64xi8",
            "b": "64x16xi8, strided<[1,64]>",
            "c": "16x16xi32",
        }[value]
        source = source.replace(
            f"%{value}: memref<{shape}>",
            f"%{value}: memref<{shape}> {{bosc_ame.logical_value_id = {number} : i64, "
            f"bosc_ame.logical_value_users = {ids(users)}}}",
            1,
        )
    source = source.replace(
        "}) {",
        "}, %d: memref<16x16xi32> {bosc_ame.logical_value_id = 23 : i64, "
        "bosc_ame.logical_value_users = array<i64>}) "
        "attributes {bosc_ame.compute_graph_complete = true, "
        "bosc_ame.compute_graph_node_count = 4 : i64} {",
        1,
    )
    source = source.replace(
        "linalg.fill ins",
        "linalg.fill {"
        + task(0, [], [20], [], 0, [0], barrier=True)
        + ", bosc_ame.logical_value_users = [array<i64: 1>], "
        "bosc_ame.source_load_origin_ids = array<i64: 41>} ins",
        1,
    )
    source = source.replace(
        "linalg.matmul {",
        "linalg.matmul {"
        + task(1, [10, 11, 20], [21], [0], 1, [1])
        + ", bosc_ame.logical_value_users = [array<i64: 2>], "
        "bosc_ame.source_load_origin_ids = array<i64: 42>, "
        "bosc_ame.invariant_program_axes = array<i64: 0, 2, 3>, ",
        1,
    )
    source = source.replace(
        "linalg.generic {",
        "linalg.generic {"
        + task(2, [12, 21], [22], [1], 1, [2], barrier=True)
        + ", bosc_ame.logical_value_users = [array<i64: 3>], "
        "bosc_ame.source_load_origin_ids = array<i64: 43>, ",
        1,
    )
    source = source.replace(
        "    memref.dealloc",
        "    memref.copy %c, %d {"
        + task(3, [22], [23], [2], 2)
        + ", bosc_ame.logical_value_users = [array<i64>]} "
        ": memref<16x16xi32> to memref<16x16xi32>\n    memref.dealloc",
        1,
    )
    return source


def alias_source(offset):
    return (
        SOURCE.replace("@GAP@", "")
        .replace(
            "func.func @f(%a: memref<16x64xi8>, %b: memref<64x16xi8, strided<[1,64]>>, %c: memref<16x16xi32>)",
            "func.func @f(%backing: memref<2048xi8>, %b: memref<64x16xi8, strided<[1,64]>>)",
        )
        .replace(
            "    %i = arith.constant 0 : index",
            "    %i = arith.constant 0 : index\n"
            f"    %offset = arith.constant {offset} : index\n"
            "    %a = memref.view %backing[%i][] : memref<2048xi8> to memref<16x64xi8>\n"
            "    %c = memref.view %backing[%offset][] : memref<2048xi8> to memref<16x16xi32>",
        )
    )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = buddy_opt_path()

    def lower(name, source, diagnostic=None):
        input_path = work / (name + ".mlir")
        input_path.write_text(source)
        result = subprocess.run(
            [
                opt,
                str(input_path),
                "--linalg-to-boscame=enable-spm-dma=true enable-ame=true "
                "dma-abi=args_aligned64_nonalias plan-report=true",
            ],
            capture_output=True,
            text=True,
        )
        if diagnostic is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0
            assert diagnostic in result.stderr, result.stderr
        return result

    for name, source in (
        ("adjacent", SOURCE.replace("@GAP@", "")),
        (
            "swapped",
            SOURCE.replace("@GAP@", "").replace(
                "arith.addi %c0, %p0", "arith.addi %p0, %c0"
            ),
        ),
    ):
        result = lower(name, source)
        assert "memref.alloc" not in result.stdout
        assert "arith.addi" not in result.stdout
        assert "bosc_ame.tile.mma" in result.stdout
        path = work / (name + ".lowered.mlir")
        path.write_text(result.stdout)
        previous = 0x7FFFFFF0
        output = bytearray(struct.pack("<i", previous) * 256)
        machine = Machine(path, "f")
        machine.launch(
            {
                "%arg0": bytearray([128] * 1024),
                "%arg1": bytearray([255] * 1024),
                "%arg2": output,
            },
            {},
        )
        expected = (previous + 64 * 128 + (1 << 31)) % (1 << 32) - (1 << 31)
        assert output == struct.pack("<i", expected) * 256

    # The runtime effect, additional product read, nonzero init and overflow
    # semantics each prevent contracting the chain.
    rejected = {
        "extra_consumer": SOURCE.replace(
            "@GAP@", "%observe = memref.load %p[%i, %i] : memref<16x16xi32>"
        ),
        "effect": SOURCE.replace(
            "module {", "module { func.func private @effect()"
        ).replace("@GAP@", "func.call @effect() : () -> ()"),
        "nonzero": SOURCE.replace("@GAP@", "").replace(
            "arith.constant 0 : i32", "arith.constant 1 : i32"
        ),
        "overflow": SOURCE.replace("@GAP@", "").replace(
            "arith.addi %c0, %p0", "arith.addi %c0, %p0 overflow<nsw>"
        ),
        "alias": alias_source(0),
        "partial_overlap": alias_source(64),
    }
    for name, source in rejected.items():
        diagnostic = {
            "extra_consumer": "scalar load must access planned host rank-zero storage",
            "effect": "call may clobber live SPM workspace",
        }.get(name)
        result = lower(name, source, diagnostic)
        if diagnostic is None:
            assert "arith.addi" in result.stdout, (name, result.stdout)
            assert "bosc_ame.tile.mma" in result.stdout

    # Preserve the old external post-store read as a precise v1 boundary.
    lower(
        "original_graph",
        graph_source(),
        "reading an external region after its terminal write",
    )
    graph_input = re.sub(r"    memref.copy %c, %d[^\n]*\n", "", graph_source())
    graph_input = graph_input.replace(
        "linalg.matmul {",
        "linalg.matmul {bosc_ame.a_source_load_origin_ids = array<i64: 42>, "
        "bosc_ame.a_source_lineage_complete = true, ",
    )
    graph = lower("rebuilt_graph", graph_input)
    assert "bosc_ame.compute_id" not in graph.stdout
    assert "bosc_ame.source_task_count = 3 : i64" in graph.stdout
    assert 'action = "eliminated", old_id = 0 : i64' in graph.stdout
    assert (
        'action = "retained", new_id = 0 : i64, old_id = 1 : i64'
        in graph.stdout
    )
    assert 'action = "eliminated", old_id = 2 : i64' in graph.stdout
    assert "input_values = array<i64: 0, 1, 2>" in graph.stdout
    assert "source_load_origin_ids = array<i64: 42>" in graph.stdout
    # Old graph metadata is derived, including stale IDs and ordering. It may
    # not prevent legal semantic fusion or override the actual use-def graph.
    corrupt = graph_input.replace(
        "bosc_ame.compute_predecessors = array<i64: 0>",
        "bosc_ame.compute_predecessors = array<i64: 99>",
    )
    rebuilt = lower("legacy_metadata_rebuilt", corrupt)
    assert rebuilt.stdout == graph.stdout

    print(
        "PASS buffer accumulator: private fusion, swapped inputs, wrap and rejection boundaries"
    )


if __name__ == "__main__":
    main()
