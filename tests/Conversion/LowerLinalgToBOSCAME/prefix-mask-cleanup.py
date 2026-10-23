# RUN: %PYTHON %s buddy-opt
"""Prefix specialization prunes only private, unused mask computations."""

import runpy
import subprocess
import sys
from pathlib import Path

graph = runpy.run_path(
    str(Path(__file__).with_name("buffer-placement-metadata.py"))
)["graph"]
OPTIONS = (
    "--linalg-to-boscame=buffer-placement=true enable-spm-dma=true "
    "dma-abi=args_aligned64_nonalias store-policy=spm-dma"
)


def ids(values):
    return (
        "array<i64"
        + (": " + ", ".join(map(str, values)) if values else "")
        + ">"
    )


def metadata(
    number, kind, inputs, output, predecessors=(), users=(), barrier=False
):
    return ", ".join(
        [
            f"bosc_ame.compute_id = {number} : i64",
            f'bosc_ame.compute_kind = "{kind}"',
            "bosc_ame.compute_stage = 0 : i64",
            f"bosc_ame.compute_input_values = {ids(inputs)}",
            f"bosc_ame.compute_output_values = {ids([output])}",
            f"bosc_ame.compute_predecessors = {ids(predecessors)}",
            f"bosc_ame.compute_barrier_after = {str(barrier).lower()}",
            "bosc_ame.rematerializable = true",
            f"bosc_ame.logical_value_ids = {ids([output])}",
            f"bosc_ame.logical_value_users = [{ids(users)}]",
        ]
    )


def source(case):
    observed = case in {"other_consumer", "alias_consumer", "ordering_consumer"}
    width = "%external_i32" if case == "external_width" else "%width"
    channel = "%external_i32" if case == "external_range" else "%channel"
    mask = "%external_mask" if case == "external_mask" else "%mask"
    mask_users = [3, 4] if case in {"other_consumer", "alias_consumer"} else [3]
    declarations = []
    for name, element in [("width", "i32"), ("channel", "i32"), ("mask", "i1")]:
        if {"width": width, "channel": channel, "mask": mask}[
            name
        ] == f"%{name}":
            declarations.append(
                f"%{name} = memref.alloc() alignment = 64 : memref<32x{element}>"
            )
    extra = ""
    if observed:
        if case == "ordering_consumer":
            extra = (
                "linalg.fill {"
                + metadata(4, "fill", [], 6, [2])
                + "} ins(%prefix : i32) outs(%external_i32 : memref<32xi32>)"
            )
        else:
            input_name, input_type = mask, "memref<32xi1>"
            if case == "alias_consumer":
                extra = (
                    f"%alias = memref.cast {mask} : memref<32xi1> "
                    "to memref<?xi1>\n"
                )
                input_name, input_type = "%alias", "memref<?xi1>"
            extra += (
                "linalg.generic {indexing_maps = [#id, #id], "
                'iterator_types = ["parallel"]} '
                f"ins({input_name} : {input_type}) "
                "outs(%observed : memref<32xi1>) attrs = {"
                + metadata(4, "elementwise", [3], 6, [2])
                + "} {\n"
                "^bb0(%value: i1, %old: i1):\n"
                "  linalg.yield %value : i1\n}"
            )
    declarations_text = "\n    ".join(declarations)
    module = f"""
#id = affine_map<(i) -> (i)>
module attributes {{bosc_ame.compute_graph_version = 1 : i32}} {{
  func.func @prefix_mask(%external_i32: memref<32xi32>,
                         %external_mask: memref<32xi1>,
                         %observed: memref<32xi1>)
      attributes {{bosc_ame.compute_graph_complete = true,
                   bosc_ame.compute_graph_node_count = {5 if observed else 4} : i64}} {{
    %prefix = arith.constant 16 : i32
    %false = arith.constant false
    {declarations_text}
    %data = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zeros = bosc_ame.spm.view {{bank = 1 : i32, offset = 0 : i64}}
      : memref<32xf32, #bosc_ame.spm<1>>
    %out = bosc_ame.spm.view {{bank = 0 : i32, offset = 128 : i64}}
      : memref<32xf32, #bosc_ame.spm<0>>
    linalg.fill {{{metadata(0, "fill", [], 1, users=[2])},
      bosc_ame.prefix_mask_aux}}
      ins(%prefix : i32) outs({width} : memref<32xi32>)
    linalg.generic {{indexing_maps = [#id], iterator_types = ["parallel"]}}
      outs({channel} : memref<32xi32>)
      attrs = {{{metadata(1, "elementwise", [], 2, users=[2])},
                bosc_ame.prefix_range}} {{
    ^bb0(%old: i32):
      %index = linalg.index 0 : index
      %integer = arith.index_cast %index : index to i32
      linalg.yield %integer : i32
    }}
    linalg.generic {{indexing_maps = [#id, #id, #id],
                     iterator_types = ["parallel"]}}
      ins({channel}, {width} : memref<32xi32>, memref<32xi32>)
      outs({mask} : memref<32xi1>)
      attrs = {{{metadata(2, "elementwise", [2, 1], 3, [0, 1], mask_users, barrier=case == "barrier_mask")},
                bosc_ame.prefix_mask_aux}} {{
    ^bb0(%index: i32, %bound: i32, %old: i1):
      %valid = arith.cmpi slt, %index, %bound : i32
      linalg.yield %valid : i1
    }}
    linalg.generic {{indexing_maps = [#id, #id, #id, #id],
                     iterator_types = ["parallel"]}}
      ins({mask}, %data, %zeros : memref<32xi1>,
        memref<32xf32, #bosc_ame.spm<0>>, memref<32xf32, #bosc_ame.spm<1>>)
      outs(%out : memref<32xf32, #bosc_ame.spm<0>>)
      attrs = {{{metadata(3, "elementwise", [3, 10, 11], 5, [2])},
                bosc_ame.zero_tail_select_prefix = 16 : i64}} {{
    ^bb0(%valid: i1, %value: f32, %zero: f32, %old: f32):
      %selected = arith.select %valid, %value, %zero : f32
      linalg.yield %selected : f32
    }}
    {extra}
    return
  }}
}}
"""
    if case == "dead_alloca":
        module = module.replace("memref.alloc()", "memref.alloca()")
    elif case == "dead_dealloc":
        module = module.replace(
            "    return",
            "    memref.dealloc %mask : memref<32xi1>\n"
            "    memref.dealloc %channel : memref<32xi32>\n"
            "    memref.dealloc %width : memref<32xi32>\n"
            "    return",
        )
    elif case == "condition_last":
        module = module.replace(
            "ins(%mask, %data, %zeros : memref<32xi1>,\n"
            "        memref<32xf32, #bosc_ame.spm<0>>, memref<32xf32, #bosc_ame.spm<1>>)",
            "ins(%data, %zeros, %mask : memref<32xf32, #bosc_ame.spm<0>>,\n"
            "        memref<32xf32, #bosc_ame.spm<1>>, memref<32xi1>)",
        ).replace(
            "^bb0(%valid: i1, %value: f32, %zero: f32, %old: f32):",
            "^bb0(%value: f32, %zero: f32, %valid: i1, %old: f32):",
        )
        module = module.replace(
            "compute_input_values = array<i64: 3, 10, 11>",
            "compute_input_values = array<i64: 10, 11, 3>",
        )
    elif case == "effectful_range":
        module = module.replace(
            "      linalg.yield %integer : i32",
            "      memref.store %integer, %external_i32[%index] : memref<32xi32>\n"
            "      linalg.yield %integer : i32",
        )
    return module


def main():
    binary = sys.argv[1]
    for case, surviving in [
        ("dead_local", {3}),
        ("dead_alloca", {3}),
        ("dead_dealloc", {3}),
        ("condition_last", {3}),
        ("other_consumer", {0, 1, 2, 3, 4}),
        ("alias_consumer", {0, 1, 2, 3, 4}),
        ("ordering_consumer", {0, 1, 2, 3, 4}),
        ("external_mask", {0, 1, 2, 3}),
        ("external_width", {0, 3}),
        ("external_range", {1, 3}),
        ("effectful_range", {1, 3}),
        ("barrier_mask", {0, 1, 2, 3}),
    ]:
        result = subprocess.run(
            [binary, "-", OPTIONS],
            input=source(case),
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (case, result.stderr)
        tasks, _, counts, _ = graph(result.stdout)
        assert set(tasks) == surviving, (case, set(tasks), surviving)
        assert counts == [f"{len(surviving)} : i64"], (case, counts)
        assert (
            tasks[3]["bosc_ame.compute_input_values"] == "array<i64: 10, 11>"
        ), case
        expected_predecessors = (
            "array<i64: 2>" if case == "barrier_mask" else "array<i64>"
        )
        assert (
            tasks[3]["bosc_ame.compute_predecessors"] == expected_predecessors
        ), case
        if case in {"other_consumer", "alias_consumer"}:
            assert (
                tasks[2]["bosc_ame.logical_value_users"] == "[array<i64: 4>]"
            ), case
            assert (
                tasks[4]["bosc_ame.compute_input_values"] == "array<i64: 3>"
            ), case
            assert (
                tasks[4]["bosc_ame.compute_predecessors"] == "array<i64: 2>"
            ), case
        elif 2 in surviving:
            assert tasks[2]["bosc_ame.logical_value_users"] == "[array<i64>]", (
                case
            )
        for producer in surviving & {0, 1}:
            users = "[array<i64: 2>]" if 2 in surviving else "[array<i64>]"
            assert tasks[producer]["bosc_ame.logical_value_users"] == users, (
                case
            )
        if case in {
            "dead_local",
            "dead_alloca",
            "dead_dealloc",
            "condition_last",
        }:
            assert "memref.alloc" not in result.stdout, result.stdout
            assert "memref.dealloc" not in result.stdout, result.stdout
        if case in {"dead_local", "external_width", "external_range"}:
            # The retained external fill/range writes need no old destination
            # value. Ordinary lowering's unused init loads must also be gone.
            assert "memref.load" not in result.stdout, result.stdout
        if case == "ordering_consumer":
            assert (
                tasks[4]["bosc_ame.compute_predecessors"] == "array<i64: 2>"
            ), case
    print(
        "PASS private dead masks removed; physical and graph consumers preserved"
    )


if __name__ == "__main__":
    main()
