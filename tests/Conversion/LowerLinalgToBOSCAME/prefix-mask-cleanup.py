# RUN: %PYTHON %s buddy-opt
"""Prefix specialization prunes only private, unused mask computations."""

import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine  # noqa: E402

OPTIONS = (
    "--linalg-to-boscame=buffer-placement=true enable-spm-dma=true "
    "dma-abi=args_aligned64_nonalias "
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


def check_standard_prefix(binary):
    # The tail is expressed by executable standard IR, not a trusted target
    # mask hint. Test in-place, separate output and a static alias descriptor.
    for case in ("in_place", "distinct", "alias"):
        false_value = -0.0 if case == "distinct" else 0.0
        result = (
            "%work"
            if case == "in_place"
            else "%alias"
            if case == "alias"
            else "%other"
        )
        setup = (
            "%alias = memref.cast %work : memref<32xf32> to memref<32xf32>"
            if case == "alias"
            else "%other = memref.alloc() alignment = 64 : memref<32xf32>"
            if case == "distinct"
            else ""
        )
        module = f"""module {{
  func.func @prefix(%input: memref<32xf32>, %output: memref<32xf32>) {{
    %bound = arith.constant 16 : index
    %zero = arith.constant {false_value} : f32
    %work = memref.alloc() alignment = 64 : memref<32xf32>
    memref.copy %input, %work : memref<32xf32> to memref<32xf32>
    {setup}
    linalg.generic {{indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
      iterator_types = ["parallel"]}}
      ins({"%alias" if case == "alias" else "%work"} : memref<32xf32>) outs({result} : memref<32xf32>) {{
    ^bb0(%x: f32, %old: f32):
      %lane = linalg.index 0 : index
      %valid = arith.cmpi slt, %lane, %bound : index
      %square = arith.mulf %x, %x : f32
      %value = arith.select %valid, %square, %zero : f32
      linalg.yield %value : f32
    }}
    memref.copy {result}, %output : memref<32xf32> to memref<32xf32>
    return
  }}
}}"""
        lowered = subprocess.run(
            [binary, "-", OPTIONS], input=module, text=True, capture_output=True
        )
        assert lowered.returncode == 0, lowered.stderr
        assert (
            "arith.select" not in lowered.stdout
            and "linalg." not in lowered.stdout
        )
        assert lowered.stdout.count('kind = "fill"') >= 1
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.mlir"
            path.write_text(lowered.stdout)
            machine = Machine(path, "prefix")
            for launch in range(2):
                # An arithmetic mask would preserve NaN/Inf in these lanes.
                # A select must store the false constant's exact zero bits.
                values = (
                    tuple((i - 17 + launch) / 4 for i in range(16))
                    + (float("nan"), float("inf"), -float("inf"), -0.0) * 4
                )
                source_bytes = struct.pack("<32f", *values)
                expected = struct.pack(
                    "<32f", *(x * x for x in values[:16]), *([false_value] * 16)
                )
                buffers = {
                    "%arg0": bytearray(source_bytes),
                    "%arg1": bytearray([0xCD] * 128),
                }
                machine.launch(buffers)
                assert buffers["%arg0"] == source_bytes
                assert buffers["%arg1"] == expected, (case, launch)


def main():
    binary = sys.argv[1]
    for case in (
        "dead_local",
        "dead_alloca",
        "dead_dealloc",
        "condition_last",
        "other_consumer",
        "alias_consumer",
        "ordering_consumer",
        "external_mask",
        "external_width",
        "external_range",
        "effectful_range",
        "barrier_mask",
    ):
        result = subprocess.run(
            [binary, "-", OPTIONS],
            input=source(case),
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0, case
        assert (
            "unsupported legacy target metadata: bosc_ame.prefix_mask_aux"
            in result.stderr
        ), (case, result.stderr)
    check_standard_prefix(binary)
    print(
        "PASS standard prefix tails exact; legacy target hints rejected for all consumers/effects"
    )


if __name__ == "__main__":
    main()
