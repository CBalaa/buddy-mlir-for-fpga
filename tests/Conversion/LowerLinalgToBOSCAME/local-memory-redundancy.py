# RUN: %PYTHON %s %t buddy-opt
"""Remove local redundant SPM accesses without losing tail or graph semantics."""

import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_sim import op_name, read_module, result_name  # noqa: E402

TYPE = "memref<32xf32, #bosc_ame.spm<0>>"
OPTIONS = "--lower-linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias "
VALUES = tuple((i - 17) / 4 for i in range(32))
OTHER = tuple(2 + i / 8 for i in range(32))


def generic(left, right, output, use_init=False, split=False):
    prefix = ""
    suffix = ""
    value = "%product"
    if use_init:
        suffix = "%with_init = arith.addf %product, %old : f32"
        value = "%with_init"
    if split:
        prefix = """
      %i = linalg.index 0 : index
      %condition = arith.cmpi slt, %i, %bound : index
"""
        suffix = """
      %added = arith.addf %product, %lhs : f32
      %selected = arith.select %condition, %product, %added : f32
"""
        value = "%selected"
    return f"""
    linalg.generic {{indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}}
      ins({left}, {right} : {TYPE}, {TYPE}) outs({output} : {TYPE}) {{
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      {prefix}
      %product = arith.mulf %lhs, %rhs : f32
      {suffix}
      linalg.yield {value} : f32
    }}
"""


def source(case):
    declarations = ""
    result = "%out"
    if case.startswith("select_"):
        # The mask is replaced by the proven fixed prefix during lowering.
        declarations = "%mask = memref.alloca() : memref<32xi1>"
        result = "%x" if case == "select_in_place" else "%out"
        if case == "select_cast":
            declarations += f"\n%alias = memref.cast %x : {TYPE} to {TYPE}"
            result = "%alias"
        computations = f"""
    linalg.fill ins(%zero : f32) outs(%y : {TYPE})
    linalg.generic {{indexing_maps = [#id, #id, #id, #id], iterator_types = ["parallel"]}}
      ins(%mask, %x, %y : memref<32xi1>, {TYPE}, {TYPE})
      outs({result} : {TYPE}) attrs = {{bosc_ame.zero_tail_select_prefix = 16 : i64}} {{
    ^bb0(%condition: i1, %yes: f32, %no: f32, %old: f32):
      %selected = arith.select %condition, %yes, %no : f32
      linalg.yield %selected : f32
    }}
"""
    elif case == "distinct_operands":
        computations = generic("%x", "%y", "%out")
    elif case == "distinct_cast_operands":
        declarations = f"%alias = memref.cast %x : {TYPE} to {TYPE}"
        computations = generic("%x", "%alias", "%out")
    elif case == "used_init":
        result = "%x"
        computations = generic("%x", "%x", "%x", use_init=True)
    elif case == "separate_loops":
        computations = generic("%x", "%x", "%x") + generic("%x", "%x", "%out")
    else:
        computations = generic("%x", "%x", "%out", split=case == "split_ranges")
    return f"""
#id = affine_map<(i) -> (i)>
module {{
  func.func @local(%input: memref<32xf32>, %other: memref<32xf32>, %output: memref<32xf32>) {{
    %zero = arith.constant 0.0 : f32
    %bound = arith.constant 16 : index
    %x = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : {TYPE}
    %y = bosc_ame.spm.view {{bank = 0 : i32, offset = 128 : i64}} : {TYPE}
    %out = bosc_ame.spm.view {{bank = 0 : i32, offset = 256 : i64}} : {TYPE}
    memref.copy %input, %x : memref<32xf32> to {TYPE}
    memref.copy %other, %y : memref<32xf32> to {TYPE}
    {declarations}
    {computations}
    memref.copy {result}, %output : {TYPE} to memref<32xf32>
    return
  }}
}}
"""


def lower(binary, text, path, options=OPTIONS):
    result = subprocess.run(
        [binary, "-", options],
        input=text,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    path.write_text(result.stdout)
    return result.stdout


def walk(operations):
    for operation in operations:
        yield operation
        for block in operation.blocks:
            yield from walk(block)


def memory_ops(loop, name):
    return [op for op in walk(loop.blocks[0]) if op_name(op.text) == name]


def spm_root(text, offset):
    return re.search(
        rf"(%[\w.]+) = bosc_ame\.spm\.view[^\n]*offset = {offset} : i64", text
    )[1]


def check_local(binary, directory, case):
    path = directory / (case + ".mlir")
    text = lower(binary, source(case), path)
    loops = [
        op for op in walk(read_module(path)) if op_name(op.text) == "scf.for"
    ]
    if case.startswith("select_"):
        destination = spm_root(text, 256 if case == "select_distinct" else 0)
        copies = []
        for loop in loops:
            loads, stores = (
                memory_ops(loop, "vector.load"),
                memory_ops(loop, "vector.store"),
            )
            if len(loads) != 1 or len(stores) != 1:
                continue
            load = re.search(r"vector\.load (%[\w.]+)\[", loads[0].text)[1]
            store = re.search(
                r"vector\.store %[^,]+, (%[\w.]+)\[", stores[0].text
            )[1]
            if store == destination and load in (
                spm_root(text, 0),
                spm_root(text, 128),
            ):
                copies.append(load)
        expected = [spm_root(text, 128)]
        if case != "select_in_place":
            expected.insert(0, spm_root(text, 0))
        assert copies == expected, (case, copies, expected, text)
        expected_values = VALUES[:16] + (0.0,) * 16
    else:
        arithmetic = [loop for loop in loops if memory_ops(loop, "arith.mulf")]
        expected_counts = {
            "duplicate": [1],
            "distinct_operands": [2],
            "distinct_cast_operands": [2],
            "used_init": [1],
            "separate_loops": [1, 1],
            "split_ranges": [1, 1],
        }[case]
        assert [
            len(memory_ops(loop, "vector.load")) for loop in arithmetic
        ] == expected_counts, text
        for loop in arithmetic:
            multiply = memory_ops(loop, "arith.mulf")[0]
            left, right = re.search(
                r"arith\.mulf (%[\w.]+), (%[\w.]+)", multiply.text
            ).groups()
            assert (left == right) == (
                case not in ("distinct_operands", "distinct_cast_operands")
            ), (
                case,
                multiply.text,
            )
            assert left in {
                result_name(load.text)
                for load in memory_ops(loop, "vector.load")
            }, text
        if case == "distinct_operands":
            expected_values = tuple(x * y for x, y in zip(VALUES, OTHER))
        elif case == "used_init":
            expected_values = tuple(x * x + x for x in VALUES)
        elif case == "separate_loops":
            expected_values = tuple(x**4 for x in VALUES)
        elif case == "split_ranges":
            expected_values = tuple(x * x for x in VALUES[:16]) + tuple(
                x * x + x for x in VALUES[16:]
            )
        else:
            expected_values = tuple(x * x for x in VALUES)
    output = bytearray(128)
    Machine(path, "local").launch(
        {
            "%arg0": bytearray(struct.pack("<32f", *VALUES)),
            "%arg1": bytearray(struct.pack("<32f", *OTHER)),
            "%arg2": output,
        }
    )
    assert output == struct.pack("<32f", *expected_values), (case, output)


def check_layernorm(binary, directory):
    fixture = Path(__file__).parent / "Inputs/layernorm-197x768-bufferized.mlir"
    text = lower(
        binary,
        fixture.read_text(),
        directory / "layernorm.mlir",
        "--linalg-to-boscame=buffer-placement=true ",
    )
    tasks = {}
    for line in text.splitlines():
        task = re.search(r"bosc_ame.compute_id = (\d+) : i64", line)
        if task:
            number = int(task[1])
            assert number not in tasks, number
            tasks[number] = line
    assert set(tasks) == set(range(22)) - {1, 2, 3}, tasks
    assert "bosc_ame.compute_graph_node_count = 19 : i64" in text, text
    for task, attribute, value in (
        (14, "compute_input_values", "26, 1"),
        (14, "compute_predecessors", "0, 13"),
        (14, "compute_output_values", "27"),
        (15, "compute_input_values", "27, 27"),
        (15, "compute_predecessors", "14"),
        (15, "compute_output_values", "28"),
    ):
        assert f"bosc_ame.{attribute} = array<i64: {value}>" in tasks[task], (
            tasks[task]
        )
    # These rank-one task bodies contain no nested regions. The surviving
    # task-14 anchor must remain on the tail loop, with task 15 on square.
    for task in (14, 15):
        closing = text.index(tasks[task])
        body = text[text.rfind("scf.for ", 0, closing) : closing]
        assert body.count("vector.load ") == 1, (task, body)
        assert body.count("vector.store ") == 1, (task, body)
        if task == 14:
            lower_bound = re.search(r"scf.for %[^ ]+ = (%[\w.]+)", body)[1]
            assert f"{lower_bound} = arith.constant 768 : index" in text, body
        else:
            left, right = re.search(
                r"arith\.mulf (%[\w.]+), (%[\w.]+)", body
            ).groups()
            assert left == right, body


def main():
    directory, binary = Path(sys.argv[1]), sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    for case in (
        "select_in_place",
        "select_distinct",
        "select_cast",
        "duplicate",
        "distinct_operands",
        "distinct_cast_operands",
        "used_init",
        "separate_loops",
        "split_ranges",
    ):
        check_local(binary, directory, case)
    check_layernorm(binary, directory)
    print(
        "PASS local memory redundancy: exact outputs, required copies/tails, load scope and 19 graph tasks preserved"
    )


if __name__ == "__main__":
    main()
