# RUN: %PYTHON %s %t buddy-opt
"""Keep operand addresses, ordered arithmetic and fixed-window numerics exact."""

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
    arithmetic = [loop for loop in loops if memory_ops(loop, "arith.mulf")]
    assert len(arithmetic) == (2 if case == "separate_loops" else 1), text
    for loop in arithmetic:
        # The emitter mechanically binds each used input. General load CSE is
        # a later standard optimization, so equal addresses may have two loads.
        loads = memory_ops(loop, "vector.load")
        assert len(loads) == 2, text
        multiply = memory_ops(loop, "arith.mulf")[0]
        operands = re.search(
            r"arith\.mulf (%[\w.]+), (%[\w.]+)", multiply.text
        ).groups()
        assert set(operands) == {result_name(load.text) for load in loads}, text
    if case == "distinct_operands":
        expected_values = tuple(x * y for x, y in zip(VALUES, OTHER))
    elif case == "used_init":
        assert sum(len(memory_ops(loop, "arith.addf")) for loop in loops) == 1
        expected_values = tuple(x * x + x for x in VALUES)
    elif case == "separate_loops":
        expected_values = tuple(x**4 for x in VALUES)
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


def check_legacy(binary):
    cases = dict.fromkeys(
        ("select_in_place", "select_distinct", "select_cast"),
        "unsupported legacy target metadata: bosc_ame.zero_tail_select_prefix",
    )
    cases["split_ranges"] = "generic requires one pure arithmetic operation"
    for case, diagnostic in cases.items():
        result = subprocess.run(
            [binary, "-", OPTIONS],
            input=source(case),
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0 and diagnostic in result.stderr, (
            case,
            result.stderr,
        )
    # The original LayerNorm artifact carries retired geometry. Current neutral
    # prefix/tail and graph coverage lives in buffer-placement-metadata.py and
    # boscame-spm-compute.py, including actual numerical comparisons.
    fixture = Path(__file__).parent / "Inputs/layernorm-197x768-bufferized.mlir"
    result = subprocess.run(
        [binary, str(fixture), "--linalg-to-boscame"],
        text=True,
        capture_output=True,
    )
    assert (
        result.returncode != 0
        and "obsolete geometry attribute bosc_ame.bm" in result.stderr
    )


def main():
    directory, binary = Path(sys.argv[1]), sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    for case in (
        "duplicate",
        "distinct_operands",
        "distinct_cast_operands",
        "used_init",
        "separate_loops",
    ):
        check_local(binary, directory, case)
    check_legacy(binary)
    print(
        "PASS exact local arithmetic and operand addresses; unsafe legacy prefix proofs rejected"
    )


if __name__ == "__main__":
    main()
