# RUN: %PYTHON %s %t buddy-opt
"""Promote private scalar scratch without dropping visible or aliased writes."""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402

SOURCE = """
module {
  func.func private @observe(memref<*xf32>)
  func.func @private_copy(%out: memref<f32>) {
    %five = arith.constant 5.0 : f32
    %one = arith.constant 1.0 : f32
    %seed = memref.alloca() : memref<f32>
    %sum = memref.alloca() : memref<f32>
    memref.store %five, %seed[] : memref<f32>
    memref.copy %seed, %sum : memref<f32> to memref<f32>
    %x = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    linalg.fill ins(%one : f32) outs(%x : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%x : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%sum : memref<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %next = arith.addf %in, %init : f32
        linalg.yield %next : f32
    }
    %value = memref.load %sum[] : memref<f32>
    memref.store %value, %out[] : memref<f32>
    return
  }
  func.func @external_result(%out: memref<f32>) {
    %five = arith.constant 5.0 : f32
    %one = arith.constant 1.0 : f32
    %x = bosc_ame.spm.view {bank = 0 : i32, offset = 128 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    memref.store %five, %out[] : memref<f32>
    linalg.fill ins(%one : f32) outs(%x : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%x : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%out : memref<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %next = arith.addf %in, %init : f32
        linalg.yield %next : f32
    }
    return
  }
  func.func @alias(%out: memref<f32>) {
    %five = arith.constant 5.0 : f32
    %sum = memref.alloca() : memref<f32>
    %alias = memref.cast %sum : memref<f32> to memref<*xf32>
    memref.store %five, %sum[] : memref<f32>
    func.call @observe(%alias) : (memref<*xf32>) -> ()
    %value = memref.load %sum[] : memref<f32>
    memref.store %value, %out[] : memref<f32>
    return
  }
  func.func @nested(%out: memref<f32>, %choose: i1) {
    %five = arith.constant 5.0 : f32
    %one = arith.constant 1.0 : f32
    %sum = memref.alloca() : memref<f32>
    memref.store %five, %sum[] : memref<f32>
    scf.if %choose {
      memref.store %one, %sum[] : memref<f32>
    }
    %value = memref.load %sum[] : memref<f32>
    memref.store %value, %out[] : memref<f32>
    return
  }
  func.func @uninitialized(%out: memref<f32>) {
    %sum = memref.alloca() : memref<f32>
    %value = memref.load %sum[] : memref<f32>
    memref.store %value, %out[] : memref<f32>
    return
  }
  func.func @heap(%out: memref<f32>) {
    %five = arith.constant 5.0 : f32
    %sum = memref.alloc() : memref<f32>
    memref.store %five, %sum[] : memref<f32>
    %value = memref.load %sum[] : memref<f32>
    memref.store %value, %out[] : memref<f32>
    memref.dealloc %sum : memref<f32>
    return
  }
}
"""


def run(binary, arguments, source):
    result = subprocess.run(
        [binary, "-", *arguments],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def function_source(name):
    body = re.search(
        r"  func.func @" + name + r"\(.*?^  }", SOURCE, re.S | re.M
    )
    assert body, name
    return (
        "module {\n  func.func private @observe(memref<*xf32>)\n"
        + body.group()
        + "\n}\n"
    )


def main():
    directory, binary = Path(sys.argv[1]), sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    options = ["--lower-linalg-to-boscame=dma-abi=args_aligned64_nonalias"]
    # Preserve every former external-scalar source as an explicit v1 boundary
    # case. The neutral planner has no independent external scalar store or
    # private-to-private scalar copy recipe.
    for name, diagnostic in (
        ("private_copy", "requires matching static rank-two SPM storage"),
        (
            "external_result",
            "reading an external region after its terminal write",
        ),
        ("alias", "no independent external or SPM scalar-store recipe"),
        ("nested", "if requires explicit then and else scopes"),
        ("uninitialized", "no initialized content version"),
        ("heap", "no independent external or SPM scalar-store recipe"),
    ):
        result = subprocess.run(
            [binary, "-", *options],
            input=function_source(name),
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0, name
        assert diagnostic in result.stderr, result.stderr

    supported = function_source("private_copy")
    supported = supported.replace(
        "func.func @private_copy(%out: memref<f32>)",
        "func.func @private_seed() -> f32",
    )
    supported = supported.replace(
        "    %seed = memref.alloca() : memref<f32>\n", ""
    )
    supported = supported.replace(
        "    memref.store %five, %seed[] : memref<f32>\n"
        "    memref.copy %seed, %sum : memref<f32> to memref<f32>",
        "    linalg.fill ins(%five : f32) outs(%sum : memref<f32>)",
    )
    supported = supported.replace(
        "    memref.store %value, %out[] : memref<f32>\n    return",
        "    return %value : f32",
    )
    text = run(binary, options, supported)
    assert "vector.reduction <add>" in text, text
    assert "memref.copy" not in text, text
    path = directory / "scalar-seed.mlir"
    path.write_text(text)
    machine = Machine(path, "private_seed")
    machine.launch({})
    returned = re.search(
        r"return (%[\w]+) : f32", machine.function.blocks[0][-1].text
    )
    assert returned
    assert machine.values[returned.group(1)] == 37.0

    # Ordinary private host scalar memory is preserved without inventing an
    # external scalar route. Its exact before/after oracle remains 5.0f.
    heap = (
        function_source("heap")
        .replace(
            "func.func @heap(%out: memref<f32>)", "func.func @heap() -> f32"
        )
        .replace("    memref.store %value, %out[] : memref<f32>\n", "")
    )
    heap = heap.replace("    return", "    return %value : f32").replace(
        "memref.store %five, %sum[] : memref<f32>",
        "linalg.fill ins(%five : f32) outs(%sum : memref<f32>)",
    )
    heap_text = run(binary, options, heap)
    path = directory / "scalar-host.mlir"
    path.write_text(heap_text)
    machine = Machine(path, "heap")
    machine.launch({})
    returned = re.search(
        r"return (%[\w]+) : f32", machine.function.blocks[0][-1].text
    )
    assert returned
    assert machine.values[returned.group(1)] == 5.0
    print(
        "PASS private scalar nonzero seed and host memory; unsupported external recipes diagnosed"
    )


if __name__ == "__main__":
    main()
