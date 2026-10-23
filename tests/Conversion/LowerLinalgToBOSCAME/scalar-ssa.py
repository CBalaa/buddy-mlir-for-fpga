# RUN: %PYTHON %s %t buddy-opt
"""Promote private scalar scratch without dropping visible or aliased writes."""

import re
import struct
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


def main():
    directory, binary = Path(sys.argv[1]), sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    text = run(
        binary,
        [
            "--lower-linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias"
        ],
        SOURCE,
    )
    functions = {
        part.split("(", 1)[0]: part for part in text.split("func.func @")[1:]
    }
    for name in ("private_copy", "heap"):
        body = functions[name]
        assert not re.search(
            r"memref\.(?:alloca?|load|copy|dealloc)\b", body
        ), body
        assert body.count("memref.store ") == 1, body
    assert "vector.reduction <add>" in functions["private_copy"]
    assert "memref.load" in functions["external_result"]
    assert functions["external_result"].count("memref.store ") == 2
    for name in ("alias", "nested", "uninitialized"):
        assert "memref.alloca" in functions[name], functions[name]
        assert "memref.load" in functions[name], functions[name]
    # Simulate before canonicalization to exercise the lowering itself.
    path = directory / "scalar-ssa.mlir"
    path.write_text(text)
    for name, scalars, expected in (
        ("private_copy", {}, 37.0),
        ("external_result", {}, 37.0),
        ("nested", {"%arg1": 0}, 5.0),
        ("nested", {"%arg1": 1}, 1.0),
        ("heap", {}, 5.0),
    ):
        output = bytearray(4)
        Machine(path, name).launch({"%arg0": output}, scalars)
        assert struct.unpack("<f", output)[0] == expected, (name, output)
    print(
        "PASS scalar SSA: private copy/nonzero seed/heap removed; external, alias and nested memory preserved"
    )


if __name__ == "__main__":
    main()
