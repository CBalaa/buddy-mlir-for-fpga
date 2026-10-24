# RUN: %PYTHON %s %t
"""Both compatibility names preserve explicit serial matmul alias semantics."""

import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402

SOURCE = """module {
  func.func @alias(%a: memref<2x2xi32>, %b: memref<2x2xi32>,
                   %c: memref<2x2xi32>) {
    linalg.matmul ins(%a, %b : memref<2x2xi32>, memref<2x2xi32>)
      outs(%c : memref<2x2xi32>)
    return
  }
  func.func @floating(%a: memref<1x4xf32>, %b: memref<4x1xf32>,
                      %c: memref<1x1xf32>) {
    linalg.matmul ins(%a, %b : memref<1x4xf32>, memref<4x1xf32>)
      outs(%c : memref<1x1xf32>)
    return
  }
  func.func @unsigned(%a: memref<1x1xi8>, %b: memref<1x1xi8>,
                      %c: memref<1x1xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
      ins(%a, %b : memref<1x1xi8>, memref<1x1xi8>)
      outs(%c : memref<1x1xi32>)
    return
  }
}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())
    outputs = []
    for entry in ("--lower-linalg-to-boscame", "--linalg-to-boscame"):
        result = subprocess.run(
            [opt, entry], input=SOURCE, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    emitted = outputs[0]
    assert "complete = false" in emitted
    assert 'contract.compute = {kind = "matmul", order = "mnk"}' in emitted
    assert "bosc_ame.tile" not in emitted and "arith.extui" in emitted
    module = work / "ordered.mlir"
    module.write_text(emitted)

    values = [1, 2, 3, 4]
    expected = list(values)
    rhs = [5, 6, 7, 8]
    for m in range(2):
        for n in range(2):
            for k in range(2):
                expected[2 * m + n] += expected[2 * m + k] * rhs[2 * k + n]
    storage = bytearray(struct.pack("<4i", *values))
    Machine(module, "alias").launch(
        {
            "%arg0": storage,
            "%arg1": bytearray(struct.pack("<4i", *rhs)),
            "%arg2": storage,
        }
    )
    assert struct.unpack("<4i", storage) == tuple(expected)

    output = bytearray(struct.pack("<f", 7.0))
    Machine(module, "floating").launch(
        {
            "%arg0": bytearray(struct.pack("<4f", 1e20, 1.0, -1e20, 1.0)),
            "%arg1": bytearray(struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)),
            "%arg2": output,
        }
    )
    assert struct.unpack("<f", output) == (1.0,)
    output = bytearray(struct.pack("<i", -17))
    Machine(module, "unsigned").launch(
        {"%arg0": bytearray([255]), "%arg1": bytearray([128]), "%arg2": output}
    )
    assert struct.unpack("<i", output) == (255 * 128 - 17,)
    lowered = subprocess.run(
        [opt, str(module), "--lower-bosc-ame"], text=True, capture_output=True
    )
    assert lowered.returncode == 0, lowered.stderr


if __name__ == "__main__":
    main()
