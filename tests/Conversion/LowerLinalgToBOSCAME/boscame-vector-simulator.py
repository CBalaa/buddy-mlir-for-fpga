# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Independent checks of the functional model's direct-SPM vector semantics."""

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_sim import SimError  # noqa: E402

SOURCE = """module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @vectors(%source: memref<16xf32>, %dest: memref<32xf32>, %ints: memref<16xi8>, %intout: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c48 = arith.constant 48 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0.0 : f32
    %two = arith.constant 2.0 : f32
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>
    %token = bosc_ame.dma.start %source, %spm, %c64 {channel = 0 : i32} : (memref<16xf32>, memref<16xf32, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %token : !bosc_ame.dma_token<0>
    %v = vector.load %spm[%c0] : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    %ordered = scf.for %lane_index = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> (f32) {
      %single = vector.load %spm[%lane_index] : memref<16xf32, #bosc_ame.spm<0>>, vector<1xf32>
      %lane = vector.extract %single[0] : f32 from vector<1xf32>
      %next = arith.addf %lane, %acc : f32
      scf.yield %next : f32
    }
    memref.store %ordered, %dest[%c3] : memref<32xf32>
    %b = vector.broadcast %two : f32 to vector<16xf32>
    %d = arith.constant dense<1.0> : vector<16xf32>
    %mul = arith.mulf %v, %b : vector<16xf32>
    %sub = arith.subf %mul, %d : vector<16xf32>
    %add = arith.addf %sub, %d : vector<16xf32>
    vector.store %add, %spm[%c0] : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    %sum = vector.reduction <add>, %v, %zero : vector<16xf32> into f32
    memref.store %sum, %dest[%c0] : memref<32xf32>
    %extract = vector.extract %v[2] : f32 from vector<16xf32>
    memref.store %extract, %dest[%c1] : memref<32xf32>
    %dynamic = vector.extract %v[%c1] : f32 from vector<16xf32>
    memref.store %dynamic, %dest[%c2] : memref<32xf32>
    vector.store %add, %dest[%c16] : memref<32xf32>, vector<16xf32>
    %iv = vector.load %ints[%c0] : memref<16xi8>, vector<16xi8>
    %signed = arith.extsi %iv : vector<16xi8> to vector<16xi32>
    %unsigned = arith.extui %iv : vector<16xi8> to vector<16xi32>
    vector.store %signed, %intout[%c0] : memref<64xi32>, vector<16xi32>
    vector.store %unsigned, %intout[%c16] : memref<64xi32>, vector<16xi32>
    %max = arith.constant dense<2147483647> : vector<16xi32>
    %one = arith.constant dense<1> : vector<16xi32>
    %itwo = arith.constant dense<2> : vector<16xi32>
    %wrapped = arith.addi %max, %one : vector<16xi32>
    %product = arith.muli %wrapped, %itwo : vector<16xi32>
    vector.store %wrapped, %intout[%c32] : memref<64xi32>, vector<16xi32>
    vector.store %product, %intout[%c48] : memref<64xi32>, vector<16xi32>
    return
  }
}
"""


def main():
    path = Path(sys.argv[1])
    path.write_text(SOURCE)
    # A tree reduction can retain the otherwise rounded-away +1. The ordered
    # scalar accumulator must instead produce 13, including the first packet.
    x = np.array([16777216, 1, -16777216] + [1] * 13, dtype=np.float32)
    integers = np.arange(-128, -112, dtype=np.int8)

    def buffers():
        return {
            "%source": bytearray(x.tobytes()),
            "%dest": bytearray(32 * 4),
            "%ints": bytearray(integers.tobytes()),
            "%intout": bytearray(64 * 4),
        }

    data = buffers()
    machine = Machine(path, "vectors")
    # The second execution reuses the first execution's SPM contents.
    for _ in range(2):
        machine.launch(data)
        got = np.frombuffer(data["%dest"], dtype=np.float32)
        expected_sum = np.float32(0)
        for lane in x:
            expected_sum = np.float32(expected_sum + lane)
        assert got[0] == expected_sum == 13
        assert got[3] == expected_sum
        assert got[1] == x[2] and got[2] == x[1]
        expected = np.add(
            np.subtract(
                np.multiply(x, np.float32(2), dtype=np.float32),
                np.float32(1),
                dtype=np.float32,
            ),
            np.float32(1),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(got[16:], expected)
        intout = np.frombuffer(data["%intout"], dtype=np.int32)
        np.testing.assert_array_equal(intout[:16], integers.astype(np.int32))
        np.testing.assert_array_equal(
            intout[16:32], integers.view(np.uint8).astype(np.int32)
        )
        assert np.all(intout[32:48] == -(1 << 31))
        assert np.all(intout[48:] == 0)
        np.testing.assert_array_equal(
            np.frombuffer(machine.spm[0], dtype=np.float32, count=16), expected
        )

    cases = (
        (
            "vector.reduction <add>",
            "vector.reduction <mul>",
            "unsupported vector reduction",
        ),
        ("vector.extract %v[2]", "vector.extract %v[16]", "outside vector"),
        (
            "bosc_ame.dma.wait %token : !bosc_ame.dma_token<0>",
            "",
            "before DMA wait",
        ),
        (
            "%v = vector.load",
            "%bad = memref.load %spm[%c0] : memref<16xf32, #bosc_ame.spm<0>>\n"
            "    %v = vector.load",
            "scalar SPM",
        ),
    )
    for old, new, diagnostic in cases:
        path.write_text(SOURCE.replace(old, new))
        try:
            Machine(path, "vectors").launch(buffers())
        except SimError as error:
            assert diagnostic in str(error), str(error)
        else:
            raise AssertionError("invalid vector execution was accepted")
    # CHECK: PASS vector model: direct SPM, per-operation f32 rounding, ordered reduction, extraction, signed extension and i32 wrap
    print(
        "PASS vector model: direct SPM, per-operation f32 rounding, ordered reduction, extraction, signed extension and i32 wrap"
    )
    # CHECK: PASS failures: unsupported reduction, out-of-bounds lane, missing DMA wait and scalar SPM
    print(
        "PASS failures: unsupported reduction, out-of-bounds lane, missing DMA wait and scalar SPM"
    )


if __name__ == "__main__":
    main()
