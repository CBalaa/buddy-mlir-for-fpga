# RUN: %PYTHON %s | FileCheck %s
"""Model actual rsqrtf rounding and the physical span of allocated memrefs."""

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine  # noqa: E402

module = """module {
  func.func @model() {
    %seven = arith.constant 7.0 : f32
    %inverse = math.rsqrt %seven : f32
    %columns = memref.alloc() : memref<3x2xi8, strided<[1, 3], offset: 2>>
    %scalar = memref.alloc() : memref<i32, strided<[]>>
    %empty = memref.alloc() : memref<0x4xi8>
    return
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "model.mlir"
    path.write_text(module)
    machine = Machine(path, "model")
    machine.block(machine.function.blocks[0])
    # Emitted LLVM calls rsqrtf; NR implements f32 sqrt then f32 division.
    # For 7.0f this yields 0x3ec18490, while rounding 1/sqrt(double) once
    # yields 0x3ec1848f. The distinction changes downstream LayerNorm values.
    assert struct.pack("<f", machine.values["%inverse"]) == bytes.fromhex(
        "9084c13e"
    )
    columns = machine.values["%columns"]
    assert columns.strides == [1, 3] and columns.offset == 2
    assert len(columns.data) == 8
    columns.store([2, 1], -127)
    assert columns.load([2, 1]) == -127 and columns.data[7] == 129
    assert len(machine.values["%scalar"].data) == 4
    assert len(machine.values["%empty"].data) == 0

# A direct AME store addresses external DDR with an explicit byte row stride.
# Seed the accumulator window as hardware would, then ensure the five-element
# pitch leaves each row's final sentinel untouched and no DMA was recorded.
direct_module = """module {
  func.func @direct(%out: memref<2x5xi32>) {
    %stride = arith.constant 20 : i64
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x3xi32, #bosc_ame.spm<0>>
    %acc = bosc_ame.tile.load_acc %spm : memref<2x3xi32, #bosc_ame.spm<0>> -> vector<2x3xi32>
    bosc_ame.msce32.m %acc, %out, %stride : vector<2x3xi32>, memref<2x5xi32>, i64
    return
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "direct.mlir"
    path.write_text(direct_module)
    output = bytearray(b"\xcc" * (2 * 5 * 4))
    direct = Machine(path, "direct")
    for index, value in enumerate((1, 2, 3, 4, 5, 6)):
        direct.spm[0][index * 4 : index * 4 + 4] = int(value).to_bytes(
            4, "little", signed=True
        )
    direct.launch({"%out": output})
    assert [
        int.from_bytes(output[index : index + 4], "little", signed=True)
        for index in (0, 4, 8, 20, 24, 28)
    ] == [1, 2, 3, 4, 5, 6]
    assert output[12:20] == b"\xcc" * 8 and not direct.pending
    assert direct.accesses == [
        ("%out", 0, 12, "write"),
        ("%out", 20, 32, "write"),
    ]

# The fused frontend form keeps the result as a vector.transfer_write directly
# to DDR.  Exercise that operation in the simulator as well as the final
# vector.store form above.
vector_module = """module {
  func.func @direct_vector(%src: memref<16xi32>, %out: memref<16xi32>) {
    %c0 = arith.constant 0 : index
    %v = vector.load %src[%c0] : memref<16xi32>, vector<16xi32>
    vector.transfer_write %v, %out[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>
    return
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "direct-vector.mlir"
    path.write_text(vector_module)
    source = bytearray().join(struct.pack("<i", value) for value in range(16))
    output = bytearray(b"\xcc" * len(source))
    vector_machine = Machine(path, "direct_vector")
    vector_machine.launch({"%src": source, "%out": output})
    assert output == source and not vector_machine.pending

# The pre-LLVM tile form carries the same direct-DDR contract.  It must be
# observable as an external write in the simulator rather than silently
# looking like an SPM operation.
direct_tile_module = """module {
  func.func @direct_tile(%out: memref<2x5xi32>) {
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x3xi32, #bosc_ame.spm<0>>
    %acc = bosc_ame.tile.load_acc %spm : memref<2x3xi32, #bosc_ame.spm<0>> -> vector<2x3xi32>
    bosc_ame.tile.store_acc %acc, %out {bosc_ame.direct_ddr} : vector<2x3xi32>, memref<2x5xi32>
    return
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "direct-tile.mlir"
    path.write_text(direct_tile_module)
    output = bytearray(b"\xcc" * (2 * 5 * 4))
    direct = Machine(path, "direct_tile")
    for index, value in enumerate((7, 8, 9, 10, 11, 12)):
        direct.spm[0][index * 4 : index * 4 + 4] = int(value).to_bytes(
            4, "little", signed=True
        )
    direct.launch({"%out": output})
    assert output[12:20] == b"\xcc" * 8 and output[32:] == b"\xcc" * 8
    assert direct.accesses == [
        ("%out", 0, 12, "write"),
        ("%out", 20, 32, "write"),
    ]

# CHECK: PASS buffer simulator rsqrtf rounding, physical layouts and direct AME DDR stores
# CHECK: PASS direct vector.transfer_write DDR store
print(
    "PASS buffer simulator rsqrtf rounding, physical layouts and direct AME DDR stores"
)
print("PASS direct vector.transfer_write DDR store")
