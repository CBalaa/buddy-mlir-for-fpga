# RUN: %PYTHON %s | FileCheck %s
"""Model actual rsqrtf rounding and the physical span of allocated memrefs."""

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine, SimError  # noqa: E402

direct_inputs = """module {
  func.func @inputs(%a: memref<256xi8>, %b: memref<256xi8>) {
    %av = memref.reinterpret_cast %a to offset: [0], sizes: [2, 128], strides: [128, 1] : memref<256xi8> to memref<2x128xi8, strided<[128, 1]>>
    %bv = memref.reinterpret_cast %b to offset: [0], sizes: [128, 2], strides: [1, 128] : memref<256xi8> to memref<128x2xi8, strided<[1, 128]>>
    %ar = bosc_ame.tile.load_a %av {bosc_ame.direct_ddr} : memref<2x128xi8, strided<[128, 1]>> -> vector<2x64xi8>
    %br = bosc_ame.tile.load_b %bv {bosc_ame.direct_ddr} : memref<128x2xi8, strided<[1, 128]>> -> vector<64x2xi8>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %acc = bosc_ame.tile.load_acc %spm : memref<2x16xi32, #bosc_ame.spm<0>> -> vector<2x16xi32>
    bosc_ame.tile.store_acc %acc, %spm : vector<2x16xi32>, memref<2x16xi32, #bosc_ame.spm<0>>
    return
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "direct-inputs.mlir"
    path.write_text(direct_inputs)
    direct = Machine(path, "inputs")
    buffers = {"%a": bytearray(range(256)), "%b": bytearray(range(256))}
    direct.launch(buffers)
    assert direct.accesses == [
        ("%a", 0, 64, "read"),
        ("%a", 128, 192, "read"),
        ("%b", 0, 64, "read"),
        ("%b", 128, 192, "read"),
    ]
    assert len(direct.values["%ar"]) == 2
    assert len(direct.values["%br"]) == 64
    for source in (
        direct_inputs.replace(" {bosc_ame.direct_ddr}", "", 1),
        direct_inputs.replace("vector<2x64xi8>", "vector<3x64xi8>"),
    ):
        path.write_text(source)
        try:
            Machine(path, "inputs").launch(buffers)
        except SimError:
            pass
        else:
            raise AssertionError(
                "invalid tile input address space/shape accepted"
            )

# Function arguments carry physical layouts too. Their static descriptor
# offset is in elements and must survive independently of pointer alignment.
argument_layout = """module {
  func.func @argument(%input: memref<3x2xi32, strided<[1, 3], offset: 2>>) -> i32 {
    %row = arith.constant 2 : index
    %col = arith.constant 1 : index
    %value = memref.load %input[%row, %col] : memref<3x2xi32, strided<[1, 3], offset: 2>>
    return %value : i32
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "argument-layout.mlir"
    path.write_text(argument_layout)
    machine = Machine(path, "argument")
    machine.launch({"%input": bytearray(struct.pack("<8i", *range(8)))})
    assert machine.values["%input"].strides == [1, 3]
    assert machine.values["%input"].offset == 2
    assert machine.values["%value"] == 7
    path.write_text(argument_layout.replace("offset: 2", "offset: ?"))
    try:
        Machine(path, "argument").launch({"%input": bytearray(32)})
    except SimError as error:
        assert "requires a caller descriptor" in str(error)
    else:
        raise AssertionError(
            "dynamic argument offset silently replaced by zero"
        )

module = """module {
  func.func @model() {
    %seven = arith.constant 7.0 : f32
    %inverse = math.rsqrt %seven : f32
    %columns = memref.alloc() : memref<3x2xi8, strided<[1, 3], offset: 2>>
    %rows = memref.transpose %columns (d0, d1) -> (d1, d0) : memref<3x2xi8, strided<[1, 3], offset: 2>> to memref<2x3xi8, strided<[3, 1], offset: 2>>
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
    rows = machine.values["%rows"]
    assert rows.data is columns.data and rows.strides == [3, 1]
    assert rows.base == columns.base and rows.offset == columns.offset
    assert rows.load([1, 2]) == -127
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
