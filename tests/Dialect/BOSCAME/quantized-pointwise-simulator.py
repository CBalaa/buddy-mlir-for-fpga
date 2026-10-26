# RUN: %PYTHON %s %t
"""Check mixed vector casts, exact ties-away rounding and maxnum reduction."""

import struct
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine, SimError  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
source = work / "quant.mlir"
source.write_text(
    """
module {
  func.func @quant(%out: memref<8xi8>, %maximum: memref<1xf32>) {
    %c0 = arith.constant 0 : index
    %input = arith.constant dense<[-127.5, -2.5, -0.5, -0.49999997, 0.49999997, 0.5, 2.5, 127.5]> : vector<8xf32>
    %low = arith.constant dense<-127.0> : vector<8xf32>
    %high = arith.constant dense<127.0> : vector<8xf32>
    %positive = arith.constant dense<0.5> : vector<8xf32>
    %negative = arith.constant dense<-0.5> : vector<8xf32>
    %one = arith.constant dense<1> : vector<8xi32>
    %minus_one = arith.constant dense<-1> : vector<8xi32>
    %zero = arith.constant dense<0> : vector<8xi32>
    %lower = arith.maxnumf %input, %low : vector<8xf32>
    %clamped = arith.minnumf %lower, %high : vector<8xf32>
    %base = arith.fptosi %clamped : vector<8xf32> to vector<8xi32>
    %base_float = arith.sitofp %base : vector<8xi32> to vector<8xf32>
    %fraction = arith.subf %clamped, %base_float : vector<8xf32>
    %up = arith.cmpf oge, %fraction, %positive : vector<8xf32>
    %down = arith.cmpf ole, %fraction, %negative : vector<8xf32>
    %minus = arith.select %down, %minus_one, %zero : vector<8xi1>, vector<8xi32>
    %adjust = arith.select %up, %one, %minus : vector<8xi1>, vector<8xi32>
    %rounded = arith.addi %base, %adjust : vector<8xi32>
    %bytes = arith.trunci %rounded : vector<8xi32> to vector<8xi8>
    vector.store %bytes, %out[%c0] : memref<8xi8>, vector<8xi8>
    %absolute = math.absf %input : vector<8xf32>
    %identity = arith.constant 0xFF800000 : f32
    %max = vector.reduction <maxnumf>, %absolute, %identity : vector<8xf32> into f32
    %max_vector = vector.broadcast %max : f32 to vector<1xf32>
    vector.store %max_vector, %maximum[%c0] : memref<1xf32>, vector<1xf32>
    scf.execute_region {
      scf.yield
    }
    return
  }
}
"""
)
buffers = {"%out": bytearray(8), "%maximum": bytearray(4)}
machine = Machine(source, "quant")
machine.launch(buffers, {})
assert list(struct.unpack("<8b", buffers["%out"])) == [
    -127,
    -3,
    -1,
    0,
    0,
    1,
    3,
    127,
]
assert struct.unpack("<f", buffers["%maximum"])[0] == 127.5

# The target implements absf by clearing the sign bit, avoiding a vector of
# scalar libm calls and spills around every lane's call.
bitwise = work / "bitwise.mlir"
bitwise.write_text(
    source.read_text().replace(
        "%absolute = math.absf %input : vector<8xf32>",
        """
    %bits = arith.bitcast %input : vector<8xf32> to vector<8xi32>
    %mask = arith.constant dense<2147483647> : vector<8xi32>
    %masked = arith.andi %bits, %mask : vector<8xi32>
    %absolute = arith.bitcast %masked : vector<8xi32> to vector<8xf32>
""",
    )
)
bit_buffers = {"%out": bytearray(8), "%maximum": bytearray(4)}
Machine(bitwise, "quant").launch(bit_buffers, {})
assert bit_buffers == buffers

# Undefined casts must fail explicitly; wrapping is appropriate only after a
# representable integer has been computed and then narrowed with trunci.
bad = work / "bad.mlir"
bad.write_text(
    """
module {
  func.func @bad() {
    %infinity = arith.constant 0x7F800000 : f32
    %invalid = arith.fptosi %infinity : f32 to i32
    return
  }
}
"""
)
try:
    Machine(bad, "bad").launch({}, {})
except SimError as error:
    assert "undefined nonfinite fptosi" in str(error)
else:
    raise AssertionError("simulator accepted an undefined fptosi")
