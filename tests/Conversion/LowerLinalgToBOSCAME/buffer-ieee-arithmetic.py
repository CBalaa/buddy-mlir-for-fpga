# RUN: %PYTHON %s
"""Interpret emitted IEEE arithmetic and exponent-bit construction exactly."""

import math
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine, SimError  # noqa: E402


def execute(directory, body, buffers=None):
    path = directory / "arithmetic.mlir"
    path.write_text("module {\n" + body + "\n}\n")
    machine = Machine(path, "arithmetic")
    machine.launch(buffers or {})
    return machine


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def check(actual, expected):
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        if math.isnan(reference):
            assert math.isnan(result), (result, reference)
        else:
            assert bits(result) == bits(reference), (result, reference)


with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    tiny = struct.unpack("<f", struct.pack("<I", 1))[0]
    largest = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
    numerator = [
        1,
        -1,
        1,
        -1,
        0,
        -0.0,
        0,
        -0.0,
        math.inf,
        -math.inf,
        math.inf,
        math.nan,
        tiny,
        -tiny,
        largest,
        -largest,
    ]
    denominator = [
        0,
        0,
        -0.0,
        -0.0,
        0,
        -0.0,
        -0.0,
        0,
        0,
        -0.0,
        math.inf,
        0,
        2,
        2,
        0.5,
        0.5,
    ]
    expected = [
        math.inf,
        -math.inf,
        -math.inf,
        math.inf,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.inf,
        math.inf,
        math.nan,
        math.nan,
        0,
        -0.0,
        math.inf,
        -math.inf,
    ]
    body = """
  func.func @arithmetic(%a: memref<16xf32>, %b: memref<16xf32>, %out: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %x = vector.load %a[%c0] : memref<16xf32>, vector<16xf32>
    %y = vector.load %b[%c0] : memref<16xf32>, vector<16xf32>
    %result = arith.divf %x, %y : vector<16xf32>
    vector.store %result, %out[%c0] : memref<16xf32>, vector<16xf32>
    return
  }
"""
    guard = b"\xda" * 64
    buffers = {
        "%a": bytearray(struct.pack("<16f", *numerator) + guard),
        "%b": bytearray(struct.pack("<16f", *denominator) + guard),
        "%out": bytearray(64) + bytearray(guard),
    }
    saved = {name: bytes(value) for name, value in buffers.items()}
    execute(directory, body, buffers)
    check(struct.unpack_from("<16f", buffers["%out"]), expected)
    assert buffers["%a"] == saved["%a"] and buffers["%b"] == saved["%b"]
    assert buffers["%out"][64:] == guard

    # Every operation rounds independently to f32, including overflow and
    # gradual underflow. No host exp/erf evaluates an approximation for us.
    body = """
  func.func @arithmetic() {
    %largest = arith.constant 0x7F7FFFFF : f32
    %tiny = arith.constant 0x00000001 : f32
    %negative_tiny = arith.constant 0x80000001 : f32
    %two = arith.constant 2.0 : f32
    %half = arith.constant 0.5 : f32
    %overflow = arith.mulf %largest, %two : f32
    %underflow = arith.mulf %tiny, %half : f32
    %negative_underflow = arith.mulf %negative_tiny, %half : f32
    %add_overflow = arith.addf %largest, %largest : f32
    return
  }
"""
    machine = execute(directory, body)
    check(
        [
            machine.values[name]
            for name in (
                "%overflow",
                "%underflow",
                "%negative_underflow",
                "%add_overflow",
            )
        ],
        [math.inf, 0, -0.0, math.inf],
    )

    # maxnum ignores one NaN and keeps NaN when no numeric operand exists.
    # Use +0 for mixed signs, and preserve -0 when every zero has that sign.
    body = """
  func.func @arithmetic() {
    %negative = arith.constant dense<-0.0> : vector<16xf32>
    %mixed = arith.constant dense<[-0.0, 0.0, -0.0, -0.0]> : vector<4xf32>
    %nan = arith.constant dense<0x7FC12345> : vector<16xf32>
    %zeros = arith.constant dense<0.0> : vector<16xf32>
    %minus_inf = arith.constant 0xFF800000 : f32
    %nan_init = arith.constant 0x7FC00000 : f32
    %negative_result = vector.reduction <maxnumf>, %negative, %minus_inf : vector<16xf32> into f32
    %mixed_result = vector.reduction <maxnumf>, %mixed, %minus_inf : vector<4xf32> into f32
    %all_nan = vector.reduction <maxnumf>, %nan, %nan_init : vector<16xf32> into f32
    %nan_with_padding = vector.reduction <maxnumf>, %nan, %minus_inf : vector<16xf32> into f32
    %ignored_nan = arith.maxnumf %nan, %zeros : vector<16xf32>
    return
  }
"""
    machine = execute(directory, body)
    check(
        [
            machine.values[name]
            for name in (
                "%negative_result",
                "%mixed_result",
                "%all_nan",
                "%nan_with_padding",
            )
        ],
        [-0.0, 0, math.nan, -math.inf],
    )
    check(machine.values["%ignored_nan"], [0] * 16)

    # Build exact 2^k factors by shifting exponent bits, as the generated exp
    # range-reduction recipe does. The high bit wraps at the declared width.
    body = """
  func.func @arithmetic() {
    %exponents = arith.constant dense<[1, 126, 127, 128, 254, 255, 256, -1]> : vector<8xi32>
    %shift = arith.constant dense<23> : vector<8xi32>
    %encoded = arith.shli %exponents, %shift : vector<8xi32>
    %floats = arith.bitcast %encoded : vector<8xi32> to vector<8xf32>
    return
  }
"""
    machine = execute(directory, body)
    assert [bits(value) for value in machine.values["%floats"]] == [
        (exponent << 23) % (1 << 32)
        for exponent in (1, 126, 127, 128, 254, 255, 256, -1)
    ]
    for amount in (-1, 32):
        invalid = """
  func.func @arithmetic() {
    %one = arith.constant 1 : i32
    %amount = arith.constant AMOUNT : i32
    %invalid = arith.shli %one, %amount : i32
    return
  }
""".replace("AMOUNT", str(amount))
        try:
            execute(directory, invalid)
        except SimError as error:
            assert "undefined out-of-range shli" in str(error)
        else:
            raise AssertionError("out-of-range shift was silently accepted")

print(
    "PASS IEEE division/overflow/subnormal/NaN/signed-zero and exact exponent shifts"
)
