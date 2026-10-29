# RUN: %PYTHON %s %t buddy-opt buddy-translate llc 2>&1 | FileCheck %s
"""Execute the ordered f32 scalar-SSA lowering and check its scope."""

import ctypes
import math
import random
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OPTION = "scalar-f32-ordered-reductions=true"


def checked(binary, source, *flags):
    result = subprocess.run(
        [str(binary), "-", *flags], input=source, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ordered(values, initial):
    result = f32(initial)
    for value in values:
        result = f32(result + f32(value))
    return result


def function(text, name):
    match = re.search(
        r"^define[^\n]* @" + name + r"\([^\n]*\).*?^}\n",
        text,
        re.M | re.S,
    )
    assert match, name
    return match[0]


def source():
    functions = []
    for lanes in (16, 17):
        functions.append(f"""
  func.func @sum{lanes}(%x: memref<{lanes}xf32>, %initial: f32) -> f32
      attributes {{llvm.emit_c_interface}} {{
    %c0 = arith.constant 0 : index
    %v = vector.load %x[%c0] : memref<{lanes}xf32>, vector<{lanes}xf32>
    %r = vector.reduction <add>, %v, %initial : vector<{lanes}xf32> into f32
    return %r : f32
  }}
""")
    functions.append("""
  func.func @loop_sizes(%x: memref<33xf32>, %initial: f32, %count: index)
      -> f32 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %r = scf.for %i = %c0 to %count step %c1
        iter_args(%acc = %initial) -> f32 {
      %a = vector.load %x[%c0] : memref<33xf32>, vector<16xf32>
      %s = vector.reduction <add>, %a, %acc : vector<16xf32> into f32
      %b = vector.load %x[%c16] : memref<33xf32>, vector<17xf32>
      %t = vector.reduction <add>, %b, %s : vector<17xf32> into f32
      scf.yield %t : f32
    }
    return %r : f32
  }
""")
    functions.append("""
  func.func @padded1024(%x: memref<768xf32>, %initial: f32)
      -> f32 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c768 = arith.constant 768 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant dense<0.0> : vector<16xf32>
    %valid = scf.for %i = %c0 to %c768 step %c16
        iter_args(%acc = %initial) -> f32 {
      %v = vector.load %x[%i] : memref<768xf32>, vector<16xf32>
      %r = vector.reduction <add>, %v, %acc : vector<16xf32> into f32
      scf.yield %r : f32
    }
    %padded = scf.for %i = %c768 to %c1024 step %c16
        iter_args(%acc = %valid) -> f32 {
      %r = vector.reduction <add>, %zero, %acc : vector<16xf32> into f32
      scf.yield %r : f32
    }
    return %padded : f32
  }
""")
    return "module {\n" + "".join(functions) + "}\n"


GATES = """
module {
  func.func @f64_add(%v: vector<16xf64>, %s: f64) -> f64 {
    %r = vector.reduction <add>, %v, %s : vector<16xf64> into f64
    return %r : f64
  }
  func.func @integer_add(%v: vector<16xi32>, %s: i32) -> i32 {
    %r = vector.reduction <add>, %v, %s : vector<16xi32> into i32
    return %r : i32
  }
  func.func @reassociated_add(%v: vector<16xf32>, %s: f32) -> f32 {
    %r = vector.reduction <add>, %v, %s fastmath<reassoc> : vector<16xf32> into f32
    return %r : f32
  }
  func.func @maxnum(%v: vector<16xf32>, %s: f32) -> f32 {
    %r = vector.reduction <maxnumf>, %v, %s : vector<16xf32> into f32
    return %r : f32
  }
}
"""


UNSUPPORTED = {
    "scalable": """
  func.func @unsupported(%v: vector<[16]xf32>, %s: f32) -> f32 {
    %r = vector.reduction <add>, %v, %s : vector<[16]xf32> into f32
    return %r : f32
  }
""",
    "implicit-acc": """
  func.func @unsupported(%v: vector<16xf32>) -> f32 {
    %r = vector.reduction <add>, %v : vector<16xf32> into f32
    return %r : f32
  }
""",
    "rank-zero": """
  func.func @unsupported(%v: vector<f32>, %s: f32) -> f32 {
    %r = vector.reduction <add>, %v, %s : vector<f32> into f32
    return %r : f32
  }
""",
    "masked": """
  func.func @unsupported(%v: vector<16xf32>, %s: f32, %mask: vector<16xi1>) -> f32 {
    %r = vector.mask %mask { vector.reduction <add>, %v, %s : vector<16xf32> into f32 }
      : vector<16xi1> -> f32
    return %r : f32
  }
""",
    "parallel": """
  func.func @unsupported(%v: vector<16xf32>, %s: f32, %out: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
      %r = vector.reduction <add>, %v, %s : vector<16xf32> into f32
      memref.store %r, %out[%i] : memref<4xf32>
      scf.reduce
    }
    return
  }
""",
}


class Memref(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.POINTER(ctypes.c_float)),
        ("aligned", ctypes.POINTER(ctypes.c_float)),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


def execute(library, name, values, initial, count=None):
    # Source and guards catch snapshot corruption across repeated invocations.
    words = [0x4F400001] * 16 + [bits(v) for v in values] + [0x4F400001] * 16
    storage = (ctypes.c_uint32 * len(words))(*words)
    pointer = ctypes.cast(
        ctypes.byref(storage, 64), ctypes.POINTER(ctypes.c_float)
    )
    descriptor = Memref(pointer, pointer, 0, len(values), 1)
    kernel = getattr(library, "_mlir_ciface_" + name)
    kernel.argtypes = [ctypes.POINTER(Memref), ctypes.c_float]
    args = [ctypes.byref(descriptor), initial]
    if count is not None:
        kernel.argtypes.append(ctypes.c_int64)
        args.append(count)
    kernel.restype = ctypes.c_float
    actual = kernel(*args)
    assert list(storage) == words, name + " changed input or guard"
    expected = f32(initial)
    for _ in range(count if count is not None else 1):
        expected = ordered(values, expected)
        if name == "padded1024":
            expected = ordered([0.0] * 256, expected)
    if math.isnan(expected):
        assert math.isnan(actual), (name, actual, expected)
    else:
        assert bits(actual) == bits(expected), (
            name,
            hex(bits(actual)),
            hex(bits(expected)),
            values,
            initial,
            count,
        )
    return actual


def main():
    work, binary, translate, llc = Path(sys.argv[1]), *sys.argv[2:]
    work.mkdir(parents=True, exist_ok=True)
    original = source()
    default = checked(binary, original, "--lower-bosc-ame")
    disabled = checked(
        binary, original, "--lower-bosc-ame=scalar-f32-ordered-reductions=false"
    )
    assert default == disabled, "the pass must remain default-off"
    assert default.count("llvm.intr.vector.reduce.fadd") == 6
    lowered = checked(binary, original, "--lower-bosc-ame=" + OPTION)
    assert "vector.reduce" not in lowered
    assert "scalar_ordered_snapshot" not in lowered
    llvm_ir = checked(translate, lowered, "--buddy-to-llvmir")
    for name, stores, lanes in (
        ("sum16", 1, 16),
        ("sum17", 1, 17),
        ("loop_sizes", 2, 17),
        ("padded1024", 2, 16),
    ):
        body = function(llvm_ir, name)
        assert len(re.findall(r"\balloca\b", body)) == 1, body
        assert re.search(rf"alloca float, i64 {lanes}, align 16", body), body
        entry = re.split(r"^\w[^\n]*:", body, maxsplit=1, flags=re.M)[0]
        assert "alloca" in entry, "snapshot allocated inside a loop"
        assert body.count("store volatile <") == stores, body
        assert body.count("load volatile float") == stores, body
        assert len(re.findall(r"\bfadd float\b", body)) == stores, body
        assert not re.search(r"fadd (?:fast|reassoc|nnan|nsz)", body), body
        assert "extractelement" not in body and "vector.reduce" not in body
    gates_default = checked(binary, GATES, "--lower-bosc-ame")
    gates_enabled = checked(binary, GATES, "--lower-bosc-ame=" + OPTION)
    assert gates_default == gates_enabled, "nonmatching reductions changed"
    assert "llvm.intr.vector.reduce.fmax" in gates_enabled
    for name, body in UNSUPPORTED.items():
        module = "module {\n" + body + "}\n"
        result = subprocess.run(
            [binary, "-", "--lower-bosc-ame=" + OPTION],
            input=module,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0, (
            name + " silently retained a vector reduction"
        )
        assert "scalar-f32-ordered-reductions requires" in result.stderr, (
            name,
            result.stderr,
        )
        assert "LLVM ERROR" not in result.stderr, (name, result.stderr)
    tile_source = original.replace(
        "module {", 'module attributes {bosc_ame.target = "nr-fpga"} {', 1
    )
    tile = checked(binary, tile_source, "--lower-bosc-ame-tile=" + OPTION)
    assert "vector.reduce" not in tile
    assert tile.count("llvm.load volatile") == 6
    assembly = checked(
        llc,
        llvm_ir,
        "-O2",
        "-mtriple=riscv64",
        "-mattr=+m,+a,+f,+d,+v",
        "-target-abi=lp64d",
    )
    assert "vfred" not in assembly and "vfmv" not in assembly, assembly
    assert len(re.findall(r"\bfadd\.s\b", assembly)) == 6, assembly
    (work / "ordered.mlir").write_text(lowered)
    (work / "ordered.ll").write_text(llvm_ir)
    (work / "ordered.s").write_text(assembly)
    clang = REPO / "llvm/build/bin/clang"
    if not clang.is_file():
        clang = shutil.which("clang")
    library_path = work / "ordered.so"
    result = subprocess.run(
        [
            str(clang),
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            str(work / "ordered.ll"),
            "-o",
            str(library_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    library = ctypes.CDLL(str(library_path.resolve()))
    cases = 0
    for lanes in (16, 17):
        name = f"sum{lanes}"
        directed = [
            ([float(i + 1) for i in range(lanes)], 7.0),
            (([1.0e20, 1.0, -1.0e20, 1.0] * 5)[:lanes], 3.0),
            ([-0.0] * lanes, -0.0),
            ([-0.0] * (lanes - 1) + [0.0], -0.0),
            ([0.0] * (lanes - 1) + [1.0], 9.0),
            ([math.inf] + [0.0] * (lanes - 1), 0.0),
            ([math.inf, -math.inf] + [0.0] * (lanes - 2), 0.0),
            ([1.0] * lanes, math.nan),
        ]
        # Each lane, including the final padding-position lane, participates.
        for index in range(lanes):
            v = [0.0] * lanes
            v[index] = math.nan
            directed.append((v, 0.0))
        for values, initial in directed:
            execute(library, name, values, initial)
            cases += 1
        rng = random.Random(734 + lanes)
        for _ in range(50):
            values = [
                f32(rng.randint(-1000000, 1000000) / 1024) for _ in range(lanes)
            ]
            execute(library, name, values, f32(rng.uniform(-16, 16)))
            cases += 1
    values = ([1.0e20, 1.0, -1.0e20, 1.0] * 9)[:33]
    for count in (0, 1, 2, 1000):
        execute(library, "loop_sizes", values, 5.0, count)
        cases += 1
    execute(library, "loop_sizes", [-0.0] * 33, -0.0, 5)
    cases += 1
    padded_cases = [
        ([-0.0] * 768, -0.0),
        ([0.0] * 767 + [1.0], 7.0),
        ([1.0e20, 1.0, -1.0e20, 1.0] * 192, 0.0),
        ([f32(((c * 13) % 511 - 255) / 64) for c in range(768)], 0.0),
        ([math.nan] + [0.0] * 767, 0.0),
    ]
    for values, initial in padded_cases:
        execute(library, "padded1024", values, initial)
        cases += 1
    assert cases == 159, cases
    # CHECK: PASS ordered scalar SSA: 159 executed cases, volatile snapshots, entry allocation, scope and RVV codegen
    print(
        "PASS ordered scalar SSA: 159 executed cases, volatile snapshots, entry allocation, scope and RVV codegen"
    )


if __name__ == "__main__":
    main()
