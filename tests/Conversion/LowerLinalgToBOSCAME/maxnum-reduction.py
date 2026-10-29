# RUN: %PYTHON %s %t buddy-opt buddy-translate llc
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Keep maxnum's initializer, NaNs and explicit masked padding on the target."""

import math
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402

SOURCE = """
module attributes {
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
} {
  func.func @maximum(%input: memref<197xf32>) -> f32 {
    %minus_inf = arith.constant 0xFF800000 : f32
    %initial = arith.constant 0xFF800000 : f32
    %x = memref.alloc() alignment = 64 : memref<256xf32>
    linalg.fill ins(%minus_inf : f32) outs(%x : memref<256xf32>)
    %valid = memref.subview %x[0] [197] [1]
      : memref<256xf32> to memref<197xf32, strided<[1]>>
    memref.copy %input, %valid {
      contract.access = {boundary = "external_ddr", kind = "load",
        mask = {kind = "rectangular", valid_shape = array<i64: 197>},
        other = 0xFF800000 : f32}}
      : memref<197xf32> to memref<197xf32, strided<[1]>>
    %max = memref.alloca() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%max : memref<f32>)
    linalg.reduce ins(%x : memref<256xf32>) outs(%max : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
        %next = arith.maxnumf %value, %acc : f32
        linalg.yield %next : f32
      }
    %result = memref.load %max[] : memref<f32>
    return %result : f32
  }
}
"""


def invoke(binary, source, *flags):
    return subprocess.run(
        [binary, "-", *flags], input=source, text=True, capture_output=True
    )


def checked(binary, source, *flags):
    result = invoke(binary, source, *flags)
    assert result.returncode == 0, result.stderr
    return result.stdout


def simulate(path, values):
    guard = b"\xa5" * 64
    raw = struct.pack("<197f", *values) + guard
    source = bytearray(raw)
    machine = Machine(path, "maximum")
    machine.spm[0][:] = b"\x7f" * len(machine.spm[0])
    machine.spm[1][:] = b"\xd3" * len(machine.spm[1])
    machine.launch({"%arg0": source})
    returned = re.search(r"return (%[\w]+) : f32", path.read_text())
    assert returned
    assert source == raw
    reads = [entry for entry in machine.accesses if entry[0] == "%arg0"]
    assert (
        sum(end - begin for _, begin, end, action in reads if action == "read")
        == 788
    )
    assert all(
        0 <= begin < end <= 788 and action == "read"
        for _, begin, end, action in reads
    )
    return machine.values[returned[1]]


def main():
    work, binary, translate, llc = Path(sys.argv[1]), *sys.argv[2:]
    work.mkdir(parents=True, exist_ok=True)
    first = None
    for swapped in (False, True):
        for initial in ("0xFF800000", "7.0"):
            source = SOURCE.replace(
                "%initial = arith.constant 0xFF800000",
                "%initial = arith.constant " + initial,
            )
            if swapped:
                source = source.replace(
                    "arith.maxnumf %value, %acc", "arith.maxnumf %acc, %value"
                )
            planned = checked(
                binary,
                source,
                "--kernel-contract-annotate",
                "--bosc-ame-plan=plan-version=1",
                "--bosc-ame-plan-verify",
            )
            assert 'strategy = "ordered-vector-maxnum"' in planned
            assert 'fp_policy = "preserve"' in planned
            emitted = checked(
                binary, planned, "--bosc-ame-emit", "--bosc-ame-spm-dma-check"
            )
            assert emitted.count("vector.reduction <maxnumf>") == 1
            assert "arith.maxnumf" not in emitted and "linalg." not in emitted
            path = work / f"max-{swapped}-{initial}.mlir"
            path.write_text(emitted)
            cases = [
                ([-197.0 + i for i in range(197)], -1.0),
                ([math.nan] * 196 + [-3.0], -3.0),
                ([math.nan] * 197, -math.inf),
                ([-math.inf] * 197, -math.inf),
                ([math.inf] + [1.0] * 196, math.inf),
                ([-0.0] * 197, -0.0),
                ([-0.0, 0.0] + [-0.0] * 195, 0.0),
            ]
            for values, expected in cases:
                actual = simulate(path, values)
                if initial == "7.0":
                    expected = max(expected, 7.0)
                assert actual == expected, (actual, expected)
                if expected == 0:
                    assert math.copysign(1, actual) == math.copysign(
                        1, expected
                    )
            if first is None:
                first = planned, emitted

    planned, emitted = first
    mutations = {
        "wrong-strategy": planned.replace(
            'strategy = "ordered-vector-maxnum"',
            'strategy = "ordered-vector-add"',
        ),
        "nan-propagating": planned.replace("arith.maxnumf", "arith.maximumf"),
        "duplicate-input": planned.replace(
            "arith.maxnumf %in, %init", "arith.maxnumf %in, %in"
        ),
        "fast-math": planned.replace(
            "%in, %init : f32", "%in, %init fastmath<nnan> : f32"
        ),
    }
    for name, mutation in mutations.items():
        assert mutation != planned, name
        for flag in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            result = invoke(binary, mutation, flag)
            assert result.returncode != 0 and "error:" in result.stderr, (
                name,
                result.stderr,
            )
            assert "LLVM ERROR" not in result.stderr

    lowered = checked(binary, emitted, "--lower-bosc-ame")
    assert "vector.reduce.fmax" in lowered
    llvm_ir = checked(translate, lowered, "--buddy-to-llvmir")
    assembly = checked(
        llc,
        llvm_ir,
        "-O2",
        "-mtriple=riscv64",
        "-mattr=+m,+a,+f,+d,+v",
        "-target-abi=lp64d",
    )
    assert "vfredmax.vs" in assembly
    calls = re.findall(r"\bcall\s+([\w.]+)", assembly)
    assert all(
        name in ("bosc_ame_dma_start", "bosc_ame_dma_wait") for name in calls
    ), calls
    (work / "max.s").write_text(assembly)
    print(
        "PASS maxnum reduction: explicit -Inf padding/init, NaNs, signed zeros, guards, verifier gates and RVV lowering"
    )


if __name__ == "__main__":
    main()
