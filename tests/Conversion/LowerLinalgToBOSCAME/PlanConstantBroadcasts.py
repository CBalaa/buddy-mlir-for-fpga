# RUN: %PYTHON %s %t
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Keep scalar constants in vector registers without allocating zero matrices."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402
from PlanBoundaryTransfers import dictionaries  # noqa: E402


def module(constant="0.0", swapped=False):
    lhs, rhs = ("%c", "%x") if swapped else ("%x", "%c")
    return f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @constant(%src: memref<64xf32>, %dst: memref<64xf32>) {{
    %c = arith.constant {constant} : f32
    %local = memref.alloc() alignment = 64 : memref<64xf32>
    memref.copy %src, %local : memref<64xf32> to memref<64xf32>
    linalg.generic {{indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
      iterator_types = ["parallel"]}} ins(%local : memref<64xf32>)
      outs(%local : memref<64xf32>) {{
    ^bb0(%x: f32, %unused: f32):
      %r = arith.addf {lhs}, {rhs} : f32
      linalg.yield %r : f32
    }}
    memref.copy %local, %dst : memref<64xf32> to memref<64xf32>
    memref.dealloc %local : memref<64xf32>
    return
  }}
}}
"""


def direct_yield(constant, dead_operation):
    initialize = (
        "%init = arith.constant 3.5 : f32\n"
        "linalg.fill ins(%init : f32) outs(%local : memref<16xf32>)"
        if dead_operation
        else ""
    )
    dead = "%dead = arith.negf %old : f32" if dead_operation else ""
    return f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @direct_yield(%dst: memref<16xf32>) {{
    %c = arith.constant {constant} : f32
    %local = memref.alloc() alignment = 64 : memref<16xf32>
    {initialize}
    linalg.generic {{indexing_maps = [affine_map<(i)->(i)>],
      iterator_types = ["parallel"]}} outs(%local : memref<16xf32>) {{
    ^bb0(%old: f32):
      {dead}
      linalg.yield %c : f32
    }}
    memref.copy %local, %dst : memref<16xf32> to memref<16xf32>
    memref.dealloc %local : memref<16xf32>
    return
  }}
}}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(source, *passes):
        result = subprocess.run(
            [opt, *passes], input=source, text=True, capture_output=True
        )
        assert result.returncode >= 0, result.stderr
        assert (
            "LLVM ERROR" not in result.stderr
            and "PLEASE submit" not in result.stderr
        )
        return result

    def reject(source, *passes):
        result = invoke(source, *passes)
        assert result.returncode != 0 and "error:" in result.stderr, result

    bits = np.resize(
        np.array(
            [
                0,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0xFFC54321,
                1,
                0x80000001,
                0x3F800000,
                0xBF800000,
            ],
            dtype=np.uint32,
        ),
        64,
    )
    raw = bits.tobytes()
    guard = bytes([0xD7]) * 64
    first = None
    for name, literal in [
        ("zero", "0.0"),
        ("negative_zero", "-0.0"),
        ("nonzero", "1.25"),
    ]:
        for swapped in (False, True):
            case = name + ("_left" if swapped else "_right")
            source = module(literal, swapped)
            planned = invoke(
                source,
                "--kernel-contract-annotate",
                "--bosc-ame-plan=plan-version=1",
                "--bosc-ame-plan-verify",
            )
            assert planned.returncode == 0, planned.stderr
            assert planned.stdout.count("memref.alloc()") == 1, planned.stdout
            records = dictionaries(planned.stdout, "plan.vector =")
            assert len(records) == 1 and "constant-broadcast" in records[0][2]
            assert f"operand = {0 if swapped else 1} : i64" in records[0][2]
            assert planned.stdout.count("arith.addf") == 1
            path = work / (case + ".planned.mlir")
            path.write_text(planned.stdout)
            emitted = invoke(
                planned.stdout, "--bosc-ame-emit", "--bosc-ame-spm-dma-check"
            )
            assert emitted.returncode == 0, emitted.stderr
            assert (
                "vector.broadcast" in emitted.stdout
                and "arith.addf" in emitted.stdout
            )
            assert emitted.stdout.count("bosc_ame.spm.view") == 1, (
                emitted.stdout
            )
            path = work / (case + ".emitted.mlir")
            path.write_text(emitted.stdout)
            buffers = {
                "%arg0": bytearray(raw + guard),
                "%arg1": bytearray(bytes(256) + guard),
            }
            machine = Machine(path, "constant")
            machine.launch(buffers)
            scalar = np.float32(float(literal))
            with np.errstate(all="ignore"):
                expected = (
                    np.add(scalar, bits.view(np.float32), dtype=np.float32)
                    if swapped
                    else np.add(bits.view(np.float32), scalar, dtype=np.float32)
                )
            actual = np.frombuffer(buffers["%arg1"], dtype=np.float32, count=64)
            assert np.array_equal(np.isnan(actual), np.isnan(expected))
            valid = ~np.isnan(expected)
            assert np.array_equal(
                actual.view(np.uint32)[valid], expected.view(np.uint32)[valid]
            )
            assert (
                buffers["%arg0"] == raw + guard
                and buffers["%arg1"][256:] == guard
            )
            if first is None:
                first = planned.stdout, records[0]

    planned, (begin, end, vector) = first
    scalar = dictionaries(vector, "scalar_operands = [")[0][2]
    malformed = {
        "value": vector.replace(
            "value = 0.000000e+00 : f32", "value = 1.000000e+00 : f32"
        ),
        "sign": vector.replace(
            "value = 0.000000e+00 : f32", "value = -0.000000e+00 : f32"
        ),
        "operand": vector.replace(
            "operand = 1 : i64, value", "operand = 0 : i64, value"
        ),
        "kind": vector.replace("constant-broadcast", "implicit"),
        "missing": re.sub(r", scalar_operands = \[\{[^}]*\}\]", "", vector),
        "array_type": re.sub(
            r"scalar_operands = \[\{[^}]*\}\]",
            'scalar_operands = "wrong"',
            vector,
        ),
        "operand_type": vector.replace(
            "operand = 1 : i64, value", "operand = 1 : i32, value"
        ),
        "duplicate": vector.replace(
            "scalar_operands = [" + scalar + "]",
            "scalar_operands = [" + scalar + ", " + scalar + "]",
        ),
    }
    for name, replacement in malformed.items():
        assert replacement != vector, (name, vector)
        bad = planned[:begin] + replacement + planned[end:]
        for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            reject(bad, gate)

    dynamic = (
        module()
        .replace(
            "%dst: memref<64xf32>)", "%dst: memref<64xf32>, %dynamic: f32)"
        )
        .replace("%c = arith.constant 0.0 : f32", "")
        .replace("arith.addf %x, %c", "arith.addf %x, %dynamic")
    )
    explicit = invoke(
        dynamic,
        "--kernel-contract-annotate",
        "--bosc-ame-plan=plan-version=1",
        "--bosc-ame-plan-verify",
    )
    assert explicit.returncode == 0, explicit.stderr
    assert 'level = "scalar"' in explicit.stdout
    assert "explicit-scalar-input" in explicit.stdout
    assert "scalar_operands" not in explicit.stdout
    emitted = invoke(
        explicit.stdout, "--bosc-ame-emit", "--bosc-ame-spm-dma-check"
    )
    assert emitted.returncode == 0, emitted.stderr
    path = work / "dynamic.emitted.mlir"
    path.write_text(emitted.stdout)
    buffers = {
        "%arg0": bytearray(raw + guard),
        "%arg1": bytearray(bytes(256) + guard),
    }
    machine = Machine(path, "constant")
    machine.launch(buffers, scalars={"%arg2": np.float32(1.25)})
    with np.errstate(all="ignore"):
        expected = np.add(
            bits.view(np.float32), np.float32(1.25), dtype=np.float32
        )
    actual = np.frombuffer(buffers["%arg1"], dtype=np.float32, count=64)
    valid = ~np.isnan(expected)
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(
        actual.view(np.uint32)[valid], expected.view(np.uint32)[valid]
    )
    assert buffers["%arg0"] == raw + guard and buffers["%arg1"][256:] == guard

    # A terminator operand is a scalar capture even when there is no scalar
    # arithmetic. A preceding unused operation also exercises its ordinal.
    for literal in ("-0.0", "1.25"):
        for dead_operation in (False, True):
            planned = invoke(
                direct_yield(literal, dead_operation),
                "--kernel-contract-annotate",
                "--bosc-ame-plan=plan-version=1",
                "--bosc-ame-plan-verify",
            )
            assert planned.returncode == 0, planned.stderr
            records = dictionaries(planned.stdout, "plan.vector =")
            assert len(records) == 1, planned.stdout
            begin, end, vector = records[0]
            scalar = dictionaries(vector, "scalar_operands = [")[0][2]
            assert 'kind = "constant-broadcast"' in scalar
            assert "operand = 0 : i64" in scalar
            assert ("operation = 1 : i64" in scalar) == dead_operation
            emitted = invoke(
                planned.stdout, "--bosc-ame-emit", "--bosc-ame-spm-dma-check"
            )
            assert emitted.returncode == 0, emitted.stderr
            name = f"direct-yield-{literal}-{int(dead_operation)}"
            path = work / (name + ".emitted.mlir")
            path.write_text(emitted.stdout)
            output = bytearray(bytes([0x93]) * 64 + guard)
            Machine(path, "direct_yield").launch({"%arg0": output})
            expected = np.full(16, float(literal), dtype=np.float32)
            assert output[:64] == expected.tobytes(), (name, output)
            assert output[64:] == guard

            malformed = {
                "missing": vector.replace(
                    ", scalar_operands = [" + scalar + "]", ""
                ),
                "empty": vector.replace(
                    "scalar_operands = [" + scalar + "]", "scalar_operands = []"
                ),
                "value": vector.replace(
                    scalar,
                    re.sub(r"value = [^,}]+", "value = 2.5 : f32", scalar),
                ),
                "operand": vector.replace(
                    "operand = 0 : i64", "operand = 1 : i64"
                ),
                "operation": vector.replace(
                    scalar,
                    re.sub(r", operation = \d+ : i64", "", scalar)[:-1]
                    + ", operation = 999 : i64}",
                ),
                "kind": vector.replace("constant-broadcast", "implicit"),
            }
            if dead_operation:
                malformed["missing_operation"] = vector.replace(
                    ", operation = 1 : i64", ""
                )
            for label, replacement in malformed.items():
                assert replacement != vector, (name, label, vector)
                bad = (
                    planned.stdout[:begin] + replacement + planned.stdout[end:]
                )
                for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
                    reject(bad, gate)
    print(
        "PASS: explicit constant vector broadcasts, exact strict add and malformed-plan gates"
    )


if __name__ == "__main__":
    main()
