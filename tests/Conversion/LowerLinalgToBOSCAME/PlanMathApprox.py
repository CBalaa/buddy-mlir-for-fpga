# RUN: %PYTHON %s %t
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Authorize f32 math explicitly, verify recipes, then execute emitted IR."""

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def accuracy(function):
    absolute, relative = (2e-6, 0.0) if function == "erf" else (2.0**-146, 2e-6)
    return (
        f'contract.math = {{{function} = {{domain = "all-f32", '
        f'special_values = "ieee", max_abs_error = {absolute:.17e} : f64, '
        f"max_rel_error = {relative:.17e} : f64}}}}"
    )


def module(function, count, authorization=True):
    attrs = f"attributes {{{accuracy(function)}}}" if authorization else ""
    return f"""module attributes {{
  contract.abi = {{args_aligned_bytes = 64 : i64, args_disjoint = true}}
}} {{
  func.func @compute(%src: memref<{count}xf32>, %dst: memref<{count}xf32>) {attrs} {{
    %local = memref.alloc() alignment = 64 : memref<{count}xf32>
    memref.copy %src, %local : memref<{count}xf32> to memref<{count}xf32>
    linalg.generic {{indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
      iterator_types = ["parallel"]}} ins(%local : memref<{count}xf32>)
      outs(%local : memref<{count}xf32>) {{
    ^bb0(%x: f32, %old: f32):
      %r = math.{function} %x : f32
      linalg.yield %r : f32
    }}
    memref.copy %local, %dst : memref<{count}xf32> to memref<{count}xf32>
    memref.dealloc %local : memref<{count}xf32>
    return
  }}
}}
"""


def inputs(function):
    values = list(
        np.linspace(
            -6 if function == "erf" else -105,
            6 if function == "erf" else 89,
            257,
            dtype=np.float32,
        )
    )
    boundaries = (
        [sign * p for sign in (-1, 1) for p in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4)]
        if function == "erf"
        else [-103.97208404541016, -126 * math.log(2), 88.72283935546875]
        + [(k + 0.5) * math.log(2) for k in (-150, -126, -1, 0, 1, 127, 128)]
    )
    for boundary in boundaries:
        center = np.float32(boundary)
        values += [
            center,
            np.nextafter(center, np.float32(-np.inf)),
            np.nextafter(center, np.float32(np.inf)),
        ]
    values += list(
        np.array(
            [
                0,
                0x80000000,
                1,
                0x80000001,
                0x007FFFFF,
                0x00800000,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
            ],
            dtype=np.uint32,
        ).view(np.float32)
    )
    return np.resize(
        np.array(values, np.float32), math.ceil(len(values) / 16) * 16
    )


def reference(function, values):
    def evaluate(value):
        try:
            return getattr(math, function)(float(value))
        except OverflowError:
            return math.inf

    return np.array([evaluate(value) for value in values], np.float64)


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def invoke(source, *passes):
        result = subprocess.run(
            [str(buddy_opt_path()), *passes],
            input=source,
            text=True,
            capture_output=True,
        )
        assert result.returncode >= 0, result.stderr
        assert (
            "LLVM ERROR" not in result.stderr
            and "PLEASE submit" not in result.stderr
        )
        return result

    def reject(source, *passes, message=None):
        result = invoke(source, *passes)
        assert result.returncode != 0 and "error:" in result.stderr, result
        if message:
            assert message in result.stderr, result.stderr

    passes = (
        "--kernel-contract-annotate",
        "--kernel-contract-verify",
        "--bosc-ame-plan=plan-version=1",
        "--bosc-ame-plan-verify",
    )
    for function in ("erf", "exp"):
        values = inputs(function)
        source = module(function, len(values))
        planned = invoke(source, *passes)
        assert planned.returncode == 0, planned.stderr
        assert 'fp_policy = "explicit-approximation"' in planned.stdout
        assert function + ".f32." in planned.stdout
        emitted = invoke(
            planned.stdout, "--bosc-ame-emit", "--bosc-ame-spm-dma-check"
        )
        assert emitted.returncode == 0, emitted.stderr
        assert (
            "math.erf" not in emitted.stdout
            and "math.exp" not in emitted.stdout
        )
        assert (
            "math.fma" not in emitted.stdout
            and "vector.fma" not in emitted.stdout
        )
        path = work / (function + ".mlir")
        path.write_text(emitted.stdout)
        guard = bytes([0xDA]) * 64
        raw = values.tobytes()
        buffers = {
            "%arg0": bytearray(raw + guard),
            "%arg1": bytearray(bytes(len(raw)) + guard),
        }
        machine = Machine(path, "compute")
        machine.spm[0][:] = bytes([0xA5]) * len(machine.spm[0])
        machine.spm[1][:] = bytes([0x5A]) * len(machine.spm[1])
        machine.launch(buffers)
        actual = np.frombuffer(buffers["%arg1"], np.float32, count=len(values))
        expected = reference(function, values)
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = expected.astype(np.float32)
        for predicate in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(predicate(actual), predicate(rounded))
        finite = np.isfinite(rounded)
        absolute, relative = (
            (2e-6, 0.0) if function == "erf" else (2.0**-146, 2e-6)
        )
        error = np.abs(actual[finite].astype(np.float64) - expected[finite])
        assert np.all(error <= absolute + relative * np.abs(expected[finite]))
        if function == "erf":
            zero = values == 0
            np.testing.assert_array_equal(
                actual.view(np.uint32)[zero], values.view(np.uint32)[zero]
            )
        else:
            np.testing.assert_array_equal(actual[values == 0], np.float32(1))
        assert (
            buffers["%arg0"] == raw + guard
            and buffers["%arg1"][len(raw) :] == guard
        )

        unauthorized = module(function, 16, False)
        neutral = invoke(
            unauthorized,
            "--kernel-contract-annotate",
            "--kernel-contract-verify",
        )
        assert neutral.returncode == 0 and f"math.{function}" in neutral.stdout
        reject(
            unauthorized,
            *passes,
            message="explicit contract.math accuracy budget",
        )
        reject(
            source.replace(
                f"math.{function} %x :", f"math.{function} %x fastmath<afn> :"
            ),
            *passes,
        )
        for bad in ('domain = "finite"', 'domain = "all-f64"'):
            reject(
                source.replace('domain = "all-f32"', bad),
                *passes,
                message="V26",
            )
        for bad in (
            "max_abs_error = -1.0 : f64",
            "max_abs_error = 0x7FF0000000000000 : f64",
            "max_abs_error = 2.0 : f32",
        ):
            malformed = re.sub(r"max_abs_error = [^,]+", bad, source)
            reject(malformed, *passes, message="V26")
        misplaced = unauthorized.replace(
            "module attributes {",
            'module attributes {contract.math = "misplaced",',
        )
        reject(misplaced, *passes, message="V26")
        misplaced = unauthorized.replace(
            "module attributes {",
            "module attributes {test.nested = {contract.math = {}},",
        )
        reject(misplaced, *passes, message="V26")
        reject(
            'module { func.func private @external() attributes {contract.math = "wrong"} }',
            "--kernel-contract-annotate",
            message="V26",
        )
        strict = re.sub(
            r"max_(abs|rel)_error = [^,}]+", r"max_\1_error = 0.0 : f64", source
        )
        reject(strict, *passes, message="accuracy budget")
        for old, new in [
            (function + ".f32.", "unverified.f32."),
            ("scalar-separate-mul-add-rne", "fma"),
            ('fp_policy = "explicit-approximation"', 'fp_policy = "preserve"'),
        ]:
            malformed = planned.stdout.replace(old, new)
            assert malformed != planned.stdout
            for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
                reject(malformed, gate)
        lowered = invoke(emitted.stdout, "--lower-bosc-ame")
        assert lowered.returncode == 0, lowered.stderr
        assert "erff" not in lowered.stdout and "expf" not in lowered.stdout
        llvm_path = work / (function + ".llvm.mlir")
        llvm_path.write_text(lowered.stdout)

        def command(arguments, input_text=None):
            result = subprocess.run(
                arguments, input=input_text, text=True, capture_output=True
            )
            assert result.returncode == 0, result.stderr
            return result.stdout

        def tool(name, directory):
            candidate = ROOT / directory / name
            return str(candidate) if candidate.is_file() else shutil.which(name)

        ll = work / (function + ".ll")
        command(
            [
                tool("buddy-translate", "build/bin"),
                "--buddy-to-llvmir",
                str(llvm_path),
                "-o",
                str(ll),
            ]
        )
        assembly = command(
            [
                tool("llc", "llvm/build/bin"),
                str(ll),
                "-O2",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-filetype=asm",
            ]
        )
        assert not re.search(
            r"\b(?:v?fmadd|v?fmsub|v?fnmadd|v?fnmsub)", assembly
        )
        encoded = command(
            [
                sys.executable,
                str(ROOT / "examples/FPGA-BOSCAME/tools/ame_to_word.py"),
            ],
            assembly,
        )
        restricted = command(
            [
                sys.executable,
                str(
                    ROOT
                    / "examples/FPGA-BOSCAME/tools/restrict_fpga_assembly.py"
                ),
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
            ],
            encoded,
        )
        assert "erff" not in restricted and "expf" not in restricted


if __name__ == "__main__":
    main()
