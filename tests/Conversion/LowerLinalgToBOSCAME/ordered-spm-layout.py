# RUN: %PYTHON %s %t
"""Ordered f32 matmul keeps exact addresses in strided SPM operands."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def source(strides):
    matrices = (
        np.array([[1e20, 1, -1e20, 1], [2, -3, 4, -5]], dtype=np.float32),
        np.array([[1, 2], [1, 3], [1, 4], [1, 5]], dtype=np.float32),
        np.array([[7, 11], [-3, -5]], dtype=np.float32),
    )
    lines = [
        "module {",
        "func.func @ordered(%out: memref<32xf32>) {",
        "%zero = arith.constant 0 : index",
    ]
    for name, matrix, pitch, offset, bank, base in zip(
        "abc", matrices, strides, (1, 2, 2), (0, 1, 0), (0, 0, 256)
    ):
        storage = np.full(32, -213.0, dtype=np.float32)
        for index in np.ndindex(matrix.shape):
            storage[offset + sum(i * s for i, s in zip(index, pitch))] = matrix[
                index
            ]
        vector = ", ".join(repr(float(value)) for value in storage)
        shape = "x".join(str(d) for d in matrix.shape)
        stride = ", ".join(str(d) for d in pitch)
        type_ = (
            f"memref<{shape}xf32, strided<[{stride}], offset: {offset}>, "
            f"#bosc_ame.spm<{bank}>>"
        )
        lines.extend(
            (
                f"%{name}_data = arith.constant dense<[{vector}]> : vector<32xf32>",
                f"%{name}_flat = bosc_ame.spm.view "
                f"{{bank = {bank} : i32, offset = {base} : i64}} "
                f": memref<32xf32, #bosc_ame.spm<{bank}>>",
                f"vector.store %{name}_data, %{name}_flat[%zero] "
                f": memref<32xf32, #bosc_ame.spm<{bank}>>, vector<32xf32>",
                f"%{name} = memref.reinterpret_cast %{name}_flat "
                f"to offset: [{offset}], sizes: [{matrix.shape[0]}, {matrix.shape[1]}], "
                f"strides: [{stride}] "
                f": memref<32xf32, #bosc_ame.spm<{bank}>> to {type_}",
            )
        )
    a, b, c = [
        f"memref<{'x'.join(str(d) for d in matrix.shape)}xf32, "
        f"strided<[{', '.join(str(d) for d in pitch)}], offset: {offset}>, "
        f"#bosc_ame.spm<{bank}>>"
        for matrix, pitch, offset, bank in zip(
            matrices, strides, (1, 2, 2), (0, 1, 0)
        )
    ]
    lines.extend(
        (
            'linalg.matmul {contract.compute = {kind = "matmul", order = "mnk"}} '
            f"ins(%a, %b : {a}, {b}) outs(%c : {c})",
            "memref.copy %c_flat, %out : memref<32xf32, #bosc_ame.spm<0>> "
            "to memref<32xf32>",
            "return",
            "}",
            "}",
        )
    )
    expected = matrices[2].copy()
    with np.errstate(all="ignore"):
        for m in range(2):
            for n in range(2):
                for k in range(4):
                    expected[m, n] = np.float32(
                        expected[m, n]
                        + np.float32(matrices[0][m, k] * matrices[1][k, n])
                    )
    output = np.full(32, -213.0, dtype=np.float32)
    for index in np.ndindex(expected.shape):
        output[2 + sum(i * s for i, s in zip(index, strides[2]))] = expected[
            index
        ]
    return "\n".join(lines), output


def run(command, input_):
    result = subprocess.run(
        [str(arg) for arg in command],
        input=input_,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LLVM ERROR" not in result.stderr
    return result.stdout


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = Path(buddy_opt_path())
    for name, strides in (
        ("column-major", ((6, 1), (1, 7), (1, 5))),
        ("nonunit", ((12, 2), (2, 11), (3, 8))),
    ):
        input_, expected = source(strides)
        emitted = run(
            [
                opt,
                "--linalg-to-boscame=enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias",
                "--canonicalize",
                "--cse",
            ],
            input_,
        )
        assert "linalg.matmul" not in emitted
        assert "arith.mulf" in emitted and "arith.addf" in emitted
        assert "math.fma" not in emitted
        for line in emitted.splitlines():
            if re.search(r"memref\.(load|store)", line):
                assert "#bosc_ame.spm" not in line, line
        path = work / f"{name}.mlir"
        path.write_text(emitted)
        guard = bytes([0xD7]) * 64
        output = bytearray(bytes(128) + guard)
        machine = Machine(path, "ordered")
        machine.launch({"%arg0": output})
        assert output[:128] == expected.tobytes(), (name, output, expected)
        assert output[128:] == guard
        lowered = run([opt, "--lower-bosc-ame"], emitted)
        llvm = run(
            [opt.with_name("buddy-translate"), "--buddy-to-llvmir"], lowered
        )
        assert "fmul float" in llvm and "fadd float" in llvm
        assert "llvm.fma" not in llvm and "llvm.fmuladd" not in llvm


if __name__ == "__main__":
    main()
