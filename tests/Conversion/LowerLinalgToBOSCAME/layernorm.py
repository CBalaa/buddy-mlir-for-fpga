# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Independent buffer Linalg LayerNorm preserves padding and DDR byte ranges."""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "examples/FPGA-BOSCAME/linalg-boscame"
sys.path.insert(0, str(EXAMPLE.parent / "tools"))
from boscame_check import buddy_opt_path  # noqa: E402


def run(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def main():
    prefix = Path(sys.argv[1])
    opt = buddy_opt_path()
    module = str(prefix) + ".boscame.mlir"
    run(
        [
            opt,
            str(EXAMPLE / "layernorm.bufferized.mlir"),
            "--lower-linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias",
            "-o",
            module,
        ]
    )
    text = Path(module).read_text()
    assert "tensor<" not in text and "unrealized_conversion_cast" not in text
    assert "bosc_ame.tile." not in text
    assert text.count("vector.reduction <add>") == 2
    assert "vector<1xf32>" not in text
    assert not re.search(r"memref\.alloca?[^\n]*: memref<f32>", text)
    assert not re.search(r"memref\.(?:load|store|copy)\b", text)
    for line in text.splitlines():
        if re.search(r"\bmemref\.(?:load|store|copy)\b", line):
            assert "#bosc_ame.spm" not in line, line
    run([opt, module, "--lower-bosc-ame", "-o", str(prefix) + ".llvm.mlir"])
    print(
        run(
            [
                sys.executable,
                str(EXAMPLE / "simcheck_layernorm.py"),
                "--module",
                module,
                "--buddy-opt",
                opt,
                "--quick",
            ]
        ),
        end="",
    )


# CHECK: SIM CHECK PASS: LayerNorm 3 rows, rtol=2e-5, atol=2e-5 against ordinary Linalg and padded ordered-f32 oracle
# CHECK-SAME: three 3072 B reads and one 3072 B write per row
# CHECK-SAME: last row, guards and persistent SPM checked
if __name__ == "__main__":
    main()
