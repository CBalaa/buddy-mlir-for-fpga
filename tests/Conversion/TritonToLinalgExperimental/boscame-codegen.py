# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Direct LLVM export and actual RVV/AME assembly, checked by the retained NR filter."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_check import buddy_opt_path  # noqa: E402


def run(command, input_text=None):
    result = subprocess.run(
        [str(x) for x in command],
        input=input_text,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = Path(buddy_opt_path())
    llc = REPO / "llvm/build/bin/llc"
    if not llc.is_file():
        llc = shutil.which("llc")
    tools = REPO / "examples/FPGA-BOSCAME/tools"
    for name, source, ame in (
        ("layernorm", "layernorm.ttir", False),
        ("gemm", "matmul-i8.ttir", True),
    ):
        module, lowered, llvm, assembly = (
            work / (name + suffix)
            for suffix in (".mlir", ".llvm.mlir", ".ll", ".s")
        )
        run(
            [
                opt,
                Path(__file__).parent / "Inputs" / source,
                "--triton-to-linalg-experimental=boscame=true dma-abi=args_aligned64_nonalias",
                "-o",
                module,
            ]
        )
        run([opt, module, "--lower-bosc-ame", "-o", lowered])
        run(
            [
                opt.with_name("buddy-translate"),
                lowered,
                "--buddy-to-llvmir",
                "-o",
                llvm,
            ]
        )
        text = llvm.read_text()
        assert not re.search(r"@(?:malloc|aligned_alloc|free)\b", text)
        assert (
            text.count("call i64 @bosc_ame_dma_start")
            == text.count("call void @bosc_ame_dma_wait")
            > 0
        )
        run(
            [
                llc,
                llvm,
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v" + (",+xboscame" if ame else ""),
                "-o",
                assembly,
            ]
        )
        encoded = assembly.read_text()
        if ame:
            encoded = run(
                [sys.executable, tools / "ame_to_word.py", "--gpr-mode=fixed"],
                encoded,
            )
        checked = run(
            [sys.executable, tools / "restrict_fpga_assembly.py"], encoded
        )
        (work / (name + ".nr.s")).write_text(checked)
        assert re.search(r"\bvle(?:8|32)\.v\b", checked) and re.search(
            r"\bvse(?:8|32)\.v\b", checked
        )
        assert not re.search(
            r"\bvlenb\b|\b(?:vs[1248]r|vl[1248]re\d+)\.v\b", checked
        )
        live = False
        for line in checked.splitlines():
            if re.search(r"#\s*mlce32", line):
                live = True
            if re.search(r"#\s*msce32", line):
                live = False
            if re.search(r"^\s*(?:call|tail)\s", line):
                assert not live, "call while AME accumulator is live: " + line
        assert not live
        assert bool(re.search(r"#\s*mqma\.b\.mm", checked)) == ame
    # CHECK: PASS codegen: direct LLVM translation, paired DMA calls, allowed fenced RVV, AME calls outside live ranges
    print(
        "PASS codegen: direct LLVM translation, paired DMA calls, allowed fenced RVV, AME calls outside live ranges"
    )


if __name__ == "__main__":
    main()
