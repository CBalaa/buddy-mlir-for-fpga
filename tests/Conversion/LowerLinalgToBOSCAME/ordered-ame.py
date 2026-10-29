# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Check actual optimized AME assembly, including frontend-padded K tiles."""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from ame_to_word import transform  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def run(command):
    result = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def check_assembly(source, inner):
    registers = {"zero": 0}
    config = {}
    tiles = {}
    accum_live = False
    observed = []
    for line in source.splitlines():
        code = line.split("#", 1)[0].strip()
        fields = code.split(None, 1)
        if len(fields) != 2:
            continue
        name, operands = fields
        args = [part.strip() for part in operands.split(",")]
        if name == "li":
            registers[args[0]] = int(args[1], 0)
        elif name == "lui":
            registers[args[0]] = int(args[1], 0) << 12
        elif name in ("addi", "addiw") and args[1] in registers:
            registers[args[0]] = registers[args[1]] + int(args[2], 0)
        elif name == "slli" and args[1] in registers:
            registers[args[0]] = registers[args[1]] << int(args[2], 0)
        elif name == "mv" and args[1] in registers:
            registers[args[0]] = registers[args[1]]
        elif name in ("call", "tail"):
            assert not accum_live, "runtime call with a live accumulator"
            for register in tuple(registers):
                if register.startswith(("a", "t")):
                    del registers[register]
        elif name.startswith("mset"):
            assert args[1] in registers, "unknown configuration " + code
            config[name] = registers[args[1]]
        elif name == "mlce32.m":
            assert config["msettype"] == 0x10042
            assert not accum_live
            accum_live = True
        elif name in ("mlae8.m", "mlbe8.m"):
            assert config["msettype"] == 0x10010
            dimension = "msettilem" if name == "mlae8.m" else "msettilen"
            tiles[args[0]] = (config[dimension], config["msettilek"])
        elif name == "mqma.b.mm":
            assert accum_live and config["msettype"] == 0x10010
            shape = tuple(config["msettile" + dim] for dim in "mnk")
            assert tiles[args[1]] == (shape[0], shape[2]), (
                "A shape differs from current MMA configuration",
                tiles,
                shape,
            )
            assert tiles[args[2]] == (shape[1], shape[2]), (
                "B shape differs from current MMA configuration",
                tiles,
                shape,
            )
            observed.append(shape)
        elif name == "msce32.m":
            assert accum_live and config["msettype"] == 0x10042
            accum_live = False
        elif re.search(r"\b(?:tr|acc)\d+\b", operands):
            raise AssertionError("unsupported matrix move or spill: " + code)
    assert not accum_live
    expected = [(64, 64, min(64, inner - k)) for k in range(0, inner, 64)]
    assert observed == expected * 2, (observed, expected)
    # This must also fit the historical single-pair mapping. It rejects
    # preloading a second A/B pair before the first pair's MMA consumes it.
    transform(source, gpr_mode="fixed")


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = Path(buddy_opt_path())
    original = (Path(__file__).parent / "Inputs/ordered-ame.mlir").read_text()
    # Keep the historical raw K tail as an explicit boundary test. Frontends
    # pad that logical K=160 to 192 before this v1 backend boundary.
    tail = work / "raw-k160.mlir"
    tail.write_text(original.replace("1024", "160"))
    rejected = subprocess.run(
        [
            str(opt),
            str(tail),
            "--linalg-to-boscame=plan-version=1 dma-abi=args_aligned64_nonalias",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "AME plan requires frontend tiling/padding" in rejected.stderr, (
        rejected.stderr
    )
    for inner in (192, 1024):
        prefix = work / f"k{inner}"
        module = prefix.with_suffix(".mlir")
        module.write_text(original.replace("1024", str(inner)))
        lowered = prefix.with_suffix(".llvm.mlir")
        llvm = prefix.with_suffix(".ll")
        assembly = prefix.with_suffix(".s")
        run(
            [
                opt,
                module,
                "--linalg-to-boscame=plan-version=1 enable-spm-dma=true enable-ame=true "
                "dma-abi=args_aligned64_nonalias target=nr-fpga",
                "--lower-bosc-ame",
                "-o",
                lowered,
            ]
        )
        run(
            [
                opt.with_name("buddy-translate"),
                lowered,
                "--buddy-to-llvmir",
                "-o",
                llvm,
            ]
        )
        run(
            [
                REPO / "llvm/build/bin/llc",
                llvm,
                "-O2",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-o",
                assembly,
            ]
        )
        check_assembly(assembly.read_text(), inner)
    # CHECK: PASS ordered AME: optimized full/padded K configuration and live registers
    print(
        "PASS ordered AME: optimized full/padded K configuration and live registers"
    )


if __name__ == "__main__":
    main()
