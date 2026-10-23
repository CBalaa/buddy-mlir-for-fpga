#!/usr/bin/env python3
"""Build the canonical standalone buffer-form BOSCAME artifacts.

Default: saved buffer-form MLIR -> BOSCAME -> LLVM dialect -> LLVM IR.
--check-abi compiles/links the ranked adapter and serial launcher as RV64 LLVM.
It also runs a host stub that checks the flattened descriptors and two calls.
--codegen additionally invokes llc; no board is contacted or image launched.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PASS_OPTIONS = (
    "enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias "
    "target=nr-fpga bm=64 bn=128 bk=1024"
)
FLAT_ABI = ["ptr", "ptr", "i64", "i64", "i64", "i64", "i64"] * 3
LAYERNORM_FLAT_ABI = ["ptr", "ptr", "i64", "i64", "i64"] * 4
CONTRACT = {
    "M": 128,
    "N": 128,
    "K": 1024,
    "BM": 64,
    "BN": 128,
    "BK": 1024,
    "grid": [2, 1],
    "program_id_source": "serial caller loop; no kernel program-id arguments",
    "A": {"shape": [128, 1024], "strides": [1024, 1], "element": "i8"},
    "B": {
        "logical_shape": [1024, 128],
        "physical_shape": [128, 1024],
        "logical_strides": [1, 1024],
        "element": "i8",
    },
    "C": {"shape": [128, 128], "strides": [128, 1], "element": "i32"},
    "abi": "three flattened rank-2 memrefs; 64-bit pointers and indices",
    "dma_abi": "args_aligned64_nonalias",
    "board_validation": "not performed",
}
LAYERNORM_CONTRACT = {
    "rows": 197,
    "valid_lanes": 768,
    "logical_lanes": 1024,
    "grid": [197, 1],
    "program_id_source": "serial caller row slicing; no kernel program-id arguments",
    "abi": "four flattened rank-1 memrefs; 64-bit pointers and indices",
    "per_program_shape": [768],
    "element": "f32",
    "other": {"X": 0, "Weight": 1, "Bias": 0},
    "sum_lanes": 1024,
    "sum_divisor": 768,
    "row_bytes": 3072,
    "dma_abi": "args_aligned64_nonalias",
    "board_validation": "not performed",
}


def find_tool(name, explicit, directories):
    if explicit:
        return str(explicit)
    for directory in directories:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    option = "host-cc" if name == "cc" else name
    raise SystemExit(f"missing {name}; specify --{option}")


def run(command):
    subprocess.run([str(part) for part in command], check=True)


def check_flat_abi(path, definition, symbol="gemm_program", expected=FLAT_ABI):
    kind = "define" if definition else "declare"
    match = re.search(
        rf"^{kind}\b[^\n]*\bvoid @{symbol}\(([^)]*)\)",
        path.read_text(),
        re.MULTILINE,
    )
    if not match:
        raise SystemExit(f"missing {kind} void @{symbol} in {path}")
    types = [argument.strip().split()[0] for argument in match[1].split(",")]
    if types != expected:
        raise SystemExit(f"unexpected @{symbol} ABI in {path}: {types}")


def check_c_abi(args, out, llvm_ir, kernel):
    clang = find_tool("clang", args.clang, [REPO / "llvm/build/bin"])
    llvm_link = find_tool(
        "llvm-link", args.llvm_link, [REPO / "llvm/build/bin"]
    )
    modules = []
    sources = (
        ("adapter", "launch") if kernel == "gemm" else ("layernorm_adapter",)
    )
    symbol = "gemm_program" if kernel == "gemm" else "layernorm_program"
    expected = FLAT_ABI if kernel == "gemm" else LAYERNORM_FLAT_ABI
    for source in sources:
        module = out / f"{source}.ll"
        run(
            [
                clang,
                "--target=riscv64-unknown-unknown",
                "-march=rv64imafdc_zicsr_zifencei",
                "-mabi=lp64d",
                "-std=c11",
                "-ffreestanding",
                "-fno-builtin",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-S",
                "-emit-llvm",
                HERE / f"{source}.c",
                "-o",
                module,
            ]
        )
        modules.append(module)
    check_flat_abi(
        modules[0], definition=False, symbol=symbol, expected=expected
    )
    run([llvm_link, llvm_ir, *modules, "-o", out / f"{kernel}.with-launch.bc"])
    host_cc = find_tool("cc", args.host_cc, [])
    check_source = "abi_check" if kernel == "gemm" else "layernorm_abi_check"
    abi_check = out / check_source
    run(
        [
            host_cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            *[HERE / f"{source}.c" for source in sources],
            HERE / f"{check_source}.c",
            "-o",
            abi_check,
        ]
    )
    run([abi_check])
    print(f"ABI CHECK PASS: {kernel}, RV64 LLVM link and host descriptor check")


def build_kernel(args, out, buddy_opt, buddy_translate, kernel):
    bufferized = out / f"{kernel}.bufferized.mlir"
    boscame = out / f"{kernel}.boscame.mlir"
    llvm_mlir = out / f"{kernel}.llvm.mlir"
    llvm_ir = out / f"{kernel}.ll"
    options = (
        PASS_OPTIONS
        if kernel == "gemm"
        else (
            "enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga"
        )
    )
    shutil.copyfile(HERE / f"{kernel}.bufferized.mlir", bufferized)
    run(
        [buddy_opt, bufferized, f"--linalg-to-boscame={options}", "-o", boscame]
    )
    run([buddy_opt, boscame, "--lower-bosc-ame", "-o", llvm_mlir])
    run([buddy_translate, llvm_mlir, "--buddy-to-llvmir", "-o", llvm_ir])
    check_flat_abi(
        llvm_ir,
        definition=True,
        symbol="gemm_program" if kernel == "gemm" else "layernorm_program",
        expected=FLAT_ABI if kernel == "gemm" else LAYERNORM_FLAT_ABI,
    )
    if args.check_abi:
        check_c_abi(args, out, llvm_ir, kernel)
    if args.codegen:
        llc = find_tool("llc", args.llc, [REPO / "llvm/build/bin"])
        run(
            [
                llc,
                llvm_ir,
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-code-model=medium",
                "-o",
                out / f"{kernel}.s",
            ]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel", choices=("gemm", "layernorm", "all"), default="gemm"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO / "build/examples/linalg-boscame"
    )
    parser.add_argument("--buddy-opt", type=Path)
    parser.add_argument("--buddy-translate", type=Path)
    parser.add_argument("--llc", type=Path)
    parser.add_argument("--clang", type=Path)
    parser.add_argument("--host-cc", type=Path)
    parser.add_argument("--llvm-link", type=Path)
    parser.add_argument("--check-abi", action="store_true")
    parser.add_argument("--codegen", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    buddy_opt = find_tool("buddy-opt", args.buddy_opt, [REPO / "build/bin"])
    buddy_translate = find_tool(
        "buddy-translate", args.buddy_translate, [REPO / "build/bin"]
    )
    kernels = ("gemm", "layernorm") if args.kernel == "all" else (args.kernel,)
    contracts = {
        kernel: CONTRACT if kernel == "gemm" else LAYERNORM_CONTRACT
        for kernel in kernels
    }
    selected = contracts[args.kernel] if len(kernels) == 1 else contracts
    (out / "contract.json").write_text(json.dumps(selected, indent=2) + "\n")
    for kernel in kernels:
        (out / f"{kernel}.contract.json").write_text(
            json.dumps(contracts[kernel], indent=2) + "\n"
        )
        build_kernel(args, out, buddy_opt, buddy_translate, kernel)
    print("Built standalone BOSCAME artifacts in " + str(out))


if __name__ == "__main__":
    main()
