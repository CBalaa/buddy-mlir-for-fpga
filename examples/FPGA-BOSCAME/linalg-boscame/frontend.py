#!/usr/bin/env python3
"""Export real canonical TTIR and run the independent buffer-form backend.

Requires the repository's Triton/triton-riscv Python environment. Generated
files live separately from the saved ranked buffer-form microkernel examples:
frontend/{gemm,layernorm}/kernel.{ttir,linalg.mlir,bufferized.mlir,boscame.mlir,
llvm.mlir,ll}. Runtime grid/pid parameters remain explicit in these artifacts.
"""

import argparse
import json
import os
import re
from pathlib import Path

from build import HERE, PASS_OPTIONS, REPO, check_flat_abi, find_tool, run


def check_frontend_abi(args, outputs):
    clang = find_tool("clang", args.clang, [REPO / "llvm/build/bin"])
    llvm_link = find_tool(
        "llvm-link", args.llvm_link, [REPO / "llvm/build/bin"]
    )
    adapter = args.out_dir.resolve() / "frontend_adapter.ll"
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
            HERE / "frontend_adapter.c",
            "-o",
            adapter,
        ]
    )
    for kernel, module in outputs:
        expected = ["ptr"] * (3 if kernel == "gemm" else 4) + ["i32"] * 6
        symbol = "_mlir_ciface_triton_" + kernel
        check_flat_abi(module, True, symbol=symbol, expected=expected)
        check_flat_abi(adapter, False, symbol=symbol, expected=expected)
        run(
            [
                llvm_link,
                module,
                adapter,
                "-o",
                module.parent / "kernel.with-launch.bc",
            ]
        )
    host_cc = find_tool("cc", args.host_cc, [])
    executable = args.out_dir.resolve() / "frontend_abi_check"
    run(
        [
            host_cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            HERE / "frontend_adapter.c",
            HERE / "frontend_abi_check.c",
            "-o",
            executable,
        ]
    )
    run([executable])
    print(
        "FRONTEND ABI PASS: actual C wrapper types, RV64 link, full-root descriptors and reverse grid calls"
    )


def configure_environment():
    plugin = REPO / "thirdparty/triton-riscv"
    os.environ.setdefault("TRITON_RISCV_DIR", str(plugin))
    os.environ.setdefault("TRITON_PLUGIN_DIRS", str(plugin))
    if not os.environ.get("TRITON_SHARED_OPT_PATH"):
        for checkout in (plugin / "triton", REPO / "thirdparty/triton"):
            candidates = sorted(
                (checkout / "build").glob(
                    "*/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
                )
            )
            if candidates:
                os.environ["TRITON_SHARED_OPT_PATH"] = str(candidates[-1])
                break


def export_ttir(kernel, destination):
    configure_environment()
    import kernels
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared.compiler import CPUBackend
    from triton.compiler import ASTSource

    target = GPUTarget("cpu", 0, 0)
    backend = CPUBackend(target)
    options = backend.parse_options({"enable_fp_fusion": False})
    signature = (
        {"A": "*i8", "B": "*i8", "C": "*i32"}
        if kernel == "gemm"
        else {"X": "*fp32", "Weight": "*fp32", "Bias": "*fp32", "Out": "*fp32"}
    )
    source = ASTSource(
        fn=getattr(kernels, kernel), signature=signature, constexprs={}
    )
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    module = backend.make_ttir(module, {}, options)
    text = re.sub(
        r"@" + kernel + r"(?=[\s(])", "@triton_" + kernel, str(module)
    )
    if "tt.func public @triton_" + kernel not in text:
        raise RuntimeError("missing real Triton entry " + kernel)
    destination.write_text(text)


def launch_grid_option(kernel, contract):
    grid = contract["grid"]
    if len(grid) != 3:
        raise ValueError("launch grid must have exactly three dimensions")
    return "launch-grid=triton_" + kernel + ":" + ",".join(map(str, grid))


def build_frontend(args, kernel):
    out = args.out_dir.resolve() / kernel
    out.mkdir(parents=True, exist_ok=True)
    ttir = out / "kernel.ttir"
    export_ttir(kernel, ttir)
    contract = {
        "kernel": kernel,
        "source": "examples/FPGA-BOSCAME/linalg-boscame/kernels.py",
        "grid": [2, 1, 1] if kernel == "gemm" else [197, 1, 1],
        "program_id_source": "tt.get_program_id; explicit runtime grid/pid arguments",
        "argument_abi": "typed unranked memrefs plus six i32 grid/pid arguments",
        "ranked_saved_buffer_abi_is_separate": True,
    }
    contract.update(
        {
            "M": 128,
            "N": 128,
            "K": 1024,
            "BM": 64,
            "BN": 128,
            "BK": 1024,
            "A_shape": [128, 1024],
            "B_logical_shape": [1024, 128],
            "B_physical_shape": [128, 1024],
            "C_shape": [128, 128],
        }
        if kernel == "gemm"
        else {
            "rows": 197,
            "valid_lanes": 768,
            "logical_lanes": 1024,
            "other": {"X": 0, "Weight": 1, "Bias": 0},
        }
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if args.stop_after == "ttir":
        return
    opt = find_tool("buddy-opt", args.buddy_opt, [REPO / "build/bin"])
    linalg = out / "kernel.linalg.mlir"
    run(
        [
            opt,
            ttir,
            "--triton-to-linalg-experimental=boscame=true dma-abi=args_aligned64_nonalias "
            + ("bm=64 bn=128 bk=1024 " if kernel == "gemm" else "")
            + "target=nr-fpga "
            + launch_grid_option(kernel, contract),
            "-o",
            linalg,
        ]
    )
    if "bosc_ame.spm.view" in linalg.read_text():
        raise RuntimeError("frontend crossed the buffer-form backend boundary")
    if args.stop_after == "linalg":
        return
    bufferized = out / "kernel.bufferized.mlir"
    run(
        [
            opt,
            linalg,
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries=true buffer-alignment=64",
            "--canonicalize",
            "--cse",
            "-o",
            bufferized,
        ]
    )
    if args.stop_after == "buffer":
        return
    boscame = out / "kernel.boscame.mlir"
    options = (
        PASS_OPTIONS
        if kernel == "gemm"
        else (
            "enable-spm-dma=true enable-ame=true "
            "dma-abi=args_aligned64_nonalias target=nr-fpga"
        )
    )
    run([opt, bufferized, f"--linalg-to-boscame={options}", "-o", boscame])
    if args.stop_after == "boscame":
        return
    llvm_mlir = out / "kernel.llvm.mlir"
    run([opt, boscame, "--lower-bosc-ame", "-o", llvm_mlir])
    translate = find_tool(
        "buddy-translate", args.buddy_translate, [REPO / "build/bin"]
    )
    llvm_ir = out / "kernel.ll"
    run([translate, llvm_mlir, "--buddy-to-llvmir", "-o", llvm_ir])
    return kernel, llvm_ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel", choices=("gemm", "layernorm", "all"), default="all"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "build/examples/linalg-boscame/frontend",
    )
    parser.add_argument("--buddy-opt", type=Path)
    parser.add_argument("--buddy-translate", type=Path)
    parser.add_argument("--clang", type=Path)
    parser.add_argument("--llvm-link", type=Path)
    parser.add_argument("--host-cc", type=Path)
    parser.add_argument("--check-abi", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("ttir", "linalg", "buffer", "boscame", "llvm"),
        default="llvm",
    )
    args = parser.parse_args()
    if args.check_abi and args.stop_after != "llvm":
        parser.error("--check-abi requires --stop-after=llvm")
    kernels = ("gemm", "layernorm") if args.kernel == "all" else (args.kernel,)
    outputs = [build_frontend(args, kernel) for kernel in kernels]
    if args.check_abi:
        check_frontend_abi(args, outputs)
    print("Exported real Triton artifacts in " + str(args.out_dir.resolve()))


if __name__ == "__main__":
    main()
