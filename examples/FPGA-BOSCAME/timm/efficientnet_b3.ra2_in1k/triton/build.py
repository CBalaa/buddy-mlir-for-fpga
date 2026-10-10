#!/usr/bin/env python3
# ===- build.py - host oracle and NR encode for Triton cases ----*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Host check first (--host), then NR encode. Reuses the parent launch.c
# and common NR runtime. Build root is triton/build or
# EFFNET_TRITON_BUILD_ROOT. Makefile print-config is cached per case
# directory so support.h include paths stay absolute.
#
# ===----------------------------------------------------------------------===//
"""Build Triton operator examples while reusing the parent launchers/runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import ROOT, PARENT, SMOKE, inventory
from export import compile_case

def ancestor(start, marker):
    for candidate in [start, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError("cannot find %s above %s" % (marker, start))

FPGA = ancestor(PARENT, "fpga_run.sh")
REPO = ancestor(PARENT, "examples/FPGA-BOSCAME")

BUILD_ROOT = Path(os.environ.get(
    "EFFNET_TRITON_BUILD_ROOT", str(ROOT / "build"))).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, cwd=None, output=None, input_text=None):
    result = subprocess.run([str(arg) for arg in args], cwd=cwd,
                            input=input_text, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if output is not None:
        Path(output).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(shlex.join(map(str, args)) + "\n" + result.stdout)
    return result.stdout


def selected(group, names):
    result = []
    for case in inventory():
        family = case["family"]
        take = (group == "all"
                or group == "ame" and family == "matmul_i8"
                or group == "conv" and family in ("conv2d_f32", "dwconv2d_f32")
                or group == "linear" and family in ("linear_f32", "matmul_f32")
                or group == "aux" and family not in (
                    "conv2d_f32", "dwconv2d_f32", "linear_f32",
                    "matmul_i8", "matmul_f32")
                or group == "smoke" and case["name"] in SMOKE)
        if names:
            take = case["name"] in names
        if take:
            result.append(case)
    if not result:
        raise ValueError("No matching cases")
    if names and set(names) - {case["name"] for case in result}:
        raise ValueError("Unknown case(s): " + ", ".join(
            sorted(set(names) - {case["name"] for case in result})))
    return result


def ensure_export(case):
    manifest = BUILD_ROOT / case["name"] / "frontend.json"
    fresh = False
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        module_path = ROOT / (case.get("kernel_module", "kernels") + ".py")
        fresh = (previous.get("kernels_sha256") == sha(ROOT / "kernels.py")
                 and previous.get("exporter_sha256") == sha(ROOT / "export.py")
                 and previous.get("cases_sha256") == sha(ROOT / "cases.py")
                 and previous.get("kernel_module_sha256", previous.get("kernels_sha256"))
                 == sha(module_path)
                 and previous.get("constexprs") == case["constexprs"]
                 and previous.get("grid") == case["grid"]
                 and previous.get("signature") == case["signature"]
                 and all((manifest.parent / filename).is_file()
                         and previous.get(key) == sha(manifest.parent / filename)
                         for key, filename in (("ttir_sha256", "kernel.ttir"),
                                               ("linalg_sha256", "kernel.linalg.mlir")))
                 and (manifest.parent / "adapter.c").is_file())
    if not fresh:
        compile_case(case, BUILD_ROOT)


def get_config(case, make_flags):
    """Read makefile print-config for this case directory.

    HOST_CFLAGS / CFLAGS embed ``-I$(ROOT)`` relative to the case dir. Model
    cases under ``model/build/.../model-cases`` override ROOT to a deeper
    relative path than root-level smoke cases, so configs must not be shared
    across different case directories.
    """
    directory = Path(case["directory"])
    data = command(["make", "--no-print-directory", "-s", "-C", directory,
                    "print-config", *make_flags])
    return json.loads(data)


def config_cache(make_flags):
    """Lazy per-directory config cache for mixed-depth case trees."""
    cache = {}

    def lookup(case):
        key = str(Path(case["directory"]).resolve())
        if key not in cache:
            cache[key] = get_config(case, make_flags)
        return cache[key]

    return lookup


def nr_ld_script(variables, directory):
    nr = Path(variables["NR"])
    if not nr.is_absolute():
        nr = (directory / nr).resolve()
    return nr / "nr.ld"


def llvm_pipeline(config):
    passes = shlex.split(config["variables"]["LOWER"])
    before = passes.index("--convert-scf-to-cf")
    additional = ["--convert-vector-to-scf",
                  "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
                  "--convert-ub-to-llvm"]
    passes[before:before] = [option for option in additional
                             if not any(existing.split("=")[0] == option.split("=")[0]
                                        for existing in passes)]
    return passes


def ame_pipeline(case, config):
    return shlex.split(config["variables"]["AME_PASS"])


def verify_abi(path, case):
    contents = Path(path).read_text()
    match = re.search(r"define\s+void\s+@" + re.escape(case["symbol"]) + r"\((.*?)\)",
                      contents, re.S)
    if not match:
        raise RuntimeError("Missing Triton entry definition: " + case["symbol"])
    actual = [part.strip().split()[0] for part in match.group(1).split(",")]
    expected = [value for _ in case["arguments"] for value in ("i64", "ptr")] + ["i32"] * 6
    if actual != expected:
        raise RuntimeError(f"Triton ABI changed: expected {expected}, got {actual}")
    if re.search(r"\b(?:call|declare)\b[^\n]*@(?:malloc|aligned_alloc|realloc|free)\b",
                 contents):
        raise RuntimeError("Unbounded per-grid heap allocation remains; stack promotion must remove it")


def audit_elf(elf, config, directory):
    variables = config["variables"]
    report = elf.with_suffix(".audit.json")
    command([*shlex.split(variables["PYTHON"]), FPGA / "tools/check_nr_elf.py",
             elf, "--objdump", variables["RISCV_OBJDUMP"], "--output", report],
            cwd=directory)
    command([*shlex.split(variables["RISCV_OBJDUMP"]), "-d", "--mattr=+v,+zicbom", elf],
            cwd=directory, output=elf.with_suffix(".disassembly"))
    return json.loads(report.read_text())


def source_provenance(cases, host):
    sources = {ROOT / filename for filename in ("kernels.py", "cases.py", "export.py", "build.py")}
    sources.update(PARENT / filename for filename in
                   ("support.c", "support.h", "common.mk", "tools/vectorize_nr.py",
                    "tools/build_suite.py", "tools/host_main.c"))
    sources.update(Path(case["directory"]) / filename for case in cases
                   for filename in ("launch.c", "metadata.json"))
    sources.update(FPGA / "common/triton" / filename for filename in
                   ("toolchain-lock.json", "setup-triton.sh", "triton-env.sh"))
    sources.add(FPGA / "common/toolchain.mk")
    if not host:
        sources.add(FPGA / "common/uart/uart.h")
        for pattern in ("*.c", "*.h", "*.S", "*.ld"):
            sources.update((FPGA / "common/nr").glob(pattern))
        sources.update(FPGA / "tools" / filename for filename in
                       ("ame_to_word.py", "restrict_fpga_assembly.py", "nr_isa.py",
                        "check_nr_elf.py"))
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(sources)}


def build_case(case, config, host, runtime_objects):
    directory = Path(case["directory"])
    variables = config["variables"]
    gpr_mode = variables.get("AME_GPR_MODE", "fixed")
    if gpr_mode not in ("fixed", "direct"):
        raise ValueError("AME_GPR_MODE must be fixed or direct")
    coalesce_setting = variables.get("NR_COALESCE_FENCES", "0")
    if coalesce_setting not in ("0", "1"):
        raise ValueError("NR_COALESCE_FENCES must be 0 or 1")
    coalesce_fences = coalesce_setting == "1"
    export_dir = BUILD_ROOT / case["name"]
    out = export_dir / ("host" if host else "nr")
    out.mkdir(parents=True, exist_ok=True)
    opt = shlex.split(variables["BUDDY_OPT"])
    translate = shlex.split(variables["BUDDY_TRANSLATE"])
    bufferized = out / "bufferized.mlir"
    command([*opt, export_dir / "kernel.linalg.mlir",
             "--empty-tensor-to-alloc-tensor",
             "--one-shot-bufferize=allow-return-allocs-from-loops=true buffer-alignment=64",
             "-o", bufferized], cwd=directory)
    lowered = bufferized
    if not host:
        lowered = out / "ame.mlir"
        effective_ame_pass = ame_pipeline(case, config)
        command([*opt, bufferized, *effective_ame_pass, "-o", lowered], cwd=directory)
        text = lowered.read_text()
        if 'bosc_ame.target = "nr-fpga"' not in text:
            raise RuntimeError("This runtime only accepts explicit nr-fpga lowering")
        if case["family"] == "matmul_i8" and (
                "bosc_ame.mqma.b.mm" not in text or "linalg.matmul" in text):
            raise RuntimeError("Integer Triton dot was not lowered to NR AME")
    else:
        effective_ame_pass = []
    promoted = out / "stack.mlir"
    command([*opt, lowered, "--canonicalize", "--cse", "--buffer-loop-hoisting",
             "--promote-buffers-to-stack=max-alloc-size-in-bytes=65536",
             "-o", promoted], cwd=directory)
    copies = out / "copies.mlir"
    command([*opt, promoted,
             "--pass-pipeline=builtin.module(func.func(convert-strided-memref-copy-to-linalg))",
             "-o", copies], cwd=directory)
    llvm_mlir = out / "kernel.llvm.mlir"
    command([*opt, copies, "--canonicalize", "--cse",
             *([] if host else ["--lower-bosc-ame"]),
             *llvm_pipeline(config), "-o", llvm_mlir], cwd=directory)
    llvm_ir = out / "kernel.ll"
    command([*translate, "--buddy-to-llvmir", llvm_mlir, "-o", llvm_ir], cwd=directory)
    verify_abi(llvm_ir, case)
    if host:
        cc = shlex.split(variables["HOST_CC"])
        # Absolute -I so support.h resolves even if a stale relative ROOT slips in.
        flags = [*shlex.split(variables["HOST_CFLAGS"]), "-I" + str(PARENT)]
        image = out / "check"
        command([*cc, *flags, llvm_ir, export_dir / "adapter.c", directory / "launch.c",
                 PARENT / "support.c", PARENT / "tools/host_main.c", "-lm", "-o", image],
                cwd=directory, output=out / "compile.log")
        result = command([image], output=out / "output.log")
        if f"verify {case['name']}: PASS" not in result:
            raise RuntimeError("Parent numerical oracle did not report PASS")
        print("host PASS " + case["name"], flush=True)
    else:
        cc = shlex.split(variables["RISCV_CC"])
        flags = [*shlex.split(variables["CFLAGS"]),
                 "-I" + str(PARENT), "-march=rv64gc_zicbom"]
        llc = shlex.split(variables["LLC"])
        assembly = out / "kernel.s"
        vector_flags = variables["RVV_FLAGS"]
        kernel_flags = ["-O2", "-filetype=asm", "-mtriple=riscv64", "-target-abi=lp64d",
                        *shlex.split(vector_flags), "-code-model=medium"]
        command([*llc, llvm_ir, *kernel_flags, "-o", assembly], cwd=directory)
        encoder = FPGA / "tools/ame_to_word.py"
        restrict = FPGA / "tools/restrict_fpga_assembly.py"
        python = shlex.split(variables["PYTHON"])
        encoded = command([*python, encoder, "--gpr-mode=" + gpr_mode],
                          input_text=assembly.read_text())
        constrained = command([*python, restrict,
                               *(["--coalesce-fences"] if coalesce_fences else [])],
                              input_text=encoded)
        nr_assembly = out / "kernel.nr.S"
        nr_assembly.write_text(constrained)
        objects = []
        for source, name in ((nr_assembly, "kernel"), (export_dir / "adapter.c", "adapter"),
                             (directory / "launch.c", "launch")):
            obj = out / (name + ".o")
            extra = ["-march=rv64gcv_zicbom_zvl512b"] if name == "kernel" else []
            command([*cc, *flags, *extra, "-c", source, "-o", obj], cwd=directory)
            objects.append(obj)
        ld = shlex.split(variables["RISCV_LD"])
        elf = out / (case["name"] + ".elf")
        image = out / (case["name"] + ".bin")
        command([*ld, "--gc-sections", "-T", nr_ld_script(variables, directory),
                 "-Map=" + str(out / "kernel.map"), "-o", elf, *objects, *runtime_objects],
                cwd=directory)
        elf_audit = audit_elf(elf, config, directory)
        command([*shlex.split(variables["RISCV_OBJCOPY"]), "-O", "binary", elf, image],
                cwd=directory)
        print("NR built " + case["name"], flush=True)
    record = {"case": case, "host": host, "image": os.path.relpath(image, ROOT),
              "sha256": sha(image), "configuration": config,
              "ttir_sha256": sha(export_dir / "kernel.ttir"),
              "frontend_manifest_sha256": sha(export_dir / "frontend.json"),
              "linalg_sha256": sha(export_dir / "kernel.linalg.mlir"),
              "llvm_sha256": sha(llvm_ir),
              "launch_sha256": sha(directory / "launch.c"),
              "adapter_sha256": sha(export_dir / "adapter.c"),
              "build_script_sha256": sha(__file__), "heap_allocations": 0,
              "source_files": source_provenance([case], host)}
    if not host:
        record["elf_audit"] = elf_audit
        record["effective_kernel_llc_flags"] = kernel_flags
        record["effective_ame_pass"] = effective_ame_pass
        record["ame_gpr_mode"] = gpr_mode
        record["nr_coalesce_fences"] = coalesce_fences
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def build_suite(cases, config_for, host, runtime_objects, group):
    out = BUILD_ROOT / ("suite-" + group + ("-host" if host else ""))
    out.mkdir(parents=True, exist_ok=True)
    declarations, calls, objects = [], [], []
    for index, case in enumerate(cases):
        name = case["name"]
        directory = Path(case["directory"])
        variables = config_for(case)["variables"]
        cc = shlex.split(variables["HOST_CC" if host else "RISCV_CC"])
        flags = [*shlex.split(variables["HOST_CFLAGS" if host else "CFLAGS"]),
                 "-I" + str(PARENT)]
        if not host:
            flags.append("-march=rv64gc_zicbom")
        symbol = "triton_launch_" + name
        declarations.append(f"int {symbol}(void);")
        calls.append(f'  nr_puts("[{index+1}/{len(cases)}] Triton {name} BEGIN\\r\\n"); failures += {symbol}() != 0;')
        export_dir = BUILD_ROOT / name
        obj = out / (name + "-launch.o")
        command([*cc, *flags, "-Dlaunch=" + symbol, "-c",
                 directory / "launch.c", "-o", obj], cwd=directory)
        objects.append(obj)
        if host:
            adapter = out / (name + "-adapter.o")
            command([*cc, *flags, "-c", export_dir / "adapter.c", "-o", adapter],
                    cwd=directory)
            objects += [adapter, export_dir / "host/kernel.ll"]
        else:
            objects += [export_dir / "nr/adapter.o", export_dir / "nr/kernel.o"]
    first = config_for(cases[0])
    variables = first["variables"]
    directory = Path(cases[0]["directory"])
    cc = shlex.split(variables["HOST_CC" if host else "RISCV_CC"])
    flags = [*shlex.split(variables["HOST_CFLAGS" if host else "CFLAGS"]),
             "-I" + str(PARENT)]
    if not host:
        flags.append("-march=rv64gc_zicbom")
    suite = out / "suite.c"
    suite.write_text('#include "support.h"\n' + "\n".join(declarations) +
                     "\nint launch(void) {\n  unsigned failures=0;\n" + "\n".join(calls) +
                     '\n  return print_check("efficientnet_b3 Triton operator suite",failures,0.0f);\n}\n')
    if host:
        image = out / "suite"
        command([*cc, *flags, suite, PARENT / "support.c", PARENT / "tools/host_main.c",
                 *objects, "-lm", "-o", image], cwd=directory)
        command([image], output=out / "output.log")
    else:
        obj = out / "suite.o"
        command([*cc, *flags, "-c", suite, "-o", obj], cwd=directory)
        elf, image = out / "suite.elf", out / "suite.bin"
        command([*shlex.split(variables["RISCV_LD"]), "--gc-sections",
                 "-T", nr_ld_script(variables, directory),
                 "-Map=" + str(out / "suite.map"), "-o", elf, obj, *objects,
                 *runtime_objects], cwd=directory)
        elf_audit = audit_elf(elf, first, directory)
        command([*shlex.split(variables["RISCV_OBJCOPY"]), "-O", "binary", elf, image],
                cwd=directory)
    record = {"group": group, "host": host, "frontend": "triton",
              "case_count": len(cases), "cases": [case["name"] for case in cases],
              "image": os.path.relpath(image, ROOT), "sha256": sha(image),
              "build_configuration": first,
              "source_files": source_provenance(cases, host)}
    if not host:
        record["elf_audit"] = elf_audit
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(image, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--group", choices=("all", "ame", "conv", "linear", "aux", "smoke"),
                        default="all")
    parser.add_argument("--host", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cases = selected(args.group, args.cases)
    for case in cases:
        ensure_export(case)
    make_flags = shlex.split(os.environ.get("EFFNET_MAKE_FLAGS", ""))
    config_for = config_cache(make_flags)
    runtime_objects = []
    if not args.host:
        # Shared NR runtime objects as absolute paths; any case tree is fine.
        directory = Path(cases[0]["directory"])
        config0 = config_for(cases[0])
        objects = shlex.split(config0["variables"]["RUNTIME_OBJS"])
        command(["make", "--no-print-directory", "-s", "-C", directory, *objects,
                 *make_flags])
        runtime_objects = [(directory / obj).resolve() for obj in objects]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(
            lambda case: build_case(case, config_for(case), args.host, runtime_objects),
            cases))
    record = {"group": args.group, "host": args.host, "count": len(results),
              "results": results}
    if args.suite:
        record["suite"] = build_suite(cases, config_for, args.host, runtime_objects,
                                      args.group)
    path = BUILD_ROOT / ("results-" + args.group +
                         ("-host" if args.host else "-nr") + ".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
