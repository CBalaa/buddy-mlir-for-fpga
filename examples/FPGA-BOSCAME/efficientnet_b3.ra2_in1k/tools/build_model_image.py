#!/usr/bin/env python3
# ===- build_model_image.py - graph + Triton lib to NR image ----*- python -*-===//
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
# Same board ABI as qwen3-0.6b/model/tools/build_model_image.py:
#   _start = 0x80000000, HIGH workspace 0xb8000000, weights in a labelled
#   NOLOAD arena, AME-encode the graph IR, link ld.lld + NR runtime +
#   libeffnet_triton.a. Do not link operator support.c.
#
# C ABI is result-first: _mlir_ciface_forward(&m_out, &p0..., &m_in).
# Launch fills NCHW input with board_input() and checks argmax vs
# host-reference.json. By default links kernel/op --wrap progress UART
# ([model]/[op]/[kernel]); pass --no-progress-uart for a quiet image.
#
# ===----------------------------------------------------------------------===//
"""Link the compiled EfficientNet graph into a bare-metal NR image."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def aligned(size):
    return (size + 63) & ~63


def workspace_block(name, size):
    return [
        f'__asm__(".section .workspace,\\"aw\\",@nobits\\n"',
        f'        ".balign 64\\n"',
        f'        "{name}_raw:\\n"',
        f'        ".skip {size}\\n"',
        f'        "{name}_end:\\n"',
        f'        ".previous\\n");',
        f'extern unsigned char {name}_raw[] __asm__("{name}_raw");',
        f'extern unsigned char {name}_end[] __asm__("{name}_end");',
    ]


def make_memref(name, rank, view, shape):
    dims = ", ".join(str(d) for d in shape)
    return f"  MemRef{rank} {name} = make_{rank}({view}, {dims});"


def main_source(layout, input_size, golden, *, progress_uart=True):
    params_bytes = aligned(layout["bytes"])
    input_elems = 1 * 3 * input_size * input_size
    output_elems = 1000
    args = []
    for index, section in enumerate(layout["sections"]):
        shape = section["shape"]
        rank = len(shape)
        view = f"weights + {section['offset_elements']}"
        args.append(make_memref(f"p{index}", rank, view, shape))
    args.append(make_memref("m_in", 4, "input_f", [1, 3, input_size, input_size]))
    args.append(make_memref("m_out", 2, "logits_f", [1, 1000]))
    n_params = len(layout["sections"])
    proto_args = ["MemRef2 *"] + [f"MemRef{len(s['shape'])} *" for s in layout["sections"]] + ["MemRef4 *"]
    proto = "extern void _mlir_ciface_forward(" + ", ".join(proto_args) + ");"
    call = "  _mlir_ciface_forward(&m_out, " + \
           ", ".join(f"&p{i}" for i in range(n_params)) + ", &m_in);"
    expect_class = golden.get("predicted_class")
    expect_bits = golden.get("max_logit_bits")
    check = []
    if expect_class is not None:
        check += [
            "  if (best != %du) {" % int(expect_class),
            '    nr_puts("[model] argmax mismatch\\r\\n");',
            "    ++errors;",
            "  }",
        ]
    if expect_bits is not None:
        check += [
            "  if (best_bits != 0x%08xu) {" % int(expect_bits),
            '    nr_puts("[model] top-logit bits mismatch\\r\\n");',
            "    ++errors;",
            "  }",
        ]
    profile_decl = (
        "void effnet_profile_reset(void);"
        if progress_uart else
        "static void effnet_profile_reset(void) {}"
    )
    return "\n".join([
        '#include "support.h"',
        '#include "nr_runtime.h"',
        "",
        *workspace_block("weight_arena", params_bytes),
        *workspace_block("input_arena", input_elems * 4),
        *workspace_block("output_arena", output_elems * 4),
        "static float *const weights = (float *)weight_arena_raw;",
        "static float *const input_f = (float *)input_arena_raw;",
        "static float *const logits_f = (float *)output_arena_raw;",
        "",
        proto,
        profile_decl,
        "",
        "int launch(void) {",
        '  nr_puts("[model] launch enter\\r\\n");',
        '  nr_puts("[model] weight_arena=");',
        "  nr_hex64((uint64_t)(uintptr_t)weight_arena_raw);",
        '  nr_puts(" bytes=");',
        f"  nr_hex64({params_bytes}ull);",
        '  nr_puts("\\r\\n");',
        '  nr_puts("[model] fill board_input begin\\r\\n");',
        f"  const unsigned input_n = {input_elems}u;",
        "  for (unsigned i = 0; i < input_n; ++i)",
        "    input_f[i] = (float)((int)((i * 13u) % 129u) - 64) * 0.125f;",
        "  for (unsigned i = 0; i < 1000u; ++i) logits_f[i] = 0.0f;",
        '  nr_puts("[model] fill board_input end; zero logits\\r\\n");',
        *args,
        '  nr_puts("[model] memrefs ready; param_sections=");',
        f"  nr_hex32({n_params}u);",
        '  nr_puts("\\r\\n");',
        "  effnet_profile_reset();",
        '  nr_puts("[model] forward begin\\r\\n");',
        "  uint64_t forward_begin = nr_cycles();",
        call,
        "  uint64_t forward_cycles = nr_cycles() - forward_begin;",
        '  nr_puts("[model] forward end cycles=");',
        "  nr_hex64(forward_cycles);",
        '  nr_puts("\\r\\n");',
        '  nr_puts("[model] argmax scan begin\\r\\n");',
        "  const float *out = (const float *)m_out.aligned + m_out.offset;",
        "  unsigned best = 0; float best_value = out[0];",
        "  for (unsigned i = 1; i < 1000u; ++i)",
        "    if (out[i] > best_value) { best_value = out[i]; best = i; }",
        "  union { float f; uint32_t u; } bits = { best_value };",
        "  uint32_t best_bits = bits.u;",
        '  nr_puts("[model] argmax=");',
        "  nr_hex32(best);",
        '  nr_puts(" logit_bits=");',
        "  nr_hex32(best_bits);",
        '  nr_puts("\\r\\n");',
        "  unsigned errors = 0;",
        *check,
        '  nr_puts("verify efficientnet_b3.ra2_in1k: ");',
        '  nr_puts(errors ? "FAIL\\r\\n" : "PASS\\r\\n");',
        '  nr_puts("[model] launch leave\\r\\n");',
        "  return errors != 0;",
        "}",
        "",
    ])


def run(command, **kwargs):
    redirected = any(key in kwargs for key in ("stdin", "stdout", "stderr"))
    if redirected:
        result = subprocess.run([str(c) for c in command], **kwargs)
        result.stdout = result.stdout or ""
        result.stderr = result.stderr or ""
        return result
    return subprocess.run([str(c) for c in command], capture_output=True, text=True, **kwargs)


def build(args, layout, golden):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    llvm_bin = args.repo_root / "llvm/build-2d26/bin"
    common = args.repo_root / "examples/FPGA-BOSCAME/common"
    nr = common / "nr"
    effnet = args.repo_root / "examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k"
    tools = args.repo_root / "examples/FPGA-BOSCAME/tools"
    model_tools = Path(__file__).resolve().parent
    params_bytes = aligned(layout["bytes"])
    progress_uart = not getattr(args, "no_progress_uart", False)
    source = output / "model_main.c"
    source.write_text(main_source(layout, args.input_size, golden,
                                  progress_uart=progress_uart))
    flags = ["--target=riscv64-unknown-elf", "-march=rv64gc_zicbom", "-mabi=lp64d",
             "-mcmodel=medany", "-O2", "-nostdlib", "-ffreestanding", "-fno-builtin",
             "-fno-pie", "-fno-vectorize", "-fno-slp-vectorize", "-ffp-contract=off",
             f"-I{effnet}", f"-I{nr}", f"-I{common}/uart"]
    steps = {}

    result = run([llvm_bin / "clang", *flags, "-c", source, "-o", output / "model_main.o"])
    steps["compile_main"] = {"returncode": result.returncode, "stderr": result.stderr[-1500:]}
    if result.returncode:
        return {"status": "FAILED compiling main", "steps": steps}

    assembly = output / "forward.s"
    encoded = output / "forward.encoded.s"
    restricted = output / "forward.nr.S"
    result = run([llvm_bin / "llc", args.graph_ir, "-O2", "-filetype=asm",
                  "-mtriple=riscv64", "-target-abi=lp64d",
                  "-mattr=+m,+a,+f,+d,+c,-v,+xboscame", "-code-model=medium",
                  "-o", assembly])
    steps["llc"] = {"returncode": result.returncode, "stderr": result.stderr[-1500:]}
    if result.returncode:
        return {"status": "FAILED compiling the graph IR", "steps": steps}
    result = run([sys.executable, tools / "ame_to_word.py"],
                 stdin=assembly.open("rb"), stdout=encoded.open("wb"))
    if result.returncode:
        return {"status": "FAILED encoding AME instructions",
                "steps": {"ame_to_word": {"returncode": result.returncode}}}
    result = run([sys.executable, tools / "restrict_fpga_assembly.py"],
                 stdin=encoded.open("rb"), stdout=restricted.open("wb"))
    if result.returncode:
        return {"status": "FAILED restricting assembly",
                "steps": {"restrict": {"returncode": result.returncode}}}
    graph_object = output / "forward.nr.o"
    result = run([llvm_bin / "clang", *flags, "-c", restricted, "-o", graph_object])
    steps["graph_object"] = {"returncode": result.returncode, "stderr": result.stderr[-1500:]}
    if result.returncode:
        return {"status": "FAILED assembling the graph", "steps": steps}

    result = run([llvm_bin / "clang", *flags, "-c", args.adapters,
                  "-o", output / "effnet_triton_adapters.o"])
    steps["compile_adapters"] = {"returncode": result.returncode, "stderr": result.stderr[-1500:]}
    if result.returncode:
        return {"status": "FAILED compiling adapters", "steps": steps}

    runtime_objects = []
    for name in ("crt.S", "nr_runtime.c", "nr_math.c", "ame_sync.c", "nr_copy.S"):
        obj = output / (Path(name).stem + ".o")
        source_flags = flags
        if name == "nr_copy.S":
            source_flags = ["-march=rv64gcv_zicbom" if f.startswith("-march=") else f
                            for f in flags]
        result = run([llvm_bin / "clang", *source_flags, "-c", nr / name, "-o", obj])
        if result.returncode:
            steps[f"runtime_{name}"] = {"returncode": result.returncode,
                                        "stderr": result.stderr[-800:]}
            return {"status": f"FAILED building NR runtime {name}", "steps": steps}
        runtime_objects.append(obj)

    profile_objects = []
    profile_flags = []
    if progress_uart:
        sys.path.insert(0, str(model_tools))
        from kernel_profile import generate_profile
        try:
            profile_source, profile_flags = generate_profile(
                args.adapters, output, progress=True, wrap_graph_ops=True)
        except ValueError as error:
            return {"status": f"FAILED generating kernel progress glue: {error}", "steps": steps}
        profile_obj = output / "kernel-profile.o"
        result = run([llvm_bin / "clang", *flags, "-c", profile_source, "-o", profile_obj])
        steps["compile_profile"] = {"returncode": result.returncode,
                                    "stderr": result.stderr[-1500:]}
        if result.returncode:
            return {"status": "FAILED compiling kernel progress glue", "steps": steps}
        profile_objects.append(profile_obj)

    elf = output / "effnet_model.elf"
    linker = llvm_bin / "ld.lld"
    result = run([linker, "-m", "elf64lriscv", "--gc-sections",
                  "-T", nr / "nr.ld", f"-Map={output / 'effnet_model.map'}",
                  *profile_flags,
                  "-o", elf,
                  output / "model_main.o", graph_object,
                  output / "effnet_triton_adapters.o",
                  *profile_objects,
                  *runtime_objects, args.archive])
    steps["link"] = {"returncode": result.returncode, "stderr": result.stderr[-3000:]}
    if result.returncode or not elf.is_file():
        return {"status": "FAILED linking the bare-metal image", "steps": steps}

    binary = output / "effnet_model.bin"
    run([llvm_bin / "llvm-objcopy", "-O", "binary", elf, binary])
    audit = run([sys.executable, tools / "check_nr_elf.py", elf,
                 "--objdump", llvm_bin / "llvm-objdump",
                 "--output", output / "elf-audit.json"])
    steps["audit"] = {"returncode": audit.returncode, "stdout": audit.stdout[-1500:],
                      "stderr": audit.stderr[-1500:]}
    undefined = run([llvm_bin / "llvm-nm", "--undefined-only", elf])
    return {
        "status": "PASS" if audit.returncode == 0 else "built but the ELF audit failed",
        "steps": steps,
        "progress_uart": progress_uart,
        "image": {"elf": str(elf), "bytes": elf.stat().st_size,
                  "bin": str(binary), "bin_bytes": binary.stat().st_size},
        "parameter_buffer_bytes": params_bytes,
        "undefined_symbols": sorted({line.split()[-1] for line in undefined.stdout.splitlines()
                                     if line.strip()}),
    }


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--graph-ir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--no-progress-uart", action="store_true",
                        help="disable [model]/[op]/[kernel] UART progress (default: on)")
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text())
    golden = {}
    if args.reference is not None and args.reference.is_file():
        report = json.loads(args.reference.read_text())
        golden["predicted_class"] = report.get("predicted_class")
        if report.get("logits_max_bits") is not None:
            golden["max_logit_bits"] = int(report["logits_max_bits"])
        elif report.get("logits_max") is not None:
            import struct
            golden["max_logit_bits"] = struct.unpack(
                "<I", struct.pack("<f", float(report["logits_max"])))[0]
    report = build(args, layout, golden)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "model-image.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "steps"},
                     indent=2))
    for name, step in report.get("steps", {}).items():
        if step.get("returncode"):
            print(f"--- {name} failed ---")
            print(step.get("stderr", "")[:1500])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(cli())
