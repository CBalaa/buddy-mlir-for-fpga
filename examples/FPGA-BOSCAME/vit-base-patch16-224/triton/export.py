#!/usr/bin/env python3
"""Compile real Triton Python ASTs into TTIR and triton-riscv Linalg IR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import ROOT, describe, inventory

REPO = ROOT.parents[3]


def configure_environment():
    plugin = REPO / "thirdparty/triton-riscv"
    os.environ.setdefault("TRITON_RISCV_DIR", str(plugin))
    os.environ.setdefault("TRITON_PLUGIN_DIRS", str(plugin))
    if not os.environ.get("TRITON_SHARED_OPT_PATH"):
        for checkout in (plugin / "triton", REPO / "thirdparty/triton"):
            candidates = sorted((checkout / "build").glob(
                "*/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"))
            if candidates:
                os.environ["TRITON_SHARED_OPT_PATH"] = str(candidates[-1])
                break


def make_patch_embed_adapter(case):
    """Compose im2col with the W8A8 AME projection.

    The graph-facing patch operator keeps its original four FP32 memrefs and
    NCHW output layout.  Its implementation materializes the non-overlapping
    patches as a row-major ``[196,768]`` matrix, quantizes activation and
    flattened OIHW weights, invokes the fixed-shape integer AME kernel, and
    dequantizes before writing the result back in the original layout.
    """
    if case["name"] != "patch_embed_196x768":
        raise ValueError("unexpected patch adapter case: " + case["name"])
    return "\n".join([
        '#include "support.h"',
        "typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;",
        # The first stage is still the Triton kernel.  Calling its raw ABI
        # here keeps im2col in the generated operator, rather than silently
        # moving the whole patch walk into this descriptor wrapper.
        "extern void triton_patch_embed_196x768(int64_t, MemRef0 *, int64_t, MemRef0 *, int64_t, MemRef0 *, int64_t, MemRef0 *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);",
        "extern void _mlir_ciface_kernel_quantize_196x768(MemRef2 *, MemRef2 *, MemRef1 *);",
        "extern void _mlir_ciface_kernel_quantize_768x768(MemRef2 *, MemRef2 *, MemRef1 *);",
        "extern void _mlir_ciface_kernel_matmul_i8_196x768x768(MemRef2 *, MemRef2 *, MemRef2 *);",
        "extern void _mlir_ciface_kernel_dequantize_196x768(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);",
        "static float patch_im2col[196 * 768] __attribute__((aligned(64)));",
        "static int8_t patch_im2col_q[196 * 768] __attribute__((aligned(64)));",
        "static float patch_im2col_scale[196] __attribute__((aligned(64)));",
        "static int8_t patch_weight_q[768 * 768] __attribute__((aligned(64)));",
        "static float patch_weight_scale[768] __attribute__((aligned(64)));",
        "static int32_t patch_acc[196 * 768] __attribute__((aligned(64)));",
        "static float patch_gemm[196 * 768] __attribute__((aligned(64)));",
        "",
        "void _mlir_ciface_kernel_patch_embed_196x768(MemRef4 *x, MemRef4 *weight,",
        "                                                   MemRef1 *bias, MemRef4 *out) {",
        "  const float *w = (const float *)weight->aligned + weight->offset;",
        "  const float *b = (const float *)bias->aligned + bias->offset;",
        "  float *dst = (float *)out->aligned + out->offset;",
        "  void *xp = (unsigned char *)x->aligned + x->offset * 4;",
        "  void *wp = (unsigned char *)weight->aligned + weight->offset * 4;",
        "  void *bp = (unsigned char *)bias->aligned + bias->offset * 4;",
        "  MemRef0 xm = {xp, xp, 0};",
        "  MemRef0 wm = {wp, wp, 0};",
        "  MemRef0 bm = {bp, bp, 0};",
        "  MemRef0 im = {patch_im2col, patch_im2col, 0};",
        "  for (int32_t p = 0; p < 196; ++p)",
        "    triton_patch_embed_196x768(0, &xm, 0, &wm, 0, &bm, 0, &im, 196, 1, 1, p, 0, 0);",
        "  MemRef2 im2 = make_2(patch_im2col, 196, 768);",
        "  MemRef2 im2q = make_2(patch_im2col_q, 196, 768);",
        "  MemRef1 ims = make_1(patch_im2col_scale, 196);",
        "  _mlir_ciface_kernel_quantize_196x768(&im2, &im2q, &ims);",
        "  MemRef2 wf = make_2((void *)w, 768, 768);",
        "  MemRef2 wq = make_2(patch_weight_q, 768, 768);",
        "  MemRef1 ws = make_1(patch_weight_scale, 768);",
        "  _mlir_ciface_kernel_quantize_768x768(&wf, &wq, &ws);",
        "  for (int i = 0; i < 196 * 768; ++i) patch_acc[i] = 0;",
        "  MemRef2 c = make_2(patch_acc, 196, 768);",
        "  _mlir_ciface_kernel_matmul_i8_196x768x768(&im2q, &wq, &c);",
        "  MemRef2 y = make_2(patch_gemm, 196, 768);",
        "  _mlir_ciface_kernel_dequantize_196x768(&c, &ims, &ws, &y);",
        "  for (int p = 0; p < 196; ++p)",
        "    for (int o = 0; o < 768; ++o)",
        "      dst[o * 196 + p] = patch_gemm[p * 768 + o] + b[o];",
        "}",
        "",
    ])


def make_adapter(case):
    """Descriptor/grid ABI glue for one Triton kernel or composed case."""
    if case["name"] == "patch_embed_196x768":
        return make_patch_embed_adapter(case)
    arguments = case["arguments"]
    # AME does not accept a short M tile.  For logical M values that are not
    # multiples of 16, compile the raw Triton kernel for the padded M and use
    # a small adapter-owned staging buffer.  The kernel's VALID_M constexpr
    # masks padded rows; only logical rows are copied back to the graph buffer.
    if case.get("family") == "matmul_i8" and case["shape"][0] % 16:
        m, n, _ = case["shape"]
        padded_m = case["constexprs"]["M"]
        if padded_m <= m or padded_m % 16:
            raise ValueError("invalid padded M for " + case["name"])
        symbol = case["symbol"]
        gx, gy, gz = case["grid"]
        return "\n".join([
            '#include "support.h"',
            'typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;',
            f'extern void {symbol}(' + ", ".join(["int64_t, MemRef0 *"] * 3 + ["int32_t"] * 6) + ');',
            f'static int8_t padded_a[{padded_m} * {case["shape"][2]}] __attribute__((aligned(64)));',
            f'static int32_t padded_c[{padded_m} * {n}] __attribute__((aligned(64)));',
            f'void _mlir_ciface_kernel_{case["name"]}(MemRef2 *a, MemRef2 *b, MemRef2 *c) {{',
            f'  const int8_t *src_a = (const int8_t *)a->aligned + a->offset;',
            f'  int32_t *dst_c = (int32_t *)c->aligned + c->offset;',
            f'  for (int i = 0; i < {padded_m} * {case["shape"][2]}; ++i) padded_a[i] = 0;',
            f'  for (int i = 0; i < {m}; ++i) for (int j = 0; j < {case["shape"][2]}; ++j)',
            f'    padded_a[i * {case["shape"][2]} + j] = src_a[i * a->strides[0] + j * a->strides[1]];',
            f'  for (int i = 0; i < {padded_m} * {n}; ++i) padded_c[i] = 0;',
            f'  for (int i = 0; i < {m}; ++i) for (int j = 0; j < {n}; ++j)',
            f'    padded_c[i * {n} + j] = ((const int32_t *)c->aligned + c->offset)[i * c->strides[0] + j * c->strides[1]];',
            f'  MemRef0 ma = {{padded_a, padded_a, 0}};',
            '  void *bp = (unsigned char *)b->aligned + b->offset;',
            '  MemRef0 mb = {bp, bp, 0};',
            '  MemRef0 mc = {padded_c, padded_c, 0};',
            f'  for (int32_t x = 0; x < {gx}; ++x)',
            f'    for (int32_t y = 0; y < {gy}; ++y)',
            f'      for (int32_t z = 0; z < {gz}; ++z)',
            f'        {symbol}(0, &ma, 0, &mb, 0, &mc, {gx}, {gy}, {gz}, x, y, z);',
            f'  for (int i = 0; i < {m}; ++i) for (int j = 0; j < {n}; ++j)',
            f'    dst_c[i * c->strides[0] + j * c->strides[1]] = padded_c[i * {n} + j];',
            '}', '',
        ])
    signature = ", ".join(f"MemRef{arg['rank']} *a{i}" for i, arg in enumerate(arguments))
    raw_signature = ", ".join(["int64_t, MemRef0 *"] * len(arguments) + ["int32_t"] * 6)
    lines = ['#include "support.h"',
             "typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;",
             f"extern void {case['symbol']}({raw_signature});",
             f"void _mlir_ciface_kernel_{case['name']}({signature}) {{"]
    item_sizes = {"f32": 4, "i8": 1, "i32": 4, "i64": 8}
    for i, arg in enumerate(arguments):
        lines += [f"  void *p{i} = (unsigned char *)a{i}->aligned + a{i}->offset * {item_sizes[arg['dtype']]};",
                  f"  MemRef0 m{i} = {{p{i}, p{i}, 0}};"]
    gx, gy, gz = case["grid"]
    params = ", ".join(f"0, &m{i}" for i in range(len(arguments)))
    lines += [f"  for (int32_t x=0; x<{gx}; ++x)",
              f"    for (int32_t y=0; y<{gy}; ++y)",
              f"      for (int32_t z=0; z<{gz}; ++z)",
              f"        {case['symbol']}({params}, {gx}, {gy}, {gz}, x, y, z);",
              "}", ""]
    return "\n".join(lines)


def compile_case(case, output_root):
    configure_environment()
    import triton
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared.compiler import CPUBackend, _ttir_to_ttsharedir
    from triton.compiler import ASTSource
    import importlib

    kernel_module = importlib.import_module(case.get("kernel_module", "kernels"))
    fn = getattr(kernel_module, case["kernel"])
    target = GPUTarget("cpu", 0, 0)
    backend = CPUBackend(target)
    options = backend.parse_options({"enable_fp_fusion": False})
    source = ASTSource(fn=fn, signature=case["signature"], constexprs=case["constexprs"])
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module = backend.make_ttir(module, {}, options)
    ttir = str(module)
    linalg = _ttir_to_ttsharedir(module)
    # This is only symbol namespacing, after genuine Triton/triton-riscv passes.
    # No computations, loop bodies or linalg operations are synthesized here.
    old_symbol = fn.__name__
    pattern = r"@" + re.escape(old_symbol) + r"(?=[\s(])"
    if not re.search(pattern, ttir) or not re.search(pattern, linalg):
        raise RuntimeError(f"Cannot identify compiled entry symbol {old_symbol!r}")
    ttir = re.sub(pattern, "@" + case["symbol"], ttir)
    linalg = re.sub(pattern, "@" + case["symbol"], linalg)
    output = Path(output_root) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "kernel.ttir").write_text(ttir)
    (output / "kernel.linalg.mlir").write_text(linalg)
    (output / "adapter.c").write_text(make_adapter(case))
    manifest = dict(case)
    manifest.update(frontend="triton.compiler.ASTSource -> CPUBackend.make_ttir -> triton-riscv _ttir_to_ttsharedir",
                    triton_version=triton.__version__,
                    kernels_sha256=hashlib.sha256((ROOT / "kernels.py").read_bytes()).hexdigest(),
                    kernel_module_sha256=hashlib.sha256(Path(kernel_module.__file__).read_bytes()).hexdigest(),
                    exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cases_sha256=hashlib.sha256((ROOT / "cases.py").read_bytes()).hexdigest(),
                    ttir_sha256=hashlib.sha256(ttir.encode()).hexdigest(),
                    linalg_sha256=hashlib.sha256(linalg.encode()).hexdigest(),
                    abi="unranked memref: (i64 rank=0, rank-0 descriptor pointer) per tensor, then six i32 grid/pid values")
    # The adapter is generated from the same case composition as the IR.  Keep
    # its digest in the manifest so a changed wrapper (for example, adding the
    # im2col -> linear composition for patch embedding) cannot be silently
    # reused by ``build.py`` merely because the old file still exists.
    manifest["adapter_sha256"] = hashlib.sha256(
        (output / "adapter.c").read_bytes()).hexdigest()
    (output / "frontend.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {case['name']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    cases = [describe(name) for name in args.cases] if args.cases else inventory()
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    for case in cases:
        compile_case(case, args.output)


if __name__ == "__main__":
    main()
