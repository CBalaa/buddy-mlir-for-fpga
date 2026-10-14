#!/usr/bin/env python3
"""Triton layernorm_197x768 [197,768] x [768] -> TTIR.

This is the standalone migration of the ViT-B/16 LayerNorm operator, and it
owns the first stage of the FPGA pipeline:

  1. Triton AST -> TTIR                (real Triton/Triton-RISC-V passes)
       -> ``kernel.ttir``
  2. TTIR -> triton-riscv shared linalg
       -> ``kernel.linalg.mlir``

``kernel.ttir`` is what the downstream BOSCAME stages consume;
``kernel.linalg.mlir`` is the shared linalg form the scalar
(Triton-to-linalg) lowering starts from, exactly as in
``../matmul_i8_128x1024x128/build.py``, whose frontend this file mirrors.

Only the frontend is here: the fixed problem size, the output paths and the
Triton/triton-riscv environment.  No LLVM tool is involved, so the stage runs
with the ``boscame`` Python environment alone.

    layernorm_197x768: X[197,768] f32, Weight[768] f32, Bias[768] f32
                       -> Out[197,768] f32, one program per row
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BUILD_ROOT = Path(
    os.environ.get("LAYERNORM_197X768_BUILD_ROOT", str(ROOT / "build"))
).resolve()
OUT = BUILD_ROOT / "layernorm_197x768"
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fixed problem size (the only place these numbers are defined).
# ---------------------------------------------------------------------------
ROWS, WIDTH = 197, 768  # logical Out[ROWS,WIDTH] = layernorm(X[ROWS,WIDTH])
BLOCK = 1 << (WIDTH - 1).bit_length()  # 1024: next power of two >= WIDTH
EPS = 1.0e-12
GRID = (ROWS, 1, 1)  # one program per row
NAME = "layernorm_197x768"
SYMBOL = "triton_" + NAME
KERNEL = "layernorm"
SIGNATURE = {
    "X": "*fp32",
    "Weight": "*fp32",
    "Bias": "*fp32",
    "Out": "*fp32",
}
CONSTEXPRS = {"ROWS": ROWS, "WIDTH": WIDTH, "BLOCK": BLOCK, "EPS": EPS}

assert BLOCK >= WIDTH and BLOCK & (BLOCK - 1) == 0, "BLOCK must be a power of 2"


def configure_environment():
    """Point Triton at the triton-riscv plugin before ``import triton``."""
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


def verify_ttir(ttir):
    """Fail the export unless the TTIR still carries the operator's structure.

    Two ``tt.reduce`` (mean and variance over the hidden dimension) plus the
    ``math.rsqrt`` of the shifted variance are the whole point of the kernel;
    an export that loses one of them is not this operator.
    """
    missing = [
        token for token in ("@" + SYMBOL, "math.rsqrt") if token not in ttir
    ]
    if ttir.count('"tt.reduce"') != 2:
        missing.append("two tt.reduce")
    if ttir.count("tt.store") != 1:
        missing.append("one tt.store")
    if missing:
        raise RuntimeError(
            "exported TTIR is not " + NAME + ": missing " + ", ".join(missing)
        )


def export_frontend():
    """Compile the real ``@triton.jit`` kernel into TTIR and shared linalg."""
    configure_environment()
    import kernel
    import triton
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared.compiler import (
        CPUBackend,
        _ttir_to_ttsharedir,
    )
    from triton.compiler import ASTSource

    fn = getattr(kernel, KERNEL)
    target = GPUTarget("cpu", 0, 0)
    backend = CPUBackend(target)
    options = backend.parse_options({"enable_fp_fusion": False})
    source = ASTSource(fn=fn, signature=SIGNATURE, constexprs=CONSTEXPRS)
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
    ttir = str(module)
    linalg = _ttir_to_ttsharedir(module)
    # Symbol namespacing only, after genuine Triton/triton-riscv passes: no
    # computation, loop body or linalg op is synthesized here.
    pattern = r"@" + re.escape(KERNEL) + r"(?=[\s(])"
    if not re.search(pattern, ttir) or not re.search(pattern, linalg):
        raise RuntimeError(f"Cannot identify compiled entry symbol {KERNEL!r}")
    ttir = re.sub(pattern, "@" + SYMBOL, ttir)
    linalg = re.sub(pattern, "@" + SYMBOL, linalg)
    verify_ttir(ttir)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kernel.ttir").write_text(ttir)
    (OUT / "kernel.linalg.mlir").write_text(linalg)
    print(
        "exported " + NAME + " (triton " + triton.__version__ + ")", flush=True
    )


def adapter_source():
    """Emit the typed-unranked C wrapper bridge with an enforced DMA ABI."""
    return f"""#include <stdint.h>
#if defined(LAYERNORM_TRACE)
extern void nh_e6_progress(uint32_t marker);
#define LAYERNORM_MARK(value) nh_e6_progress(value)
#else
#define LAYERNORM_MARK(value) ((void)0)
#endif
typedef struct {{ void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; }} MemRef2;
typedef struct {{ void *allocated, *aligned; int64_t offset, size, stride; }} MemRef1;
typedef struct {{ int64_t rank; void *descriptor; }} Unranked;
extern void _mlir_ciface_{SYMBOL}(Unranked *, Unranked *, Unranked *, Unranked *,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_{NAME}(MemRef2 *x, MemRef1 *weight, MemRef1 *bias, MemRef2 *out) {{
  float *xp = (float *)x->aligned + x->offset;
  float *wp = (float *)weight->aligned + weight->offset;
  float *bp = (float *)bias->aligned + bias->offset;
  float *op = (float *)out->aligned + out->offset;
  if (((uintptr_t)xp | (uintptr_t)wp | (uintptr_t)bp | (uintptr_t)op) & 63u) __builtin_trap();
  if (x->sizes[0] < {ROWS} || x->sizes[1] < {WIDTH} ||
      out->sizes[0] < {ROWS} || out->sizes[1] < {WIDTH} ||
      weight->size < {WIDTH} || bias->size < {WIDTH} ||
      x->strides[0] != {WIDTH} || x->strides[1] != 1 ||
      out->strides[0] != {WIDTH} || out->strides[1] != 1 ||
      weight->stride != 1 || bias->stride != 1) __builtin_trap();
  uintptr_t starts[4] = {{(uintptr_t)xp, (uintptr_t)wp, (uintptr_t)bp, (uintptr_t)op}};
  uintptr_t lengths[4] = {{{ROWS} * {WIDTH} * 4, {WIDTH} * 4, {WIDTH} * 4, {ROWS} * {WIDTH} * 4}};
  for (int i = 0; i < 4; ++i) {{
    if (lengths[i] > UINTPTR_MAX - starts[i]) __builtin_trap();
    for (int j = 0; j < i; ++j)
      if (starts[i] < starts[j] + lengths[j] && starts[j] < starts[i] + lengths[i]) __builtin_trap();
  }}
  MemRef1 dx = {{x->allocated, xp, 0, (int64_t){ROWS} * {WIDTH}, 1}};
  MemRef1 dw = {{weight->allocated, wp, 0, {WIDTH}, 1}};
  MemRef1 db = {{bias->allocated, bp, 0, {WIDTH}, 1}};
  MemRef1 dout = {{out->allocated, op, 0, (int64_t){ROWS} * {WIDTH}, 1}};
  Unranked mx = {{1, &dx}}, mw = {{1, &dw}}, mb = {{1, &db}}, mo = {{1, &dout}};
  for (int32_t index = 0; index < {ROWS}; ++index) {{
#if defined(LAYERNORM_REVERSE)
    int32_t row = {ROWS} - 1 - index;
#else
    int32_t row = index;
#endif
    LAYERNORM_MARK(0x1000u + (uint32_t)row);
    _mlir_ciface_{SYMBOL}(&mx, &mw, &mb, &mo, {ROWS}, 1, 1, row, 0, 0);
  }}
}}
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Write the typed-unranked C ABI adapter to stdout",
    )
    parser.add_argument(
        "--ttir-only",
        action="store_true",
        help="Only export kernel.ttir / kernel.linalg.mlir",
    )
    parser.add_argument(
        "--output", type=Path, help="Build root (defaults to build/)"
    )
    args = parser.parse_args()
    if args.adapter:
        sys.stdout.write(adapter_source())
        return
    global BUILD_ROOT, OUT
    if args.output is not None:
        BUILD_ROOT = args.output.resolve()
        OUT = BUILD_ROOT / NAME
    (BUILD_ROOT / ".scratch").mkdir(parents=True, exist_ok=True)
    export_frontend()


if __name__ == "__main__":
    main()
