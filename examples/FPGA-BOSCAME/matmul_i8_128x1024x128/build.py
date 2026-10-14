#!/usr/bin/env python3
"""Triton matmul_i8 [128,128] x [1024,128]^T -> BOSCAME NR FPGA image.

The makefile builds TTIR with the experimental BOSCAME path, then runs
``--lower-bosc-ame`` and LLVM translation. This file owns the real Triton
frontend, fixed shape, typed-unranked ABI adapter and host reference path.

    A[i8, 128, 128] row-major @ B[i8, 1024, 128] physical [N,K]
      -> C[i32, 128, 1024]

Each dot is materialized at its boundary; the outer K loop and original C
addition retain the Triton semantics. Ordinary aligned stack buffers stage
Linalg and AME packing. No caller scratch argument is passed to the new kernel.
The legacy ``--adapter triton-to-boscame`` generator remains available.

Build artifacts (``build/matmul_i8_128x1024x128/``) are intentionally limited
to IR and the final image:

    kernel.ttir  kernel.linalg.mlir  kernel.bufferized.mlir  kernel.ame.mlir
    kernel.stack.mlir  kernel.copies.mlir  kernel.llvm.mlir  kernel.ll
    matmul_i8_128x1024x128.bin

``kernel.ame.mlir``, ``kernel.boscame.mlir`` and ``kernel.llvm.mlir`` exist
only for ``AME=1``; the ``AME=0`` variant adds ``kernel.bufferized.mlir``,
``kernel.stack.mlir`` and ``kernel.copies.mlir`` instead.

Generated C, assembly, objects, the ELF, the audit report and the host oracle
binary live in ``build/.scratch/``.  That directory is removed after a
successful build and kept on failure for inspection.

Run with the ``boscame`` Python 3.12 environment:

    python build.py           # NR ELF audit + flat .bin
    python build.py --host    # host oracle built from the same LLVM IR
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
# The shared BOSCAME runtime moved to runtime/boscame during the migration.
RUNTIME_ROOT = REPO / "runtime" / "boscame"
COMMON = RUNTIME_ROOT
NR = RUNTIME_ROOT / "nr"
AME_RUNTIME = RUNTIME_ROOT / "runtime"
UART = RUNTIME_ROOT / "uart"
FPGA_TOOLS = ROOT.parent / "tools"
BUILD_ROOT = Path(
    os.environ.get("MATMUL_I8_BUILD_ROOT", str(ROOT / "build"))
).resolve()
OUT = BUILD_ROOT / "matmul_i8_128x1024x128"
SCRATCH = BUILD_ROOT / ".scratch"
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fixed problem size (the only place these numbers are defined).
# ---------------------------------------------------------------------------
M, N, K = 128, 1024, 128  # logical C[M,N] = A[M,K] @ B[N,K]^T
AME_M_TILE = 16  # AME msettilem is fixed at 16 rows on patch6
PADDED_M = (M + AME_M_TILE - 1) // AME_M_TILE * AME_M_TILE  # 128
VALID_M = M  # rows >= VALID_M are zero padding only
BM = 16  # Triton M block; PADDED_M / BM = 8 programs
BN = 16  # Triton N block; must match AME nr-tile-n
BK = 64  # K reduction block; K / BK = 2 iterations
GRID = (PADDED_M // BM, N // BN, 1)
# The pass transfers the accumulator tile of one program into a caller
# scratch window. It also holds one masked A/B slab during prologue packing.
# All views reuse offset zero, after the preceding DMA has completed.
SCRATCH_BYTES = max(
    BM * ((BN * 4 + 63) // 64 * 64), max(BM, BN) * ((BK + 63) // 64 * 64)
)
NAME = "matmul_i8_128x1024x128"
SYMBOL = "triton_" + NAME
KERNEL = "matmul_i8"
SIGNATURE = {"A": "*i8", "B": "*i8", "C": "*i32"}
CONSTEXPRS = {
    "M": PADDED_M,
    "VALID_M": VALID_M,
    "N": N,
    "K": K,
    "BM": BM,
    "BN": BN,
    "BK": BK,
}

assert PADDED_M % AME_M_TILE == 0 and PADDED_M >= M
assert K % BK == 0, "the reduction loop must not need a partial k block"


# ---------------------------------------------------------------------------
# Toolchain and flags.  Repository builds are preferred, then PATH; every
# value can be overridden with the environment variable of the same name.
# ---------------------------------------------------------------------------
def _env_list(name, default):
    value = os.environ.get(name)
    return shlex.split(value) if value else list(default)


def _tool_dir(candidates, executable):
    return next(
        (path for path in candidates if (path / executable).is_file()), None
    )


LLVM_BIN = (
    Path(os.environ["LLVM_BIN"]).resolve()
    if os.environ.get("LLVM_BIN")
    else _tool_dir(
        (REPO / "llvm/build-2d26/bin", REPO / "llvm/build/bin"), "clang"
    )
)


def llvm_tool(name):
    """Repository LLVM first, then PATH (same order as common/toolchain.mk)."""
    if LLVM_BIN is not None and (LLVM_BIN / name).is_file():
        return str(LLVM_BIN / name)
    return name


BUDDY_BIN = _tool_dir(
    (REPO / "build-migrate/bin", REPO / "build/bin"), "buddy-opt"
)
BUDDY_OPT = os.environ.get("BUDDY_OPT") or (
    str(BUDDY_BIN / "buddy-opt") if BUDDY_BIN else "buddy-opt"
)
BUDDY_TRANSLATE = os.environ.get("BUDDY_TRANSLATE") or (
    str(BUDDY_BIN / "buddy-translate") if BUDDY_BIN else "buddy-translate"
)
LLC = os.environ.get("LLC") or llvm_tool("llc")
HOST_CC = os.environ.get("HOST_CC") or llvm_tool("clang")
RISCV_CC = _env_list(
    "RISCV_CC", [llvm_tool("clang"), "--target=riscv64-unknown-elf"]
)
RISCV_LD = _env_list("RISCV_LD", ["ld.lld", "-m", "elf64lriscv"])
RISCV_OBJCOPY = os.environ.get("RISCV_OBJCOPY") or llvm_tool("llvm-objcopy")
RISCV_OBJDUMP = os.environ.get("RISCV_OBJDUMP") or llvm_tool("llvm-objdump")

# The integer AME kernel needs no RVV registers and must not acquire
# accidental vector memory operations.
SCALAR_FLAGS = _env_list("SCALAR_FLAGS", ["-mattr=+m,+a,+f,+d,+c,-v,+xboscame"])
CFLAGS = _env_list(
    "CFLAGS",
    [
        "-march=rv64gc_zicbom",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-nostdlib",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-pie",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
        "-ffunction-sections",
        "-fdata-sections",
        f"-I{ROOT}",
        f"-I{NR}",
        f"-I{AME_RUNTIME}",
        f"-I{UART}",
    ],
)
HOST_CFLAGS = _env_list(
    "HOST_CFLAGS", ["-O2", "-ffp-contract=off", "-DHOST_TEST", f"-I{ROOT}"]
)
LOWER = _env_list(
    "LOWER",
    [
        "--convert-linalg-to-loops",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-vector-to-scf",
        "--convert-vector-to-llvm",
        "--convert-math-to-llvm",
        "--convert-math-to-libm",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-index-to-llvm",
        "--memref-expand",
        "--finalize-memref-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts",
    ],
)
AME_PASS = _env_list("AME_PASS", ["--lower-linalg-to-boscame=target=nr-fpga"])
# Opt-in v0.5 operand encoding; the previously verified wrappers are default.
AME_GPR_MODE = os.environ.get("AME_GPR_MODE", "fixed")
# Experimental adjacent identical fence sharing; the ELF audit still runs.
NR_COALESCE_FENCES = os.environ.get("NR_COALESCE_FENCES", "0")
if AME_GPR_MODE not in ("fixed", "direct"):
    raise SystemExit("AME_GPR_MODE must be fixed or direct")
if NR_COALESCE_FENCES not in ("0", "1"):
    raise SystemExit("NR_COALESCE_FENCES must be 0 or 1")

# One NR runtime set per image: startup, platform, memory and operator support
# (common/README.md).  Extra flags are appended after CFLAGS, so they win.
RUNTIME_SOURCES = (
    (NR / "crt.S", "crt.o", []),
    (NR / "nr_runtime.c", "nr_runtime.o", []),
    (NR / "nr_memory.c", "nr_memory.o", []),
    (NR / "nr_math.c", "nr_math.o", []),
    (NR / "ame_sync.c", "ame_sync.o", []),
    (NR / "nr_copy.S", "nr_copy.o", ["-march=rv64gcv_zicbom"]),
    # The TTIR -> BOSCAME path stages DDR through real DMA, which the linalg
    # path did not use, so its runtime helper is part of the image now.
    (AME_RUNTIME / "bosc_ame_dma.c", "bosc_ame_dma.o", []),
    (ROOT / "support.c", "support.o", []),
)


def command(args, cwd=None, output=None, input_text=None):
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if output is not None:
        Path(output).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(shlex.join(map(str, args)) + "\n" + result.stdout)
    return result.stdout


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


def adapter_source():
    """Descriptor/grid glue around the raw Triton entry.

    AME does not accept a short M tile, so the kernel is compiled for
    ``PADDED_M`` while the public ABI stays ``[M,N]``.  ``M == PADDED_M`` for
    this shape, so the staging copy is the identity, but the adapter keeps the
    general form: it stages ``PADDED_M`` rows and copies only the logical ones
    back.

    This is the bridge of the *scalar* kernel (the ``AME=0`` image and the host
    oracle): three descriptors and no scratch, because no DMA/SPM is involved.
    ``-DMATMUL_I8_TRACE=1`` publishes the same per-program progress markers as
    the board bridge, so an ``AME=0`` debug image is observable too.
    """
    return (
        f"""#include "support.h"
#if defined(MATMUL_I8_TRACE)
extern void nh_e6_progress(uint32_t marker);
#define MATMUL_I8_MARK(value) nh_e6_progress(value)
#else
#define MATMUL_I8_MARK(value) ((void)0)
#endif
typedef struct {{ void *allocated, *aligned; int64_t offset; }} MemRef0;
extern void {SYMBOL}("""
        + ", ".join(["int64_t, MemRef0 *"] * 3 + ["int32_t"] * 6)
        + f""");
static int8_t padded_a[{PADDED_M} * {K}] __attribute__((aligned(64)));
static int32_t padded_c[{PADDED_M} * {N}] __attribute__((aligned(64)));
void _mlir_ciface_kernel_{NAME}(MemRef2 *a, MemRef2 *b, MemRef2 *c) {{
  const int8_t *src_a = (const int8_t *)a->aligned + a->offset;
  int32_t *dst_c = (int32_t *)c->aligned + c->offset;
  for (int i = 0; i < {PADDED_M} * {K}; ++i) padded_a[i] = 0;
  for (int i = 0; i < {M}; ++i) for (int j = 0; j < {K}; ++j)
    padded_a[i * {K} + j] = src_a[i * a->strides[0] + j * a->strides[1]];
  for (int i = 0; i < {PADDED_M} * {N}; ++i) padded_c[i] = 0;
  for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j)
    padded_c[i * {N} + j] = ((const int32_t *)c->aligned + c->offset)[i * c->strides[0] + j * c->strides[1]];
  MemRef0 ma = {{padded_a, padded_a, 0}};
  void *bp = (unsigned char *)b->aligned + b->offset;
  MemRef0 mb = {{bp, bp, 0}};
  MemRef0 mc = {{padded_c, padded_c, 0}};
  for (int32_t x = 0; x < {GRID[0]}; ++x)
    for (int32_t y = 0; y < {GRID[1]}; ++y)
      for (int32_t z = 0; z < {GRID[2]}; ++z) {{
        MATMUL_I8_MARK(0x1000u + (uint32_t)((x * {GRID[1]} + y) & 0xfffu));
        {SYMBOL}(0, &ma, 0, &mb, 0, &mc, {GRID[0]}, {GRID[1]}, {GRID[2]}, x, y, z);
      }}
  for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j)
    dst_c[i * c->strides[0] + j * c->strides[1]] = padded_c[i * {N} + j];
}}
"""
    )


def adapter_source_boscame():
    """Glue for the `--triton-to-boscame` ABI.

    The generated MLIR C wrapper takes four descriptor pointers and six grid
    scalars. Calling the raw expanded LLVM signature with C structs by value
    is incompatible with the RISC-V C ABI. Original buffers need no padding;
    the pass handles masked boundaries in the separate DDR scratch window.

    `-DMATMUL_I8_TRACE=1` publishes a progress marker before every program, so
    a board run shows which (program_id_x, program_id_y) is executing (and, if
    the platform reports TIMEOUT, which one stopped).
    """
    return f"""#include <stdint.h>
#if defined(MATMUL_I8_TRACE)
extern void nh_e6_progress(uint32_t marker);
#define MATMUL_I8_MARK(value) nh_e6_progress(value)
#else
#define MATMUL_I8_MARK(value) ((void)0)
#endif
typedef struct {{ void *allocated, *aligned; int64_t offset; int64_t sizes[2], strides[2]; }} MemRef2;
typedef struct {{ void *allocated, *aligned; int64_t offset, size, stride; }} Descriptor1D;
extern void _mlir_ciface_{SYMBOL}(Descriptor1D *, Descriptor1D *, Descriptor1D *, Descriptor1D *,
                     int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_{NAME}(MemRef2 *a, MemRef2 *b, MemRef2 *c,
                                void *scratch) {{
  /* The kernel computes byte addresses from the root, so every descriptor is
   * one contiguous byte range: A[M,K] i8, B[N,K] i8 and C[M,N] i32. */
  void *ap = (int8_t *)a->aligned + a->offset;
  void *bp = (int8_t *)b->aligned + b->offset;
  void *cp = (int32_t *)c->aligned + c->offset;
  Descriptor1D ma = {{a->allocated, ap, 0, (int64_t){M} * {K}, 1}};
  Descriptor1D mb = {{b->allocated, bp, 0, (int64_t){N} * {K}, 1}};
  Descriptor1D mc = {{c->allocated, cp, 0, (int64_t){M} * {N} * 4, 1}};
  /* The accumulator tile leaves SPM through a transfer (the E6 cannot read
   * the SPM window), so the kernel writes it here before the DDR-side
   * epilogue reads it back. */
  Descriptor1D ms = {{scratch, scratch, 0, (int64_t){SCRATCH_BYTES}, 1}};
  for (int32_t x = 0; x < {GRID[0]}; ++x)
    for (int32_t y = 0; y < {GRID[1]}; ++y)
      for (int32_t z = 0; z < {GRID[2]}; ++z) {{
        MATMUL_I8_MARK(0x1000u + (uint32_t)((x * {GRID[1]} + y) & 0xfffu));
        _mlir_ciface_{SYMBOL}(&ma, &mb, &mc, &ms, {GRID[0]}, {GRID[1]}, {GRID[2]}, x, y, z);
      }}
}}
"""


def adapter_source_experimental():
    """Typed unranked descriptors for the experimental BOSCAME C wrapper.

    Descriptor sizes/offsets are elements, including i32 C. The launcher facade
    retains its legacy scratch parameter, but the new kernel has three buffers.
    The launcher's aligned, nonalias contiguous buffers satisfy the DMA ABI.
    """
    return f"""#include <stdint.h>
#if defined(MATMUL_I8_TRACE)
extern void nh_e6_progress(uint32_t marker);
#define MATMUL_I8_MARK(value) nh_e6_progress(value)
#else
#define MATMUL_I8_MARK(value) ((void)0)
#endif
typedef struct {{ void *allocated, *aligned; int64_t offset; int64_t sizes[2], strides[2]; }} MemRef2;
typedef struct {{ void *allocated, *aligned; int64_t offset, size, stride; }} Descriptor1D;
typedef struct {{ int64_t rank; void *descriptor; }} Unranked;
extern void _mlir_ciface_{SYMBOL}(Unranked *, Unranked *, Unranked *,
                     int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_{NAME}(MemRef2 *a, MemRef2 *b, MemRef2 *c,
                                void *scratch) {{
  (void)scratch;
  void *ap = (int8_t *)a->aligned + a->offset;
  void *bp = (int8_t *)b->aligned + b->offset;
  void *cp = (int32_t *)c->aligned + c->offset;
  if (((uintptr_t)ap | (uintptr_t)bp | (uintptr_t)cp) & 63u) __builtin_trap();
  if (a->sizes[0] < {M} || a->sizes[1] < {K} ||
      b->sizes[0] < {N} || b->sizes[1] < {K} ||
      c->sizes[0] < {M} || c->sizes[1] < {N} ||
      a->strides[0] != {K} || a->strides[1] != 1 ||
      b->strides[0] != {K} || b->strides[1] != 1 ||
      c->strides[0] != {N} || c->strides[1] != 1) __builtin_trap();
  uintptr_t starts[3] = {{(uintptr_t)ap, (uintptr_t)bp, (uintptr_t)cp}};
  uintptr_t lengths[3] = {{{M} * {K}, {N} * {K}, {M} * {N} * 4}};
  for (int i = 0; i < 3; ++i) {{
    if (lengths[i] > UINTPTR_MAX - starts[i]) __builtin_trap();
    for (int j = 0; j < i; ++j)
      if (starts[i] < starts[j] + lengths[j] && starts[j] < starts[i] + lengths[i]) __builtin_trap();
  }}
  Descriptor1D da = {{a->allocated, ap, 0, (int64_t){M} * {K}, 1}};
  Descriptor1D db = {{b->allocated, bp, 0, (int64_t){N} * {K}, 1}};
  Descriptor1D dc = {{c->allocated, cp, 0, (int64_t){M} * {N}, 1}};
  Unranked ma = {{1, &da}}, mb = {{1, &db}}, mc = {{1, &dc}};
  for (int32_t x = 0; x < {GRID[0]}; ++x)
    for (int32_t y = 0; y < {GRID[1]}; ++y)
      for (int32_t z = 0; z < {GRID[2]}; ++z) {{
        MATMUL_I8_MARK(0x1000u + (uint32_t)((x * {GRID[1]} + y) & 0xfffu));
        _mlir_ciface_{SYMBOL}(&ma, &mb, &mc, {GRID[0]}, {GRID[1]}, {GRID[2]}, x, y, z);
      }}
}}
"""


def export_frontend():
    """Compile the real ``@triton.jit`` kernel into TTIR and shared linalg."""
    configure_environment()
    import kernels
    import triton
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared.compiler import (
        CPUBackend,
        _ttir_to_ttsharedir,
    )
    from triton.compiler import ASTSource

    fn = getattr(kernels, KERNEL)
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kernel.ttir").write_text(ttir)
    (OUT / "kernel.linalg.mlir").write_text(linalg)
    print(
        "exported " + NAME + " (triton " + triton.__version__ + ")", flush=True
    )


def llvm_pipeline():
    passes = list(LOWER)
    before = passes.index("--convert-scf-to-cf")
    additional = [
        "--convert-vector-to-scf",
        "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
        "--convert-ub-to-llvm",
    ]
    passes[before:before] = [
        option
        for option in additional
        if not any(
            existing.split("=")[0] == option.split("=")[0]
            for existing in passes
        )
    ]
    return passes


def ame_pipeline():
    """Keep the Triton N blocking and the NR hardware tiling in agreement."""
    passes = list(AME_PASS)
    candidates = [
        i
        for i, value in enumerate(passes)
        if value.startswith("--lower-linalg-to-boscame=")
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one explicit NR AME lowering pass")
    index = candidates[0]
    found = re.search(r"\bnr-tile-n=(\d+)", passes[index])
    if found and int(found[1]) != BN:
        raise ValueError("Triton BN and NR nr-tile-n disagree")
    if not found and BN != 16:
        passes[index] += " nr-tile-n=" + str(BN)
    return passes


def verify_abi(path):
    contents = Path(path).read_text()
    match = re.search(
        r"define\s+void\s+@" + re.escape(SYMBOL) + r"\((.*?)\)", contents, re.S
    )
    if not match:
        raise RuntimeError("Missing Triton entry definition: " + SYMBOL)
    actual = [part.strip().split()[0] for part in match.group(1).split(",")]
    expected = [value for _name in SIGNATURE for value in ("i64", "ptr")] + [
        "i32"
    ] * 6
    if actual != expected:
        raise RuntimeError(
            f"Triton ABI changed: expected {expected}, got {actual}"
        )
    if re.search(
        r"\b(?:call|declare)\b[^\n]*@(?:malloc|aligned_alloc|realloc|free)\b",
        contents,
    ):
        raise RuntimeError(
            "Unbounded per-grid heap allocation remains; stack promotion must remove it"
        )


def audit_elf(elf):
    """Run the shared NR ELF audit and fail the build on any finding."""
    report = SCRATCH / "audit.json"
    command(
        [
            sys.executable,
            FPGA_TOOLS / "check_nr_elf.py",
            elf,
            "--objdump",
            RISCV_OBJDUMP,
            "--output",
            report,
        ],
        cwd=ROOT,
    )
    audit = json.loads(report.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("errors")
        or not audit.get("ame_instructions")
    ):
        raise RuntimeError("NR ELF audit failed: " + json.dumps(audit))
    return audit


def lower(stage, host):
    """linalg -> memrefs -> AME -> LLVM dialect -> kernel.ll in ``stage``."""
    opt = shlex.split(BUDDY_OPT)
    translate = shlex.split(BUDDY_TRANSLATE)
    if not host:
        boscame = stage / "kernel.boscame.mlir"
        llvm_mlir = stage / "kernel.llvm.mlir"
        llvm_ir = stage / "kernel.ll"
        command(
            [
                *opt,
                OUT / "kernel.ttir",
                "--triton-to-linalg-experimental=boscame=true dma-abi=args_aligned64_nonalias",
                "-o",
                boscame,
            ],
            cwd=ROOT,
        )
        command([*opt, boscame, "--lower-bosc-ame", "-o", llvm_mlir], cwd=ROOT)
        command(
            [*translate, llvm_mlir, "--buddy-to-llvmir", "-o", llvm_ir],
            cwd=ROOT,
        )
        verify_abi(llvm_ir)
        return llvm_ir
    bufferized = stage / "kernel.bufferized.mlir"
    command(
        [
            *opt,
            OUT / "kernel.linalg.mlir",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true buffer-alignment=64",
            "-o",
            bufferized,
        ],
        cwd=ROOT,
    )
    lowered = bufferized
    promoted = stage / "kernel.stack.mlir"
    command(
        [
            *opt,
            lowered,
            "--canonicalize",
            "--cse",
            "--buffer-loop-hoisting",
            "--promote-buffers-to-stack=max-alloc-size-in-bytes=65536",
            "-o",
            promoted,
        ],
        cwd=ROOT,
    )
    copies = stage / "kernel.copies.mlir"
    command(
        [
            *opt,
            promoted,
            "--pass-pipeline=builtin.module(func.func(convert-strided-memref-copy-to-linalg))",
            "-o",
            copies,
        ],
        cwd=ROOT,
    )
    llvm_mlir = stage / "kernel.llvm.mlir"
    command(
        [
            *opt,
            copies,
            "--canonicalize",
            "--cse",
            *([] if host else ["--lower-bosc-ame"]),
            *llvm_pipeline(),
            "-o",
            llvm_mlir,
        ],
        cwd=ROOT,
    )
    llvm_ir = stage / "kernel.ll"
    command(
        [*translate, "--buddy-to-llvmir", llvm_mlir, "-o", llvm_ir], cwd=ROOT
    )
    verify_abi(llvm_ir)
    return llvm_ir


def run_host():
    """Build and run the host oracle from the very same LLVM IR."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "adapter.c").write_text(adapter_source())
    llvm_ir = lower(SCRATCH, host=True)
    image = SCRATCH / "host-check"
    command(
        [
            *shlex.split(HOST_CC),
            *HOST_CFLAGS,
            llvm_ir,
            SCRATCH / "adapter.c",
            ROOT / "launch-host.c",
            ROOT / "support.c",
            "-lm",
            "-o",
            image,
        ],
        cwd=ROOT,
    )
    result = command([image], output=SCRATCH / "host-output.log")
    if f"verify {NAME}: PASS" not in result:
        raise RuntimeError("Host numerical oracle did not report PASS")
    print("host PASS " + NAME, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        action="store_true",
        help="Build and run the host oracle instead of the NR image",
    )
    parser.add_argument(
        "--output", type=Path, help="Build root (defaults to build/)"
    )
    parser.add_argument(
        "--ttir-only",
        action="store_true",
        help="Only export kernel.ttir / kernel.linalg.mlir",
    )
    parser.add_argument(
        "--check-abi",
        type=Path,
        help="Verify the typed entry LLVM ABI and absence of heap allocation",
    )
    parser.add_argument(
        "--adapter",
        choices=(
            "experimental-boscame",
            "triton-to-boscame",
            "linalg-to-boscame",
        ),
        help="Only write the launcher adapter for that path",
    )
    parser.add_argument(
        "--llvm",
        choices=("ame", "noame"),
        help="Only lower kernel.linalg.mlir to kernel.ll: `ame` runs "
        "the experimental BOSCAME path, `noame` lowers straight to scalar CPU code "
        "(the AME=0 image and the host oracle share this path)",
    )
    parser.add_argument(
        "--stage",
        type=Path,
        help="Output directory for --llvm (defaults to the build output dir)",
    )
    args = parser.parse_args()
    if args.check_abi:
        verify_abi(args.check_abi)
        return
    global BUILD_ROOT, OUT, SCRATCH
    if args.output is not None:
        BUILD_ROOT = args.output.resolve()
        OUT = BUILD_ROOT / NAME
        SCRATCH = BUILD_ROOT / ".scratch"
    if args.adapter is not None:
        generator = {
            "experimental-boscame": adapter_source_experimental,
            "triton-to-boscame": adapter_source_boscame,
            "linalg-to-boscame": adapter_source,
        }[args.adapter]
        sys.stdout.write(generator())
        return
    if args.llvm is not None:
        stage = (args.stage or OUT).resolve()
        stage.mkdir(parents=True, exist_ok=True)
        lowered = lower(stage, host=args.llvm == "noame")
        print("lowered " + str(lowered), flush=True)
        return
    if args.ttir_only:
        (BUILD_ROOT / ".scratch").mkdir(parents=True, exist_ok=True)
        export_frontend()
        return
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        export_frontend()
        if args.host:
            run_host()
        else:
            raise SystemExit(
                "the NR image path moved to the makefile; "
                "use `make` or `make host`"
            )
    except Exception:
        print(
            "build failed; intermediates kept in " + str(SCRATCH),
            file=sys.stderr,
        )
        raise
    # All build artifacts the user asked to see stay in OUT (IR + image);
    # everything else is intermediate and removed here.
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
