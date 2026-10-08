#!/usr/bin/env python3
# ===- generate_cases.py - EfficientNet operator case emitter --*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Emit one directory per operator specialization (same layout as
# ../qwen3-0.6b generated cases):
#
#   <case>/kernel.mlir     bufferized linalg, llvm.emit_c_interface
#   <case>/metadata.json   shapes, kind, workspace layout
#   <case>/launch.c        independent C oracle (NCHW)
#   <case>/makefile        includes ../common.mk (relocatable)
#
# Buddy graph tensors stay NCHW (Conv2dOp layout NCHW_FCHW). The oracle,
# linalg kernels, and Triton JIT use the same layout so host check and the
# board compare the same numbers.
#
# Makefiles must stay relocatable: never write absolute checkout paths.
# Cases next to common.mk use the Qwen pattern ``include ../common.mk``.
# Cases emitted under another tree (model-shape cases) override ROOT with
# paths relative to that case directory.
#
# ===----------------------------------------------------------------------===//
"""Generate EfficientNet-B3 operator cases: linalg kernel.mlir + C oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PREAMBLE = r'''/*===- launch.c - EfficientNet operator oracle -------------------*- C -*-===*
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * See the LICENSE file at the repository root.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated C oracle. Do not hand-edit; regenerate with generate_cases.py.
 *
 *===----------------------------------------------------------------------===*/
#include "support.h"
#ifdef HOST_TEST
typedef double oracle_float;
static float oracle_exp(float x) { return (float)exp((double)x); }
#else
typedef float oracle_float;
static float oracle_exp(float x) { return expf(x); }
#endif
static void compare(float actual, float expected, float atol, float rtol,
                    int *errors, float *maximum) {
  if (actual == expected) return;
  if (!__builtin_isfinite(actual) || !__builtin_isfinite(expected)) {
    ++*errors; *maximum = __builtin_inff(); return;
  }
  if (!check_close(actual, expected, atol, rtol)) ++*errors;
  float d = actual - expected; if (d < 0) d = -d;
  if (d > *maximum) *maximum = d;
}
'''


def numel(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def typ(shape, dtype="f32"):
    return "memref<" + "x".join(map(str, shape)) + "x" + dtype + ">"


def amap(rank, results=None):
    dims = [f"d{i}" for i in range(rank)]
    mapped = dims if results is None else results
    return "affine_map<(" + ", ".join(dims) + ") -> (" + ", ".join(mapped) + ")>"


def generic(inputs, outputs, maps, iters, body):
    args = inputs + outputs
    txt = ("    linalg.generic {indexing_maps = [" + ", ".join(maps)
           + "], iterator_types = ["
           + ", ".join('"' + i + '"' for i in iters) + "]}\n")
    if inputs:
        txt += ("      ins(" + ", ".join(a[0] for a in inputs) + " : "
                + ", ".join(a[1] for a in inputs) + ")\n")
    txt += ("      outs(" + ", ".join(a[0] for a in outputs) + " : "
            + ", ".join(a[1] for a in outputs) + ") {\n")
    txt += ("    ^bb0(" + ", ".join(
        f"%v{i}: {a[2] if len(a) > 2 else 'f32'}"
        for i, a in enumerate(args)) + "):\n")
    return txt + body + "\n    }\n"


class Case:
    def __init__(self, name, kind, detail, extra=None):
        self.name, self.kind, self.detail = name, kind, detail
        self.extra = extra or {}
        self.buffers, self.offset = [], 0
        self.init, self.verify, self.body = [], [], []

    def buffer(self, name, shape, dtype="f32"):
        ctype, size = {"f32": ("float", 4), "i8": ("int8_t", 1),
                       "i32": ("int32_t", 4)}[dtype]
        self.offset = (self.offset + 63) // 64 * 64
        offset = self.offset
        self.offset += numel(shape) * size
        self.buffers.append((name, tuple(shape), dtype, offset, ctype))
        return ("%" + name, typ(shape, dtype), dtype)

    def emit(self, root: Path):
        directory = root / self.name
        directory.mkdir(parents=True, exist_ok=True)
        function = "kernel_" + self.name
        args = ", ".join(
            "%" + name + ": " + typ(shape, dtype)
            for name, shape, dtype, _, _ in self.buffers)
        mlir = (
            "// Generated by tools/generate_cases.py.\n"
            "module {\n  func.func @" + function + "(" + args
            + ") attributes {llvm.emit_c_interface} {\n"
            + "".join(self.body) + "    return\n  }\n}\n"
        )
        proto = "extern void _mlir_ciface_" + function + "(" + ", ".join(
            f"MemRef{len(shape)} *" for _, shape, *_ in self.buffers) + ");\n"
        launch = PREAMBLE + proto + "int launch(void) {\n  int errors = 0; float maximum = 0;\n"
        for name, shape, _, offset, ctype in self.buffers:
            launch += f"  {ctype} *{name} = ({ctype} *)workspace({offset}u);\n"
            launch += f"  MemRef{len(shape)} m_{name} = make_{len(shape)}({name}, " + \
                      ", ".join(map(str, shape)) + ");\n"
        launch += "\n".join(self.init) + "\n  _mlir_ciface_" + function + "(" + \
                  ", ".join("&m_" + n for n, *_ in self.buffers) + ");\n"
        launch += "\n".join(self.verify)
        launch += f'\n  return print_check("{self.name}", errors, maximum);\n}}\n'
        meta = {
            "name": self.name, "kind": self.kind, "dtype": "f32",
            "description": self.detail, "entry": function,
            "workspace_bytes": self.offset,
            "buffers": [{"name": n, "shape": list(s), "dtype": d, "offset": o}
                        for n, s, d, o, _ in self.buffers],
            "validation": "All output elements versus an independent C oracle; host oracle uses libm double precision.",
            **self.extra,
        }
        (directory / "kernel.mlir").write_text(mlir)
        (directory / "launch.c").write_text(launch)
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        (directory / "makefile").write_text(case_makefile(directory))
        return meta


def case_makefile(case_dir: Path):
    """Write a makefile that works after the tree is moved or packaged.

    Sibling cases (``efficientnet_b3.ra2_in1k/<name>/``) match Qwen:
    ``include ../common.mk``. Cases generated under another directory
    (for example ``model/build/model-cases/<name>/``) override ROOT using
    paths relative to *that* case so reviewers are not locked to one
    absolute checkout.
    """
    example_root = Path(__file__).resolve().parents[1]
    header = "\n".join([
        "##===- makefile - generated EfficientNet case ----------*- Makefile -*-===//",
        "##",
        "## Licensed under the Apache License, Version 2.0.",
        "## See the LICENSE file at the repository root.",
        "## SPDX-License-Identifier: Apache-2.0",
        "##",
        "## Generated file. Do not hand-edit; regenerate with generate_cases.py.",
        "##",
        "##===----------------------------------------------------------------------===//",
        "",
    ])
    try:
        if case_dir.resolve().parent == example_root.resolve():
            return header + "include ../common.mk\n"
    except OSError:
        pass
    rel_root = os.path.relpath(example_root, case_dir)
    rel_common = os.path.relpath(example_root.parent / "common", case_dir)
    rel_repo = os.path.relpath(example_root.parents[2], case_dir)
    rel_tools = os.path.relpath(example_root / "tools", case_dir)
    rel_mk = os.path.relpath(example_root / "common.mk", case_dir)
    return header + "\n".join([
        ".DEFAULT_GOAL := print-config",
        f"override ROOT := {rel_root}",
        f"override COMMON := {rel_common}",
        f"override REPO_ROOT := {rel_repo}",
        f"override TOOLS := {rel_tools}",
        f"include {rel_mk}",
        "",
    ])


def pointwise(root, kind, shape):
    name = f"{kind}_" + "x".join(map(str, shape))
    c = Case(name, kind, f"{kind} NCHW/flattened elementwise")
    x = c.buffer("x", shape)
    y = c.buffer("y", shape) if kind in ("add", "mul") else None
    out = c.buffer("out", shape)
    n = numel(shape)
    extra = " y[i] = (float)((i*7)%31-15)*0.0625f;" if y else ""
    c.init += [f"  for (int i=0; i<{n}; ++i) {{ x[i]=(float)((i*13)%129-64)*0.125f; out[i]=9999.0f;{extra} }}"]
    rank = len(shape)
    maps = [amap(rank)] * (2 if y else 1) + [amap(rank)]
    iters = ["parallel"] * rank
    if kind == "silu":
        body = (
            "      %one = arith.constant 1.0 : f32\n"
            "      %neg = arith.negf %v0 : f32\n"
            "      %e = math.exp %neg : f32\n"
            "      %den = arith.addf %one, %e : f32\n"
            "      %r = arith.divf %v0, %den : f32\n"
            "      linalg.yield %r : f32"
        )
        ref = "x[i] / (1.0f + oracle_exp(-x[i]))"
        inputs = [x]
    elif kind == "sigmoid":
        body = (
            "      %one = arith.constant 1.0 : f32\n"
            "      %neg = arith.negf %v0 : f32\n"
            "      %e = math.exp %neg : f32\n"
            "      %den = arith.addf %one, %e : f32\n"
            "      %r = arith.divf %one, %den : f32\n"
            "      linalg.yield %r : f32"
        )
        ref = "1.0f / (1.0f + oracle_exp(-x[i]))"
        inputs = [x]
    elif kind == "mul":
        body = "      %r = arith.mulf %v0, %v1 : f32\n      linalg.yield %r : f32"
        ref = "x[i] * y[i]"
        inputs = [x, y]
    else:
        body = "      %r = arith.addf %v0, %v1 : f32\n      linalg.yield %r : f32"
        ref = "x[i] + y[i]"
        inputs = [x, y]
    c.body += [generic(inputs, [out], maps, iters, body)]
    c.verify += [f"  for (int i=0; i<{n}; ++i) compare(out[i], {ref}, 2e-6f, 3e-5f, &errors, &maximum);"]
    return c.emit(root)


def gap(root, h, w, channels):
    name = f"gap_{h}x{w}x{channels}"
    c = Case(name, "gap", "Global average pool NCHW [1,C,H,W] -> [1,C]")
    x = c.buffer("x", (1, channels, h, w))
    out = c.buffer("out", (1, channels))
    spat = h * w
    c.init += [
        f"  for (int i=0; i<{spat * channels}; ++i) x[i]=(float)((i*5)%41-20)*0.03125f;",
        f"  for (int i=0; i<{channels}; ++i) out[i]=9999.0f;",
    ]
    c.body += [
        "    %zero = arith.constant 0.0 : f32\n"
        f"    linalg.fill ins(%zero : f32) outs({out[0]} : {out[1]})\n"
    ]
    c.body += [generic(
        [x], [out],
        [amap(4), amap(4, ["d0", "d1"])],
        ["parallel", "parallel", "reduction", "reduction"],
        "      %s = arith.addf %v0, %v1 : f32\n      linalg.yield %s : f32",
    )]
    c.body += [generic(
        [], [out], [amap(2)], ["parallel", "parallel"],
        f"      %denom = arith.constant {spat}.0 : f32\n"
        "      %r = arith.divf %v0, %denom : f32\n"
        "      linalg.yield %r : f32",
    )]
    c.verify += [
        f"  for (int ch=0; ch<{channels}; ++ch) {{",
        "    oracle_float s=0;",
        f"    for (int p=0; p<{spat}; ++p) s += x[ch*{spat}+p];",
        f"    compare(out[ch], (float)(s/{spat}.0), 3e-6f, 5e-5f, &errors, &maximum);",
        "  }",
    ]
    return c.emit(root)


def se_scale(root, h, w, channels):
    name = f"se_scale_{h}x{w}x{channels}"
    c = Case(name, "se_scale", "Broadcast channel scale onto NCHW activations")
    x = c.buffer("x", (1, channels, h, w))
    scale = c.buffer("scale", (1, channels))
    out = c.buffer("out", (1, channels, h, w))
    spat = h * w
    n = spat * channels
    c.init += [
        f"  for (int i=0; i<{n}; ++i) {{ x[i]=(float)((i*9)%29-14)*0.0625f; out[i]=9999.0f; }}",
        f"  for (int i=0; i<{channels}; ++i) scale[i]=(float)((i%13)+1)*0.125f;",
    ]
    c.body += [generic(
        [x, scale], [out],
        [amap(4), amap(4, ["d0", "d1"]), amap(4)],
        ["parallel"] * 4,
        "      %r = arith.mulf %v0, %v1 : f32\n      linalg.yield %r : f32",
    )]
    c.verify += [
        f"  for (int ch=0; ch<{channels}; ++ch) for (int p=0; p<{spat}; ++p)",
        f"    compare(out[ch*{spat}+p], x[ch*{spat}+p]*scale[ch], 2e-6f, 3e-5f, &errors, &maximum);"
    ]
    return c.emit(root)


def linear(root, m, n, k):
    name = f"linear_{m}x{n}x{k}"
    c = Case(name, "linear_f32", "A[M,K] * W[N,K]^T + bias -> [M,N]",
             extra={"shape": [m, n, k]})
    a = c.buffer("a", (m, k))
    w = c.buffer("w", (n, k))
    bias = c.buffer("bias", (n,))
    out = c.buffer("out", (m, n))
    c.init += [
        f"  for (int i=0; i<{m*k}; ++i) a[i]=(float)((i*11)%17-8)*0.0625f;",
        f"  for (int i=0; i<{n*k}; ++i) w[i]=(float)((i*3)%19-9)*0.03125f;",
        f"  for (int i=0; i<{n}; ++i) bias[i]=(float)((i%7)-3)*0.125f;",
        f"  for (int i=0; i<{m*n}; ++i) out[i]=9999.0f;",
    ]
    c.body += [
        "    %zero = arith.constant 0.0 : f32\n"
        f"    linalg.fill ins(%zero : f32) outs({out[0]} : {out[1]})\n"
        "    linalg.matmul indexing_maps = ["
        "affine_map<(m, n, k) -> (m, k)>, "
        "affine_map<(m, n, k) -> (n, k)>, "
        "affine_map<(m, n, k) -> (m, n)>] "
        f"ins({a[0]}, {w[0]} : {a[1]}, {w[1]}) "
        f"outs({out[0]} : {out[1]})\n"
    ]
    c.body += [generic(
        [bias], [out],
        [amap(2, ["d1"]), amap(2)],
        ["parallel", "parallel"],
        "      %r = arith.addf %v0, %v1 : f32\n      linalg.yield %r : f32",
    )]
    c.verify += [
        f"  for (int i=0; i<{m}; ++i) for (int j=0; j<{n}; ++j) {{",
        "    oracle_float acc = bias[j];",
        f"    for (int t=0; t<{k}; ++t) acc += a[i*{k}+t] * w[j*{k}+t];",
        f"    compare(out[i*{n}+j], (float)acc, 5e-5f, 1e-4f, &errors, &maximum);",
        "  }",
    ]
    return c.emit(root)


def _padded_sample(x_ssa, x_ty, n_idx, c_idx, oh, ow, kh, kw):
    """Load x[n,c,ih,iw] with zero outside the padded window (index is unsigned)."""
    return f"""
        %ohs = arith.muli %{oh}, %cSH : index
        %ows = arith.muli %{ow}, %cSW : index
        %ihp = arith.addi %ohs, %{kh} : index
        %iwp = arith.addi %ows, %{kw} : index
        %ih_ok_lo = arith.cmpi uge, %ihp, %cPH : index
        %iw_ok_lo = arith.cmpi uge, %iwp, %cPW : index
        %ih = arith.subi %ihp, %cPH : index
        %iw = arith.subi %iwp, %cPW : index
        %ih_ok_hi = arith.cmpi ult, %ih, %cH : index
        %iw_ok_hi = arith.cmpi ult, %iw, %cW : index
        %ok_lo = arith.andi %ih_ok_lo, %iw_ok_lo : i1
        %ok_hi = arith.andi %ih_ok_hi, %iw_ok_hi : i1
        %ok = arith.andi %ok_lo, %ok_hi : i1
        %xv = scf.if %ok -> f32 {{
          %loaded = memref.load {x_ssa}[%{n_idx}, %{c_idx}, %ih, %iw] : {x_ty}
          scf.yield %loaded : f32
        }} else {{
          scf.yield %zero : f32
        }}
"""


def conv2d(root, h, w, c_in, oc, kh, kw, sh, sw, ph, pw, tag=""):
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    name = tag or f"conv2d_{h}x{w}_{c_in}x{oc}_k{kh}_s{sh}"
    c = Case(name, "conv2d_f32", "NCHW conv, FCHW weight, fused bias",
             extra={"conv": {"H": h, "W": w, "C": c_in, "OC": oc,
                             "KH": kh, "KW": kw, "SH": sh, "SW": sw,
                             "PH": ph, "PW": pw, "OH": oh, "OW": ow}})
    x = c.buffer("x", (1, c_in, h, w))
    weight = c.buffer("weight", (oc, c_in, kh, kw))
    bias = c.buffer("bias", (oc,))
    out = c.buffer("out", (1, oc, oh, ow))
    c.init += [
        f"  for (int i=0; i<{h*w*c_in}; ++i) x[i]=(float)((i*13)%21-10)*0.03125f;",
        f"  for (int i=0; i<{oc*c_in*kh*kw}; ++i) weight[i]=(float)((i*5)%15-7)*0.03125f;",
        f"  for (int i=0; i<{oc}; ++i) bias[i]=(float)((i%5)-2)*0.125f;",
        f"  for (int i=0; i<{oh*ow*oc}; ++i) out[i]=9999.0f;",
    ]
    sample = _padded_sample(x[0], x[1], "n", "ic", "oh", "ow", "kh", "kw")
    # Output-only generic (invertible maps). Reductions are scf.for so padding
    # can stay in memref.load rather than affine indexing maps.
    c.body += [
        "    %zero = arith.constant 0.0 : f32\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        f"    %cIC = arith.constant {c_in} : index\n"
        f"    %cKH = arith.constant {kh} : index\n"
        f"    %cKW = arith.constant {kw} : index\n"
        f"    %cSH = arith.constant {sh} : index\n"
        f"    %cSW = arith.constant {sw} : index\n"
        f"    %cPH = arith.constant {ph} : index\n"
        f"    %cPW = arith.constant {pw} : index\n"
        f"    %cH = arith.constant {h} : index\n"
        f"    %cW = arith.constant {w} : index\n"
    ]
    c.body += [generic(
        [], [out], [amap(4)], ["parallel"] * 4,
        "      %n = linalg.index 0 : index\n"
        "      %oc = linalg.index 1 : index\n"
        "      %oh = linalg.index 2 : index\n"
        "      %ow = linalg.index 3 : index\n"
        f"      %init = memref.load {bias[0]}[%oc] : {bias[1]}\n"
        "      %sum = scf.for %ic = %c0 to %cIC step %c1 iter_args(%acc_ic = %init) -> f32 {\n"
        "        %mid = scf.for %kh = %c0 to %cKH step %c1 iter_args(%acc_kh = %acc_ic) -> f32 {\n"
        "          %inner = scf.for %kw = %c0 to %cKW step %c1 iter_args(%acc_kw = %acc_kh) -> f32 {"
        + sample +
        f"            %wv = memref.load {weight[0]}[%oc, %ic, %kh, %kw] : {weight[1]}\n"
        "            %prod = arith.mulf %xv, %wv : f32\n"
        "            %next = arith.addf %acc_kw, %prod : f32\n"
        "            scf.yield %next : f32\n"
        "          }\n"
        "          scf.yield %inner : f32\n"
        "        }\n"
        "        scf.yield %mid : f32\n"
        "      }\n"
        "      linalg.yield %sum : f32",
    )]
    c.verify += [f"""  for (int oc=0; oc<{oc}; ++oc) for (int oh=0; oh<{oh}; ++oh) for (int ow=0; ow<{ow}; ++ow) {{
    oracle_float acc = bias[oc];
    for (int ic=0; ic<{c_in}; ++ic) for (int kh=0; kh<{kh}; ++kh) for (int kw=0; kw<{kw}; ++kw) {{
      int ih = oh*{sh} - {ph} + kh;
      int iw = ow*{sw} - {pw} + kw;
      if (ih>=0 && iw>=0 && ih<{h} && iw<{w})
        acc += x[(ic*{h}+ih)*{w}+iw] *
               weight[((oc*{c_in}+ic)*{kh}+kh)*{kw}+kw];
    }}
    compare(out[(oc*{oh}+oh)*{ow}+ow], (float)acc, 8e-5f, 2e-4f, &errors, &maximum);
  }}"""]
    return c.emit(root)


def dwconv2d(root, h, w, c_in, kh, kw, sh, sw, ph, pw, tag=""):
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    name = tag or f"dwconv2d_{h}x{w}x{c_in}_k{kh}_s{sh}"
    c = Case(name, "dwconv2d_f32", "NCHW depthwise, weight [C,1,KH,KW]",
             extra={"conv": {"H": h, "W": w, "C": c_in, "OC": c_in,
                             "KH": kh, "KW": kw, "SH": sh, "SW": sw,
                             "PH": ph, "PW": pw, "OH": oh, "OW": ow}})
    x = c.buffer("x", (1, c_in, h, w))
    weight = c.buffer("weight", (c_in, 1, kh, kw))
    bias = c.buffer("bias", (c_in,))
    out = c.buffer("out", (1, c_in, oh, ow))
    c.init += [
        f"  for (int i=0; i<{h*w*c_in}; ++i) x[i]=(float)((i*7)%23-11)*0.03125f;",
        f"  for (int i=0; i<{c_in*kh*kw}; ++i) weight[i]=(float)((i*3)%11-5)*0.0625f;",
        f"  for (int i=0; i<{c_in}; ++i) bias[i]=(float)((i%5)-2)*0.125f;",
        f"  for (int i=0; i<{oh*ow*c_in}; ++i) out[i]=9999.0f;",
    ]
    sample = _padded_sample(x[0], x[1], "n", "ch", "oh", "ow", "kh", "kw")
    c.body += [
        "    %zero = arith.constant 0.0 : f32\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        f"    %cKH = arith.constant {kh} : index\n"
        f"    %cKW = arith.constant {kw} : index\n"
        f"    %cSH = arith.constant {sh} : index\n"
        f"    %cSW = arith.constant {sw} : index\n"
        f"    %cPH = arith.constant {ph} : index\n"
        f"    %cPW = arith.constant {pw} : index\n"
        f"    %cH = arith.constant {h} : index\n"
        f"    %cW = arith.constant {w} : index\n"
    ]
    c.body += [generic(
        [], [out], [amap(4)], ["parallel"] * 4,
        "      %n = linalg.index 0 : index\n"
        "      %ch = linalg.index 1 : index\n"
        "      %oh = linalg.index 2 : index\n"
        "      %ow = linalg.index 3 : index\n"
        f"      %init = memref.load {bias[0]}[%ch] : {bias[1]}\n"
        "      %sum = scf.for %kh = %c0 to %cKH step %c1 iter_args(%acc_kh = %init) -> f32 {\n"
        "        %inner = scf.for %kw = %c0 to %cKW step %c1 iter_args(%acc_kw = %acc_kh) -> f32 {"
        + sample +
        f"          %wv = memref.load {weight[0]}[%ch, %c0, %kh, %kw] : {weight[1]}\n"
        "          %prod = arith.mulf %xv, %wv : f32\n"
        "          %next = arith.addf %acc_kw, %prod : f32\n"
        "          scf.yield %next : f32\n"
        "        }\n"
        "        scf.yield %inner : f32\n"
        "      }\n"
        "      linalg.yield %sum : f32",
    )]
    c.verify += [f"""  for (int ch=0; ch<{c_in}; ++ch) for (int oh=0; oh<{oh}; ++oh) for (int ow=0; ow<{ow}; ++ow) {{
    oracle_float acc = bias[ch];
    for (int kh=0; kh<{kh}; ++kh) for (int kw=0; kw<{kw}; ++kw) {{
      int ih = oh*{sh} - {ph} + kh;
      int iw = ow*{sw} - {pw} + kw;
      if (ih>=0 && iw>=0 && ih<{h} && iw<{w})
        acc += x[(ch*{h}+ih)*{w}+iw] * weight[(ch*{kh}+kh)*{kw}+kw];
    }}
    compare(out[(ch*{oh}+oh)*{ow}+ow], (float)acc, 8e-5f, 2e-4f, &errors, &maximum);
  }}"""]
    return c.emit(root)


SMOKE = [
    "silu_1x256", "sigmoid_1x32", "add_1x256", "mul_1x32",
    "gap_9x9x32", "se_scale_9x9x32", "linear_1x16x32",
    "conv2d_stem_tiny", "conv2d_1x1_9x9_32x16",
    "dwconv2d_9x9x32_k3", "dwconv2d_10x10x16_k5",
]


def catalog(root: Path, include_large=False):
    root = Path(root)
    metas = [
        pointwise(root, "silu", (1, 256)),
        pointwise(root, "silu", (1, 1536)),
        pointwise(root, "silu", (1, 384, 9, 9)),
        pointwise(root, "sigmoid", (1, 32)),
        pointwise(root, "sigmoid", (1, 384)),
        pointwise(root, "add", (1, 256)),
        pointwise(root, "add", (1, 384, 9, 9)),
        pointwise(root, "mul", (1, 32)),
        gap(root, 9, 9, 32),
        gap(root, 9, 9, 384),
        gap(root, 9, 9, 1536),
        se_scale(root, 9, 9, 32),
        se_scale(root, 9, 9, 384),
        linear(root, 1, 16, 32),
        linear(root, 1, 16, 384),
        linear(root, 1, 384, 16),
        linear(root, 1, 1000, 1536),
        conv2d(root, 16, 16, 3, 8, 3, 3, 2, 2, 1, 1, "conv2d_stem_tiny"),
        conv2d(root, 9, 9, 32, 16, 1, 1, 1, 1, 0, 0, "conv2d_1x1_9x9_32x16"),
        conv2d(root, 9, 9, 384, 1536, 1, 1, 1, 1, 0, 0, "conv2d_head_1x1"),
        conv2d(root, 18, 18, 96, 576, 1, 1, 1, 1, 0, 0),
        conv2d(root, 18, 18, 576, 96, 1, 1, 1, 1, 0, 0),
        dwconv2d(root, 9, 9, 32, 3, 3, 1, 1, 1, 1, "dwconv2d_9x9x32_k3"),
        dwconv2d(root, 10, 10, 16, 5, 5, 1, 1, 2, 2, "dwconv2d_10x10x16_k5"),
        dwconv2d(root, 18, 18, 576, 3, 3, 1, 1, 1, 1),
    ]
    if include_large:
        metas.append(conv2d(root, 288, 288, 3, 40, 3, 3, 2, 2, 1, 1, "conv2d_stem_288"))
    index = {
        "count": len(metas),
        "smoke": SMOKE,
        "cases": [m["name"] for m in metas],
    }
    (root / "case-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(metas)} cases under {root}")
    return metas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-large", action="store_true")
    args = parser.parse_args()
    catalog(args.root, include_large=args.include_large)
