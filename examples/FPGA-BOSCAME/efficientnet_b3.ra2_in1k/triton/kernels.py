# ===- kernels.py - EfficientNet Triton JIT bodies ---------------*- python -*-===//
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
# These @triton.jit functions are the only operator implementations. The host
# specialises shapes and launches a grid; every load, MAC and store lives
# here. Layout is NCHW / FCHW to match Buddy Conv2dOp. No handwritten
# linalg kernel.mlir is synthesised for these cases.
#
# conv2d_nchw broadcasts the input-channel index with ic[None, :] because
# tt.expand_dims needs an explicit axis. gap divides by (SPATIAL * 1.0)
# so the constexpr stays float32 (SPATIAL.to(tl.float32) is not legal).
#
# ===----------------------------------------------------------------------===//
"""Real Triton kernels for EfficientNet-B3 operator examples."""
import triton
import triton.language as tl


@triton.jit
def linear(A, W, Bias, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """FP32 linear with bias (classifier / SE)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k = tl.arange(0, BK)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(W + cols[None, :] * K + kk[:, None],
                    (cols[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc = acc + tl.load(Bias + cols, cols < N, other=0.0)[None, :]
    tl.store(Out + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           INTEGER: tl.constexpr):
    """AME W8A8 / FP32 matmul without bias. C is load-accumulate (matches Qwen)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    if INTEGER:
        accumulator = tl.full((BM, BN), 0, tl.int32)
    else:
        accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        if M % BM == 0 and K % BK == 0:
            a = tl.load(A + rows[:, None] * K + kk[None, :])
        else:
            a = tl.load(A + rows[:, None] * K + kk[None, :],
                        (rows[:, None] < M) & (kk[None, :] < K), other=0)
        if N % BN == 0 and K % BK == 0:
            b = tl.load(B + cols[None, :] * K + kk[:, None])
        else:
            b = tl.load(B + cols[None, :] * K + kk[:, None],
                        (cols[None, :] < N) & (kk[:, None] < K), other=0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    address = C + rows[:, None] * N + cols[None, :]
    if M % BM == 0 and N % BN == 0:
        previous = tl.load(address)
        tl.store(address, accumulator + previous)
    else:
        previous = tl.load(address, valid, other=0)
        tl.store(address, accumulator + previous, valid)


@triton.jit
def quantize(X, Q, Scale, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + channel, channel < WIDTH, other=0)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    value = tl.div_rn(x, scale)
    rounded = value + tl.where(value >= 0.0, 0.5, -0.5)
    saturated = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    integer = saturated.to(tl.int32).to(tl.int8)
    tl.store(Scale + row, scale)
    tl.store(Q + row * WIDTH + channel, integer, channel < WIDTH)


@triton.jit
def dequantize(X, Row, Column, Out, ROWS: tl.constexpr, COLS: tl.constexpr,
               BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ROWS * COLS
    value = tl.load(X + index, valid, other=0).to(tl.float32)
    row = tl.load(Row + index // COLS, valid, other=0)
    column = tl.load(Column + index % COLS, valid, other=0)
    tl.store(Out + index, (value * row) * column, valid)


@triton.jit
def conv2d_nchw(X, Weight, Bias, Out,
                H: tl.constexpr, W: tl.constexpr, C: tl.constexpr, OC: tl.constexpr,
                KH: tl.constexpr, KW: tl.constexpr,
                SH: tl.constexpr, SW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
                OH: tl.constexpr, OW: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    spatial = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    kdim: tl.constexpr = KH * KW * C
    k = tl.arange(0, BK)
    oh = spatial // OW
    ow = spatial % OW
    for block in range(tl.cdiv(kdim, BK)):
        kk = block * BK + k
        kw = kk % KW
        tmp = kk // KW
        kh = tmp % KH
        ic = tmp // KH
        ih = oh[:, None] * SH - PH + kh[None, :]
        iw = ow[:, None] * SW - PW + kw[None, :]
        ic2 = ic[None, :]
        inside = (spatial[:, None] < OH * OW) & (kk[None, :] < kdim) & \
                 (ih >= 0) & (iw >= 0) & (ih < H) & (iw < W)
        a = tl.load(X + ic2 * (H * W) + ih * W + iw, inside, other=0.0)
        b = tl.load(Weight + cols[None, :] * kdim + kk[:, None],
                    (cols[None, :] < OC) & (kk[:, None] < kdim), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    acc = acc + tl.load(Bias + cols, cols < OC, other=0.0)[None, :]
    tl.store(Out + cols[None, :] * (OH * OW) + spatial[:, None], acc,
             (spatial[:, None] < OH * OW) & (cols[None, :] < OC))


@triton.jit
def dwconv2d_nchw(X, Weight, Bias, Out,
                  H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
                  KH: tl.constexpr, KW: tl.constexpr,
                  SH: tl.constexpr, SW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
                  OH: tl.constexpr, OW: tl.constexpr, BLOCK: tl.constexpr):
    pix = tl.program_id(0)
    ch0 = tl.program_id(1) * BLOCK
    ch = ch0 + tl.arange(0, BLOCK)
    oh = pix // OW
    ow = pix % OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH - PH + kh
            iw = ow * SW - PW + kw
            inside = (pix < OH * OW) & (ch < C) & (ih >= 0) & (iw >= 0) & (ih < H) & (iw < W)
            x = tl.load(X + ch * (H * W) + ih * W + iw, inside, other=0.0)
            wt = tl.load(Weight + ch * (KH * KW) + kh * KW + kw, ch < C, other=0.0)
            acc += x * wt
    acc = acc + tl.load(Bias + ch, ch < C, other=0.0)
    tl.store(Out + ch * (OH * OW) + pix, acc, (pix < OH * OW) & (ch < C))


@triton.jit
def binary(X, Y, Out, COUNT: tl.constexpr, MULTIPLY: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < COUNT, other=0.0)
    y = tl.load(Y + index, index < COUNT, other=0.0)
    if MULTIPLY:
        result = x * y
    else:
        result = x + y
    tl.store(Out + index, result, index < COUNT)


@triton.jit
def silu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < COUNT, other=0.0)
    tl.store(Out + index, tl.div_rn(x, 1.0 + tl.exp(-x)), index < COUNT)


@triton.jit
def sigmoid(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < COUNT, other=0.0)
    tl.store(Out + index, tl.div_rn(1.0, 1.0 + tl.exp(-x)), index < COUNT)


@triton.jit
def gap(X, Out, SPATIAL: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    ch = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p in range(SPATIAL):
        acc += tl.load(X + ch * SPATIAL + p, ch < C, other=0.0)
    tl.store(Out + ch, tl.div_rn(acc, SPATIAL * 1.0), ch < C)


@triton.jit
def se_scale(X, Scale, Out, SPATIAL: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    pix = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + ch * SPATIAL + pix, (pix < SPATIAL) & (ch < C), other=0.0)
    s = tl.load(Scale + ch, ch < C, other=0.0)
    tl.store(Out + ch * SPATIAL + pix, x * s, (pix < SPATIAL) & (ch < C))
