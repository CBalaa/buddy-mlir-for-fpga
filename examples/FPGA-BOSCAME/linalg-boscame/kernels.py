"""Real Triton sources for the canonical standalone backend examples."""

import triton
import triton.language as tl

M = tl.constexpr(128)
N = tl.constexpr(128)
K = tl.constexpr(1024)
BM = tl.constexpr(64)
BN = tl.constexpr(128)
BK = tl.constexpr(1024)
ROWS = tl.constexpr(197)
WIDTH = tl.constexpr(768)
BLOCK = tl.constexpr(1024)


@triton.jit
def gemm(A, B, C):
    """Grid 2x1; full roots, physical B[N,K], nonzero original C added once."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    depth = tl.arange(0, BK)
    a = tl.load(A + rows[:, None] * K + depth[None, :])
    b = tl.load(B + columns[None, :] * K + depth[:, None])
    zero = tl.full((BM, BN), 0, tl.int32)
    product = tl.dot(a, b, zero, input_precision="ieee")
    address = C + rows[:, None] * N + columns[None, :]
    previous = tl.load(address)
    tl.store(address, product + previous)


@triton.jit
def layernorm(X, Weight, Bias, Out):
    """Grid 197x1; preserve all 1024 logical lanes and the 768 valid prefix."""
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    weight = tl.load(Weight + channel, valid, other=1.0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WIDTH
    centered = tl.where(valid, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    out = centered * tl.rsqrt(variance + 1.0e-12) * weight + bias
    tl.store(Out + row * WIDTH + channel, out, valid)
