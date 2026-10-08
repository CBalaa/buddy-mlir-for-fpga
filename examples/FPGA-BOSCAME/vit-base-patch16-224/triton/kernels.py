"""Triton kernels used by the ViT-B/16 FPGA example.

The kernels intentionally use contiguous, flattened tensors.  Shape constants
are supplied by :mod:`cases` so the same source is lowered to host LLVM and to
BOSCAME NR without a model specific scalar implementation.
"""
import triton
import triton.language as tl


@triton.jit
def linear(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[M,N] += A[M,K] @ B[N,K]^T, where B is stored as [N,K]."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + cols[None, :] * K + kk[:, None],
                    (cols[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    # Keeping += semantics allows a graph to initialize C with a bias or a
    # residual.  The normal linear cases initialize C to zero in their launch.
    previous = tl.load(C + rows[:, None] * N + cols[None, :], valid, other=0.0)
    tl.store(C + rows[:, None] * N + cols[None, :], acc + previous, valid)


@triton.jit
def matmul_i8(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              VALID_M: tl.constexpr):
    """Signed INT8 matrix multiply for the patch6 AME path.

    ``A`` is row-major physical ``[M,K]`` and ``B`` is the physical model
    layout ``[N,K]``.  ``VALID_M`` is the logical row count; rows in
    ``VALID_M..M`` are zero-padded by the adapter.  The result is an
    accumulating ``int32`` ``[M,N]`` buffer;
    callers clear it before launch because AME's ``mqma.b.mm`` has +=
    semantics.  Keeping this as a separate Triton entry point prevents an
    accidental FP32 dot from reaching the integer lowering pass.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        valid_a = (rows[:, None] < VALID_M) & (kk[None, :] < K)
        valid_b = (cols[None, :] < N) & (kk[:, None] < K)
        a = tl.load(A + rows[:, None] * K + kk[None, :], valid_a, other=0)
        b = tl.load(B + cols[None, :] * K + kk[:, None], valid_b, other=0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    valid = (rows[:, None] < VALID_M) & (cols[None, :] < N)
    address = C + rows[:, None] * N + cols[None, :]
    previous = tl.load(address, valid, other=0)
    tl.store(address, acc + previous, valid)


@triton.jit
def quantize(X, Q, Scale, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """Per-row symmetric signed INT8 quantization, ties away from zero."""
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    value = tl.div_rn(x, scale)
    rounded = value + tl.where(value >= 0.0, 0.5, -0.5)
    saturated = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    integer = saturated.to(tl.int32).to(tl.int8)
    tl.store(Scale + row, scale)
    tl.store(Q + row * WIDTH + channel, integer, valid)


@triton.jit
def dequantize(X, Row, Column, Out, ROWS: tl.constexpr, COLS: tl.constexpr,
               BLOCK: tl.constexpr):
    """Convert AME INT32 accumulators back to FP32 using row/column scales."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ROWS * COLS
    value = tl.load(X + index, valid, other=0).to(tl.float32)
    row = tl.load(Row + index // COLS, valid, other=1.0)
    column = tl.load(Column + index % COLS, valid, other=1.0)
    tl.store(Out + index, (value * row) * column, valid)


@triton.jit
def patch_embed(X, Weight, Bias, Out, HEIGHT: tl.constexpr,
                WIDTH: tl.constexpr, CHANNELS: tl.constexpr,
                PATCH: tl.constexpr, HIDDEN: tl.constexpr,
                GRID: tl.constexpr, BLOCK: tl.constexpr):
    """Materialize the convolution patches as a row-major im2col matrix.

    The projection is a ``[196, 768] @ [768, 768]^T`` operation.  The old
    kernel performed that reduction while loading each output channel, which
    made the weight and result accesses strided by the opposite matrix
    dimension.  This stage only walks the input image and writes one
    contiguous ``[patch, channel*PATCH*PATCH + dy*PATCH + dx]`` row.  The
    graph-facing C adapter then feeds the matrix to the existing RVV linear
    kernel and transposes its result back to the NCHW contract.

    ``Weight`` and ``Bias`` remain in the signature for ABI compatibility with
    the graph replacement.  They are intentionally unused here.
    """
    patch = tl.program_id(0)
    row = patch // GRID
    col = patch % GRID
    for channel in range(CHANNELS):
        for dy in range(PATCH):
            for dx in range(PATCH):
                input_index = channel * HEIGHT * WIDTH + (row * PATCH + dy) * WIDTH + col * PATCH + dx
                column = channel * PATCH * PATCH + dy * PATCH + dx
                # Every row is contiguous in the destination even though the
                # source is traversed through the NCHW image layout.
                tl.store(Out + patch * HIDDEN + column,
                         tl.load(X + input_index))


@triton.jit
def layernorm(X, Weight, Bias, Out, ROWS: tl.constexpr,
              WIDTH: tl.constexpr, BLOCK: tl.constexpr,
              EPS: tl.constexpr):
    """LayerNorm over the final (hidden) dimension, with affine parameters."""
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    x = tl.load(X + row * WIDTH + channel, valid, other=0.0).to(tl.float32)
    weight = tl.load(Weight + channel, valid, other=1.0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WIDTH
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    out = centered * tl.rsqrt(variance + EPS) * weight + bias
    tl.store(Out + row * WIDTH + channel, out, valid)


@triton.jit
def gelu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """Exact Hugging Face GELU (the ViT config uses approximate="none")."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0).to(tl.float32)
    # tl.erf is lowered by triton-riscv to the target math implementation.
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
    tl.store(Out + index, y, valid)


@triton.jit
def add(X, Y, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    tl.store(Out + index,
             tl.load(X + index, valid, other=0.0) +
             tl.load(Y + index, valid, other=0.0), valid)


@triton.jit
def bias_add(X, Bias, Out, ROWS: tl.constexpr, WIDTH: tl.constexpr,
             BLOCK: tl.constexpr):
    """Add a broadcast row bias without vector remainder/modulo operations."""
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < ROWS) & (channel < WIDTH)
    index = row * WIDTH + channel
    x = tl.load(X + index, valid, other=0.0)
    bias = tl.load(Bias + channel, channel < WIDTH, other=0.0)
    tl.store(Out + index, x + bias, valid)


@triton.jit
def attention_dot(A, B, C, HEADS: tl.constexpr, M: tl.constexpr,
                  N: tl.constexpr, K: tl.constexpr,
                  SCALE: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Batched head matmul for QK^T and attention-probability V."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(A + head * M * K + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + head * K * N + kk[:, None] * N + cols[None, :],
                    (kk[:, None] < K) & (cols[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(C + head * M * N + rows[:, None] * N + cols[None, :], acc * SCALE, valid)


@triton.jit
def softmax(X, Out, ROWS: tl.constexpr, LENGTH: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < LENGTH
    x = tl.load(X + row * LENGTH + column, valid, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    exponential = tl.exp(x - maximum)
    total = tl.sum(exponential, axis=0)
    tl.store(Out + row * LENGTH + column, exponential / total, valid)
