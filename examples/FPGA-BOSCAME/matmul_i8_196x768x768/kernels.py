"""Triton kernel for the standalone signed INT8 AME matmul example.

``matmul_i8`` is the kernel used by the ViT-B/16 W8A8 path
(``examples/FPGA-BOSCAME/vit-base-patch16-224/triton/kernels.py``).  It is
reproduced here unchanged so this example is self-contained: the operational
shape ``[196, 768] @ [768, 768]^T`` is supplied by :mod:`cases` as
``tl.constexpr`` values, and the very same source is lowered both to host LLVM
and to BOSCAME NR by the build pipeline.
"""
import triton
import triton.language as tl


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
