"""Row-contiguous dequantization specialization for RVV lowering."""
import triton
import triton.language as tl


@triton.jit
def dequantize_rows(X, Row, Column, Out, ROWS: tl.constexpr,
                    COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.load(Row + row)
    valid = column < COLS
    value = tl.load(X + row * COLS + column, valid, other=0).to(tl.float32)
    weight_scale = tl.load(Column + column, valid, other=0.0)
    tl.store(Out + row * COLS + column, (value * scale) * weight_scale, valid)
