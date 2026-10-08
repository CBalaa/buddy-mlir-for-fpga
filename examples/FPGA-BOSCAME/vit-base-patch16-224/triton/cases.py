"""Static ViT-B/16 Triton kernel specializations.

The metadata in each sibling case directory is intentionally small and stable;
it describes the operation shape used by the model graph replacement pass.  No
PyTorch or Triton import is needed to enumerate cases.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent


def cdiv(value, divisor):
    return (value + divisor - 1) // divisor


def power_of_two(value):
    return 1 << (value - 1).bit_length()


def _case(name, kind, arguments, constexprs, grid, shape, roles=(), kernel=None):
    pointer_type = {"f32": "*fp32", "i32": "*i32", "i8": "*i8"}
    return {
        "name": name,
        "family": kind,
        "kernel": kernel or kind,
        "kernel_module": "kernels",
        "symbol": "triton_" + name,
        "arguments": arguments,
        "signature": {arg["name"]: pointer_type[arg["dtype"]] for arg in arguments},
        "constexprs": constexprs,
        "grid": list(grid),
        "directory": str(PARENT / name),
        "launch_source": str(PARENT / name / "launch.c"),
        "shape": list(shape),
        "roles": list(roles),
        "validation": "Independent launch oracle; FP32 contiguous tensors",
    }


def _arg(name, rank, dtype="f32"):
    return {"name": name, "rank": rank, "dtype": dtype}


def describe(name):
    path = PARENT / name / "metadata.json"
    if not path.is_file():
        raise ValueError("unknown case: " + name)
    metadata = json.loads(path.read_text())
    kind = metadata["kind"]
    shape = metadata["shape"]
    if kind == "matmul_i8":
        m, n, k = shape
        # AME tiles have a fixed 16-row M dimension on patch6.  Keep the
        # public tensor shape logical (for the graph ABI), but compile the
        # standalone kernel against a zero-padded physical M dimension.  The
        # adapter supplies that padded buffer and copies only the logical rows
        # back to the caller.  This avoids issuing a short msettilem on the
        # final tile (for M=196/197 and M=1), which is not supported reliably
        # by the hardware.
        padded_m = cdiv(m, 16) * 16
        # The NR AME lowering supports board-validated N tiles of 16, 32 and
        # 64.  Keep 16 as the conservative default, while allowing the model
        # build to select a wider tile once its standalone numerical case has
        # passed.  The value is part of the exported case manifest, so a
        # kernel and its lowering pass cannot silently disagree.
        tile_n = int(os.environ.get("VIT_TRITON_AME_N", "16"))
        if tile_n not in (16, 32, 64):
            raise ValueError("VIT_TRITON_AME_N must be 16, 32, or 64")
        return _case(
            name, kind,
            [_arg("A", 2, "i8"), _arg("B", 2, "i8"), _arg("C", 2, "i32")],
            dict(M=padded_m, VALID_M=m, N=n, K=k,
                 BM=power_of_two(min(padded_m, 16)), BN=tile_n, BK=k & -k),
            (cdiv(padded_m, power_of_two(min(padded_m, 16))), cdiv(n, tile_n), 1), shape,
            roles=metadata.get("roles", ()), kernel="matmul_i8")
    if kind == "per_token_quantization":
        rows, width = shape
        quant_mode = os.environ.get("VIT_TRITON_QUANT", "baseline")
        if quant_mode not in ("baseline", "rvv"):
            raise ValueError("VIT_TRITON_QUANT must be baseline or rvv")
        return _case(
            name, kind,
            [_arg("X", 2), _arg("Q", 2, "i8"), _arg("Scale", 1)],
            dict(WIDTH=width, BLOCK=power_of_two(width)), (rows, 1, 1), shape,
            kernel="quantize") | {"quantization_lowering": quant_mode}
    if kind == "dequantization":
        rows, width = shape
        dequant_mode = os.environ.get("VIT_TRITON_DEQUANT", "baseline")
        if dequant_mode not in ("baseline", "rvv"):
            raise ValueError("VIT_TRITON_DEQUANT must be baseline or rvv")
        case = _case(
            name, kind,
            [_arg("X", 2, "i32"), _arg("Row", 1), _arg("Column", 1),
             _arg("Out", 2)],
            dict(ROWS=rows, COLS=width, BLOCK=128),
            (cdiv(rows * width, 128), 1, 1), shape, kernel="dequantize")
        if dequant_mode == "rvv":
            case["kernel"] = "dequantize_rows"
            case["kernel_module"] = "kernels_dequant"
            case["grid"] = [cdiv(width, 128), rows, 1]
        return case
    if kind == "linear":
        m, n, k = shape
        return _case(name, kind, [_arg("A", 2), _arg("B", 2), _arg("C", 2)],
                     dict(M=m, N=n, K=k, BM=power_of_two(min(m, 16)), BN=16,
                          BK=power_of_two(min(k, 64))),
                     (cdiv(m, power_of_two(min(m, 16))), cdiv(n, 16), 1), shape)
    if kind == "patch_embed":
        # The external ABI remains NCHW/OIHW; the composed adapter uses this
        # case to materialize a row-major [196,768] im2col matrix before the
        # dedicated RVV linear case.
        return _case(name, kind,
                     [_arg("X", 4), _arg("Weight", 4), _arg("Bias", 1), _arg("Out", 4)],
                     dict(HEIGHT=224, WIDTH=224, CHANNELS=3, PATCH=16,
                         HIDDEN=768, GRID=14, BLOCK=1024), (196, 1, 1), shape)
    if kind == "layernorm":
        rows, width = shape
        return _case(name, kind,
                     [_arg("X", 2), _arg("Weight", 1), _arg("Bias", 1), _arg("Out", 2)],
                     dict(ROWS=rows, WIDTH=width, BLOCK=power_of_two(width), EPS=1.0e-12),
                     (rows, 1, 1), shape)
    if kind == "gelu":
        rows, width = shape
        count = rows * width
        return _case(name, kind, [_arg("X", 2), _arg("Out", 2)],
                     dict(COUNT=count, BLOCK=128), (cdiv(count, 128), 1, 1), shape)
    if kind == "bias_add":
        rows, width = shape
        block = 1024
        return _case(name, kind, [_arg("X", 2), _arg("Bias", 1), _arg("Out", 2)],
                     dict(ROWS=rows, WIDTH=width, BLOCK=block),
                     (rows, cdiv(width, block), 1), shape)
    if kind == "add":
        rows, width = shape
        count = rows * width
        return _case(name, kind, [_arg("X", 2), _arg("Y", 2), _arg("Out", 2)],
                     dict(COUNT=count, BLOCK=128), (cdiv(count, 128), 1, 1), shape)
    if kind in ("attention_qk", "attention_pv"):
        heads, m, n, k = shape
        return _case(name, kind,
                     [_arg("A", 3), _arg("B", 3), _arg("C", 3)],
                     dict(HEADS=heads, M=m, N=n, K=k, BM=16, BN=16,
                          BK=power_of_two(min(k, 64)),
                          # The replacement matcher folds the explicit
                          # 1/sqrt(64) MulOp into the QK call.  The standalone
                          # case therefore owns the scale exactly once.
                          SCALE=(1.0 / (k ** 0.5) if kind == "attention_qk" else 1.0)),
                     (cdiv(m, 16), cdiv(n, 16), heads), shape, kernel="attention_dot")
    if kind == "softmax":
        heads, rows, length = shape
        return _case(name, kind, [_arg("X", 3), _arg("Out", 3)],
                     dict(ROWS=heads * rows, LENGTH=length, BLOCK=power_of_two(length)),
                     (heads * rows, 1, 1), shape)
    raise ValueError(f"unsupported ViT metadata kind: {kind}")


def inventory():
    return [describe(path.parent.name)
            for path in sorted(PARENT.glob("*/metadata.json"))]


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
