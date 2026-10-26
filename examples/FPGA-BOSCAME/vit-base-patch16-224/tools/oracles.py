"""Independent NumPy references for the ported ViT operator semantics."""

import math

import numpy as np

DTYPES = {"f32": np.dtype("<f4"), "i32": np.dtype("<i4"), "i8": np.dtype("i1")}


def output_names(case):
    """Writable entry arguments; a matmul's C is also an input."""
    family = case["family"]
    if family in (
        "linear",
        "matmul_i8",
        "matmul_i8_overwrite",
    ) or family.startswith("attention_"):
        return ("C",)
    if family == "per_token_quantization":
        return ("Q", "Scale")
    return ("Out",)


def make_inputs(case, shapes, stress=False):
    """Deterministic inputs, including zero rows and signed integer extremes."""
    rng = np.random.default_rng(107 if stress else 71)
    outputs = output_names(case)
    arrays = {}
    for argument, shape in zip(case["arguments"], shapes):
        name, dtype = argument["name"], DTYPES[argument["dtype"]]
        if dtype == DTYPES["i8"]:
            if stress:
                choices = np.array(
                    [-128, -127, -1, 0, 1, 126, 127], dtype=dtype
                )
                value = rng.choice(choices, size=shape)
            else:
                value = rng.integers(-15, 16, size=shape, dtype=dtype)
        elif dtype == DTYPES["i32"]:
            if stress:
                choices = np.array(
                    [-2147483648, -2147483600, -1, 1, 2147483600, 2147483647],
                    dtype=dtype,
                )
                value = rng.choice(choices, size=shape)
            else:
                value = rng.integers(-100, 101, size=shape, dtype=dtype)
        elif name in ("Row", "Column", "Scale"):
            value = rng.uniform(0.002, 0.02, size=shape).astype(dtype)
        elif name == "Weight":
            value = rng.uniform(-1, 2, size=shape).astype(dtype)
        elif name == "Bias":
            value = rng.uniform(-0.5, 0.5, size=shape).astype(dtype)
        else:
            value = rng.normal(0.2, 1.1, size=shape).astype(dtype)
        if name in outputs and name != "C":
            # A visible sentinel proves every required element was written.
            value = np.full(
                shape, 0x55 if dtype.kind == "i" else np.nan, dtype=dtype
            )
        arrays[name] = np.ascontiguousarray(value, dtype=dtype)
    if case["family"] == "per_token_quantization":
        arrays["X"][0] = 0.0
        if arrays["X"].shape[0] > 1:
            width = arrays["X"].shape[-1]
            ties = np.resize(
                np.array(
                    [-127, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 127],
                    dtype=np.float32,
                ),
                width,
            )
            arrays["X"][1] = ties
    if stress and case["family"] == "attention_pv":
        # Distinguish block-local accumulation from reassociation across K
        # blocks. The unit term is rounded away inside the second block.
        a, b = arrays["A"], arrays["B"]
        block = case["constexprs"]["BK"]
        a.fill(0)
        signs = np.where(
            (np.indices(a.shape[:-1]).sum(axis=0) % 2) == 0, 1, -1
        ).astype(np.float32)
        if a.shape[-1] > block + 1:
            a[..., 0] = signs * np.float32(2**25)
            a[..., block] = -signs * np.float32(2**25)
            a[..., block + 1] = signs
        if a.shape[-1] > 2 * block:
            a[..., 2 * block] = signs * np.float32(0.125)
        tail = a.shape[-1] % block
        if tail:
            a[..., -tail:] = signs[..., None] * np.resize(
                np.array([2, -4, 0.5, 1.25, -0.25], dtype=np.float32), tail
            )
        columns = np.arange(b.shape[-1])
        scales = np.ldexp(
            np.where(columns % 2, np.float32(-1), np.float32(1)),
            columns % 5 - 2,
        )
        b[...] = scales
        arrays["C"].fill(np.nan)
    return arrays


def _matmul_sequential(a, b):
    """One f32 multiply and one f32 add per K, with no reassociation/FMA."""
    result = np.zeros(a.shape[:-1] + (b.shape[-1],), dtype=np.float32)
    for k in range(a.shape[-1]):
        product = np.multiply(
            a[..., :, k, None], b[..., k, None, :], dtype=np.float32
        )
        np.add(result, product, out=result)
    return result


def _matmul_blocked_sequential(a, b, block):
    """Original floating dot expansion: zero product, then add per K block.

    Retain the entire declared block, including positive-zero padding in the
    last one. Each multiply and addition rounds independently to f32; neither
    the block reductions nor the additions between blocks are reassociated.
    """
    if block <= 0:
        raise ValueError("matmul reduction block must be positive")
    result = np.zeros(a.shape[:-1] + (b.shape[-1],), dtype=np.float32)
    for start in range(0, a.shape[-1], block):
        valid = min(block, a.shape[-1] - start)
        left = np.zeros(a.shape[:-1] + (block,), dtype=np.float32)
        right = np.zeros(b.shape[:-2] + (block, b.shape[-1]), dtype=np.float32)
        left[..., :valid] = a[..., start : start + valid]
        right[..., :valid, :] = b[..., start : start + valid, :]
        product = _matmul_sequential(left, right)
        np.add(result, product, out=result)
    return result


def reference(case, arrays):
    """Return references only for the writable arguments of this case."""
    family = case["family"]
    if family in ("matmul_i8", "matmul_i8_overwrite"):
        a, b, c = (arrays[name] for name in ("A", "B", "C"))
        return {
            "C": (
                (c.astype(np.int64) if family == "matmul_i8" else 0)
                + a.astype(np.int64) @ b.astype(np.int64).T
            ).astype("<i4")
        }
    if family == "linear":
        product = _matmul_sequential(arrays["A"], arrays["B"].T)
        return {"C": np.add(product, arrays["C"], dtype=np.float32)}
    if family.startswith("attention_"):
        product = (
            _matmul_blocked_sequential(
                arrays["A"], arrays["B"], case["constexprs"]["BK"]
            )
            if family == "attention_pv"
            else _matmul_sequential(arrays["A"], arrays["B"])
        )
        return {
            "C": np.multiply(
                product,
                np.float32(case["constexprs"]["SCALE"]),
                dtype=np.float32,
            )
        }
    if family == "add":
        return {"Out": np.add(arrays["X"], arrays["Y"], dtype=np.float32)}
    if family == "bias_add":
        return {"Out": np.add(arrays["X"], arrays["Bias"], dtype=np.float32)}
    if family == "layernorm":
        rows, width = case["shape"]
        block = case["constexprs"]["BLOCK"]
        padded = np.pad(arrays["X"], ((0, 0), (0, block - width)))
        mean = np.divide(
            np.cumsum(padded, axis=1, dtype=np.float32)[:, -1],
            np.float32(width),
            dtype=np.float32,
        )
        centered = np.subtract(padded, mean[:, None], dtype=np.float32)
        centered[:, width:] = np.float32(0)
        squared = np.multiply(centered, centered, dtype=np.float32)
        variance = np.divide(
            np.cumsum(squared, axis=1, dtype=np.float32)[:, -1],
            np.float32(width),
            dtype=np.float32,
        )
        shifted = np.add(
            variance, np.float32(case["constexprs"]["EPS"]), dtype=np.float32
        )
        inverse = np.divide(
            np.float32(1), np.sqrt(shifted, dtype=np.float32), dtype=np.float32
        )
        normalized = np.multiply(
            centered[:, :width], inverse[:, None], dtype=np.float32
        )
        scaled = np.multiply(normalized, arrays["Weight"], dtype=np.float32)
        return {"Out": np.add(scaled, arrays["Bias"], dtype=np.float32)}
    if family == "per_token_quantization":
        x = arrays["X"]
        maximum = np.max(np.abs(x), axis=-1)
        scale = np.where(
            maximum == 0,
            np.float32(1),
            np.divide(maximum, np.float32(127), dtype=np.float32),
        ).astype(np.float32)
        value = np.divide(x, scale[:, None], dtype=np.float32)
        saturated = np.clip(value, np.float32(-127), np.float32(127))
        base = saturated.astype(np.int32)
        fraction = np.subtract(
            saturated, base.astype(np.float32), dtype=np.float32
        )
        adjustment = np.where(
            np.abs(fraction) >= np.float32(0.5),
            np.where(saturated >= 0, 1, -1),
            0,
        )
        q = (base + adjustment).astype(np.int8)
        return {"Q": q, "Scale": scale}
    if family == "dequantization":
        first = np.multiply(
            arrays["X"].astype(np.float32),
            arrays["Row"][:, None],
            dtype=np.float32,
        )
        return {
            "Out": np.multiply(
                first, arrays["Column"][None, :], dtype=np.float32
            )
        }
    if family == "gelu":
        x = arrays["X"]
        erf_input = np.multiply(
            x, np.float32(0.7071067811865475), dtype=np.float32
        )
        erf = np.fromiter(
            (math.erf(float(v)) for v in erf_input.flat),
            dtype=np.float32,
            count=x.size,
        ).reshape(x.shape)
        factor = np.add(np.float32(1), erf, dtype=np.float32)
        half_x = np.multiply(np.float32(0.5), x, dtype=np.float32)
        return {"Out": np.multiply(half_x, factor, dtype=np.float32)}
    if family == "softmax":
        length, block = (
            case["constexprs"]["LENGTH"],
            case["constexprs"]["BLOCK"],
        )
        x = arrays["X"].reshape(-1, length)
        padded = np.pad(
            x, ((0, 0), (0, block - length)), constant_values=-np.inf
        )
        centered = np.subtract(
            padded, np.max(padded, axis=1)[:, None], dtype=np.float32
        )
        exponential = np.exp(centered, dtype=np.float32)
        total = np.cumsum(exponential, axis=1, dtype=np.float32)[:, -1]
        result = np.divide(
            exponential[:, :length], total[:, None], dtype=np.float32
        )
        return {"Out": result.reshape(arrays["Out"].shape)}
    raise ValueError("no complete independent oracle for " + family)


def compare(case, actual, expected):
    """Exact arithmetic cases compare bits; nonlinear f32 keeps tolerances."""
    max_abs = 0.0
    tolerant = case["family"] in ("layernorm", "gelu", "softmax")
    for name, want in expected.items():
        got = actual[name]
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(f"{name}: output shape/dtype differs")
        delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
        max_abs = max(max_abs, float(np.max(delta, initial=0)))
        if tolerant:
            np.testing.assert_allclose(
                got, want, rtol=2e-5, atol=2e-5, err_msg=name
            )
        else:
            np.testing.assert_array_equal(
                got.view(np.uint8),
                want.view(np.uint8),
                err_msg=f"{name}: bitwise comparison",
            )
    return max_abs
