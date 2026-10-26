# RUN: %PYTHON %s
"""Independent references and rational bounds for authorized math recipes."""

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "math_approx_evidence",
    ROOT / "docs/validation/nonlinear-math/generate.py",
)
APPROX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APPROX)


def test_certificates_and_coefficients():
    assert APPROX.HEADER.read_text() == APPROX.generated_header()
    certificate = APPROX.certificate()
    assert certificate["erf"]["global_finite_abs_upper"] < 2e-6
    assert certificate["exp"]["normal_relative_upper"] < 2e-6
    assert certificate["exp"]["subnormal_abs_upper"] < 8 * 2**-149
    assert certificate["exp"]["thresholds"][
        "previous_result_finite_even_with_approximation_bound"
    ]


def test_boundaries_and_arbitrary_binary32():
    rng = np.random.default_rng(3072197)
    # Random encodings exercise gradual underflow, signs, huge finite values,
    # and NaNs. Dyadic segment endpoints get explicit immediate neighbors.
    bits = rng.integers(0, 2**32, 20000, dtype=np.uint32)
    values = bits.view(np.float32)
    erf_edges = np.array(
        [v for b in np.arange(-4, 4.25, 0.5) for v in APPROX.neighbors(b)],
        dtype=np.float32,
    )
    erf_input = np.concatenate((values, erf_edges))
    erf_actual = APPROX.f32_erf(erf_input)
    erf_reference = np.array(
        [math.erf(float(v)) for v in erf_input], dtype=np.float64
    )
    finite = np.isfinite(erf_reference)
    assert np.max(np.abs(erf_actual[finite] - erf_reference[finite])) < 2e-6
    assert np.array_equal(np.isnan(erf_actual), np.isnan(erf_reference))

    exp_edges = np.array(
        [
            v
            for k in (-150, -126, -1, 0, 1, 127, 128)
            for v in APPROX.neighbors((k + 0.5) * math.log(2))
        ]
        + APPROX.neighbors(-103.97207708399179, 8)
        + APPROX.neighbors(88.72283905206835, 8),
        dtype=np.float32,
    )
    exp_input = np.concatenate((values, exp_edges))
    exp_actual = APPROX.f32_exp(exp_input)
    with np.errstate(all="ignore"):
        exp_reference = np.exp(exp_input.astype(np.float64))
    normal = (exp_reference >= np.finfo(np.float32).tiny) & (
        exp_reference <= np.finfo(np.float32).max
    )
    subnormal = exp_reference < np.finfo(np.float32).tiny
    assert np.max(np.abs(exp_actual[normal] / exp_reference[normal] - 1)) < 2e-6
    assert (
        np.max(np.abs(exp_actual[subnormal] - exp_reference[subnormal]))
        < 8 * 2**-149
    )
    assert np.array_equal(np.isnan(exp_actual), np.isnan(exp_reference))
    overflow = exp_reference > np.finfo(np.float32).max
    assert np.isposinf(exp_actual[overflow]).all()


def test_special_and_subnormal():
    minimum = np.nextafter(np.float32(0), np.float32(1))
    values = np.array(
        [np.nan, -np.inf, np.inf, -0.0, 0.0, -minimum, minimum],
        dtype=np.float32,
    )
    erf = APPROX.f32_erf(values)
    exponential = APPROX.f32_exp(values)
    assert np.isnan(erf[0]) and np.isnan(exponential[0])
    assert np.array_equal(erf[1:5], [-1, 1, -0.0, 0])
    assert np.array_equal(exponential[1:], [0, np.inf, 1, 1, 1, 1])
    assert np.signbit(erf[3]) and not np.signbit(erf[4])
    assert erf[5] == -minimum and erf[6] == minimum
    # Check both sides of the exact binary32 underflow and overflow switches.
    thresholds = APPROX.certificate()["exp"]["thresholds"]
    x = np.array(
        [
            thresholds["last_zero_f32"],
            thresholds["next_nonzero_f32"],
            thresholds["previous_f32"],
            thresholds["first_overflow_f32"],
        ],
        dtype=np.float32,
    )
    y = APPROX.f32_exp(x)
    assert y[0] == 0 and y[1] == minimum
    assert np.isfinite(y[2]) and np.isposinf(y[3])


test_certificates_and_coefficients()
test_boundaries_and_arbitrary_binary32()
test_special_and_subnormal()
