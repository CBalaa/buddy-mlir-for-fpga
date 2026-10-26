#!/usr/bin/env python3
"""Generate backend f32 coefficients and reproducible error evidence.

The analytic certificate uses fractions only, including rational enclosures of
pi, exp, erf, and ln(2). NumPy dense samples and 90-digit mpmath boundary samples
are separate evidence and are deliberately not called a proof.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HEADER = (
    ROOT
    / "midend/include/Conversion/LowerLinalgToBOSCAME/BOSCAMEMathApproxCoefficients.h"
)
U = Q(1, 2**24)
ETA = Q(1, 2**150)


def alternating_atan(z, count=150):
    value = sum(
        ((-1) ** n * z ** (2 * n + 1) / (2 * n + 1) for n in range(count)), Q()
    )
    next_term = (-1) ** count * z ** (2 * count + 1) / (2 * count + 1)
    return min(value, value + next_term), max(value, value + next_term)


def constants():
    alo, ahi = alternating_atan(Q(1, 5))
    blo, bhi = alternating_atan(Q(1, 239))
    pi_lo, pi_hi = 16 * alo - 4 * bhi, 16 * ahi - 4 * blo
    scale = 10**100
    root_lo = Q(
        math.isqrt(pi_lo.numerator * scale**2 // pi_lo.denominator), scale
    )
    root_hi = Q(
        math.isqrt(pi_hi.numerator * scale**2 // pi_hi.denominator) + 1, scale
    )
    inv_sqrt = (2 / root_hi, 2 / root_lo)
    count = 150
    log_lo = 2 * sum(
        (Q(1, 3) ** (2 * n + 1) / (2 * n + 1) for n in range(count)), Q()
    )
    log_hi = log_lo + 2 * Q(1, 3) ** (2 * count + 1) / (
        (2 * count + 1) * (1 - Q(1, 9))
    )
    return inv_sqrt, (log_lo, log_hi)


TWO_OVER_SQRT_PI, LN2 = constants()


def interval_mul(a, b):
    endpoints = [x * y for x in a for y in b]
    return min(endpoints), max(endpoints)


def exp_positive(z, count=180):
    """Rational enclosure: positive Taylor sum and a geometric tail."""
    assert z >= 0
    term, total = Q(1), Q(1)
    for n in range(1, count + 1):
        term *= z / n
        total += term
    ratio = z / (count + 2)
    assert ratio < 1
    return total, total + term * z / (count + 1) / (1 - ratio)


def exp_negative(z):
    lo, hi = exp_positive(z)
    return 1 / hi, 1 / lo


def erf_interval(x, count=180):
    """Maclaurin alternating remainder; terms decrease after this count."""
    value = sum(
        (
            (-1) ** n * x ** (2 * n + 1) / (math.factorial(n) * (2 * n + 1))
            for n in range(count)
        ),
        Q(),
    )
    following = (
        (-1) ** count
        * x ** (2 * count + 1)
        / (math.factorial(count) * (2 * count + 1))
    )
    assert x * x / count < 1
    return interval_mul(
        TWO_OVER_SQRT_PI,
        (min(value, value + following), max(value, value + following)),
    )


def hermite(n, x):
    return math.factorial(n) * sum(
        (
            (-1) ** j
            * (2 * x) ** (n - 2 * j)
            / (math.factorial(j) * math.factorial(n - 2 * j))
            for j in range(n // 2 + 1)
        ),
        Q(),
    )


def erf_coefficients(center, degree=11):
    result = [erf_interval(center)]
    exponential = exp_negative(center * center)
    for n in range(1, degree + 1):
        multiplier = (
            (-1) ** (n - 1) * hermite(n - 1, center) / math.factorial(n)
        )
        result.append(
            interval_mul(
                TWO_OVER_SQRT_PI,
                interval_mul((multiplier, multiplier), exponential),
            )
        )
    return result


def quantize(interval):
    # The exact rational enclosure below accounts for this chosen binary32
    # value, so the coefficient proof does not depend on float64 tie handling.
    return np.float32(float(sum(interval) / 2))


def rational(value):
    return Q.from_float(float(value))


def upper(value):
    """Print a rational upper bound without rounding its decimal down."""
    return math.nextafter(float(value), math.inf)


def coefficient_error(interval, value):
    return max(abs(rational(value) - end) for end in interval)


def maxabs(interval):
    return max(abs(x) for x in interval)


def horner_bound(coefficients, values, radius):
    """Forward bound for separate binary32 mul/add with gradual underflow."""
    magnitude = maxabs(coefficients[-1])
    error = coefficient_error(coefficients[-1], values[-1])
    for interval, value in zip(coefficients[-2::-1], values[-2::-1]):
        product = radius * magnitude
        error = radius * error + U * radius * (magnitude + error) + ETA
        quantization = coefficient_error(interval, value)
        error += (
            quantization
            + U * (product + maxabs(interval) + error + quantization)
            + ETA
        )
        magnitude = product + maxabs(interval)
    return magnitude, error


CENTERS = [Q(3 + 2 * n, 4) for n in range(7)]
ERF_INTERVALS = [erf_coefficients(center) for center in CENTERS]
ERF_COEFF = np.array(
    [[quantize(c) for c in row] for row in ERF_INTERVALS], dtype=np.float32
)
ERF_SMALL_INTERVALS = [
    interval_mul(
        TWO_OVER_SQRT_PI, (Q((-1) ** n, math.factorial(n) * (2 * n + 1)),) * 2
    )
    for n in range(8)
]
ERF_SMALL = np.array(
    [quantize(c) for c in ERF_SMALL_INTERVALS], dtype=np.float32
)
EXP_INTERVALS = [(Q(1, math.factorial(n)),) * 2 for n in range(8)]
EXP_COEFF = np.array([quantize(c) for c in EXP_INTERVALS], dtype=np.float32)
LN2_HI = np.float32(0.693359375)
LN2_LO = quantize((LN2[0] - rational(LN2_HI), LN2[1] - rational(LN2_HI)))
LOG2E = quantize((1 / LN2[1], 1 / LN2[0]))


def hex_literal(value):
    return float(value).hex() + "f"


def generated_header():
    text = """//===- BOSCAMEMathApproxCoefficients.h - Generated f32 constants ---------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generated by docs/validation/nonlinear-math/generate.py; do not edit.
// See that tool and README.md for domains, certificates, and rounding rules.
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMATHAPPROXCOEFFICIENTS_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMATHAPPROXCOEFFICIENTS_H

namespace buddy::boscame::mathapprox {
// clang-format off
"""
    for name, value in (
        ("kLn2Hi", LN2_HI),
        ("kLn2Lo", LN2_LO),
        ("kLog2E", LOG2E),
    ):
        text += f"inline constexpr float {name} = {hex_literal(value)};\n"
    for name, values in (
        ("kExpCoefficients", EXP_COEFF),
        ("kErfSmallCoefficients", ERF_SMALL),
        ("kErfCenters", CENTERS),
    ):
        text += (
            f"inline constexpr float {name}[] = {{\n  "
            + ",\n  ".join(hex_literal(v) for v in values)
            + "\n};\n"
        )
    text += "inline constexpr float kErfCoefficients[7][12] = {\n"
    for row in ERF_COEFF:
        text += "  {" + ", ".join(hex_literal(v) for v in row) + "},\n"
    text += "};\n// clang-format on\n\n} // namespace buddy::boscame::mathapprox\n\n#endif\n"
    return text


def certificate():
    segments = []
    for center, intervals, values in zip(CENTERS, ERF_INTERVALS, ERF_COEFF):
        a, b, radius, degree = center - Q(1, 4), center + Q(1, 4), Q(1, 4), 11
        absolute_hermite = math.factorial(degree) * sum(
            (
                (2 * b) ** (degree - 2 * j)
                / (math.factorial(j) * math.factorial(degree - 2 * j))
                for j in range(degree // 2 + 1)
            ),
            Q(),
        )
        remainder = (
            TWO_OVER_SQRT_PI[1]
            * exp_negative(a * a)[1]
            * absolute_hermite
            * radius ** (degree + 1)
            / math.factorial(degree + 1)
        )
        _, evaluation = horner_bound(intervals, values, radius)
        segments.append(
            {
                "interval": [float(a), float(b)],
                "taylor_remainder_upper": upper(remainder),
                "coefficient_and_fp32_upper": upper(evaluation),
                "total_abs_upper": upper(remainder + evaluation),
            }
        )
    radius = Q(1, 4)
    z_error = U * radius + ETA
    magnitude, evaluation = horner_bound(
        ERF_SMALL_INTERVALS, ERF_SMALL, radius + z_error
    )
    derivative = sum(
        (
            n * maxabs(ERF_SMALL_INTERVALS[n]) * (radius + z_error) ** (n - 1)
            for n in range(1, 8)
        ),
        Q(),
    )
    small_remainder = (
        TWO_OVER_SQRT_PI[1] * Q(1, 2) ** 17 / (math.factorial(8) * 17)
    )
    small_error = (
        Q(1, 2) * (evaluation + derivative * z_error)
        + U * Q(1, 2) * (magnitude + evaluation)
        + ETA
        + small_remainder
    )
    tail_error = TWO_OVER_SQRT_PI[1] / 8 * exp_negative(Q(16))[1]
    erf_bound = max(
        [s["total_abs_upper"] for s in segments]
        + [upper(small_error), upper(tail_error)]
    )

    log2e_interval = (1 / LN2[1], 1 / LN2[0])
    product_error = (
        104 * coefficient_error(log2e_interval, LOG2E)
        + U * 104 * abs(rational(LOG2E))
        + ETA
    )
    add_error = U * (104 * abs(rational(LOG2E)) + product_error + Q(1, 2)) + ETA
    residual_radius = (Q(1, 2) + product_error + add_error) * LN2[1]
    assert residual_radius < Q(347, 1000)
    # k*ln2_hi is exact: ln2_hi=355/512 and |k|<=150. Bound
    # fl(x-k*hi) using its small result, rather than the sizes of x/k*hi.
    lo_interval = (LN2[0] - rational(LN2_HI), LN2[1] - rational(LN2_HI))
    first_sub_bound = residual_radius + 150 * maxabs(lo_interval)
    first_error = U * first_sub_bound + ETA
    low_error = (
        150 * coefficient_error(lo_interval, LN2_LO)
        + U * 150 * abs(rational(LN2_LO))
        + ETA
    )
    reduction_error = (
        first_error
        + low_error
        + U * (residual_radius + first_error + low_error)
        + ETA
    )
    assert residual_radius + reduction_error < Q(347, 1000)
    radius = Q(347, 1000)
    _, evaluation = horner_bound(EXP_INTERVALS, EXP_COEFF, radius)
    taylor = exp_positive(radius)[1] * radius**8 / math.factorial(8)
    # Relative errors: polynomial absolute error / exp(-R), and the exp
    # sensitivity to reduction. exp(delta)-1 <= delta/(1-delta).
    polynomial_relative = (evaluation + taylor) * exp_positive(radius)[1]
    reduction_relative = reduction_error / (1 - reduction_error)
    relative = (
        polynomial_relative
        + reduction_relative
        + polynomial_relative * reduction_relative
    )
    subnormal_absolute = relative * Q(1, 2**126) + ETA
    assert erf_bound < 2e-6
    assert relative < Q(2, 10**6)
    assert subnormal_absolute < 8 * Q(1, 2**149)
    overflow = np.float32(88.72283935546875)
    previous = np.nextafter(overflow, np.float32(-np.inf))
    maximum = rational(np.finfo(np.float32).max)
    overflow_midpoint = Q(2**128 - 2**103)
    overflow_exp_lower = exp_positive(rational(overflow))[0]
    previous_exp_upper = exp_positive(rational(previous))[1]
    assert overflow_exp_lower > overflow_midpoint
    assert previous_exp_upper * (1 + relative) < maximum
    zero_threshold = np.float32(-103.97208404541016)
    next_nonzero = np.nextafter(zero_threshold, np.float32(np.inf))
    zero_exp_upper = exp_negative(-rational(zero_threshold))[1]
    nonzero_exp_lower = exp_negative(-rational(next_nonzero))[0]
    assert zero_exp_upper < Q(1, 2**150)
    assert nonzero_exp_lower > Q(1, 2**150)
    thresholds = {
        "first_overflow_f32": float(overflow),
        "previous_f32": float(previous),
        "previous_result_finite_even_with_approximation_bound": True,
        "last_zero_f32": float(zero_threshold),
        "next_nonzero_f32": float(next_nonzero),
        "round_to_nearest_thresholds_proved_by_rational_exp_enclosures": True,
    }
    # Composition against the existing f32 oracle (same rounded argument,
    # arithmetic order, and sequential 256-cell positive sum). Both oracles'
    # elementary-call rounding is included. Large erf arguments round exactly
    # to +/-1; |argument|<4 implies |half_x|<3 for the actual f32 constant.
    erf_difference = Q.from_float(erf_bound) + U / 2
    gelu_factor = erf_difference + U * (4 + Q.from_float(erf_bound)) + 2 * ETA
    gelu_output = (
        3 * gelu_factor + U * 3 * (4 + Q.from_float(erf_bound)) + 2 * ETA
    )
    gamma = 255 * U / (1 - 255 * U)
    softmax_ratio = (1 + relative) * (1 + U) ** 2 * (1 + gamma) / (
        (1 - relative) * (1 - U) ** 2 * (1 - gamma)
    ) - 1
    assert gelu_output < Q(2, 10**6)
    assert softmax_ratio < Q(4, 10**5)
    return {
        "operator_composition": {
            "gelu_finite_output_abs_upper": upper(gelu_output),
            "softmax_normal_output_relative_upper": upper(softmax_ratio),
            "softmax_sum_cells": 256,
            "softmax_small_probability_absolute_upper": float(Q(1, 2**90)),
            "existing_output_atol": 2e-5,
            "existing_output_rtol": 2e-5,
            "scope": "source operation order with mathematically RN-f32 elementary references, finite inputs; NumPy correct rounding not assumed; see README for small-probability split",
        },
        "proof": "analytic rational enclosures plus forward binary32 rounding bounds; independent mul/add, RN-even, gradual underflow",
        "erf": {
            "segments": segments,
            "small_interval_abs_upper": upper(small_error),
            "saturated_tail_abs_upper": upper(tail_error),
            "global_finite_abs_upper": erf_bound,
            "contract_max_abs_error": 2e-6,
        },
        "exp": {
            "thresholds": thresholds,
            "reduced_exact_radius_upper": upper(residual_radius),
            "range_reduction_abs_upper": upper(reduction_error),
            "taylor_remainder_upper": upper(taylor),
            "coefficient_and_fp32_abs_upper": upper(evaluation),
            "normal_relative_upper": upper(relative),
            "subnormal_abs_upper": upper(subnormal_absolute),
            "contract_max_rel_error": 2e-6,
            "contract_subnormal_max_abs_error": float(8 * Q(1, 2**149)),
        },
        "not_proved": [
            "correct rounding or <=1 ULP",
            "whole-model accuracy",
            "SFU exp2 accuracy",
            "NaN payload/signaling behavior",
        ],
    }


def f32_exp(values):
    original = np.asarray(values, dtype=np.float32)
    with np.errstate(all="ignore"):
        x = np.clip(
            np.where(np.isnan(original), np.float32(0), original),
            np.float32(-104),
            np.float32(89),
        ).astype(np.float32)
        offset = np.where(x < 0, np.float32(-0.5), np.float32(0.5))
        k = np.add(
            np.multiply(x, LOG2E, dtype=np.float32), offset, dtype=np.float32
        ).astype(np.int32)
        r = np.subtract(
            np.subtract(
                x,
                np.multiply(k.astype(np.float32), LN2_HI, dtype=np.float32),
                dtype=np.float32,
            ),
            np.multiply(k.astype(np.float32), LN2_LO, dtype=np.float32),
            dtype=np.float32,
        )
        p = np.full(x.shape, EXP_COEFF[-1], dtype=np.float32)
        for c in EXP_COEFF[-2::-1]:
            p = np.add(np.multiply(p, r, dtype=np.float32), c, dtype=np.float32)
        k1 = np.clip(k, -126, 127)
        k2 = k - k1
        primary = ((k1 + 127).astype(np.uint32) << np.uint32(23)).view(
            np.float32
        )
        secondary = ((k2 + 127).astype(np.uint32) << np.uint32(23)).view(
            np.float32
        )
        result = np.multiply(
            np.multiply(p, secondary, dtype=np.float32),
            primary,
            dtype=np.float32,
        )
        result = np.where(
            original <= np.float32(-103.97208404541016), np.float32(0), result
        )
        result = np.where(
            original >= np.float32(88.72283935546875),
            np.float32(np.inf),
            result,
        )
        return np.where(np.isnan(original), original, result).astype(np.float32)


def f32_erf(values):
    original = np.asarray(values, dtype=np.float32)
    with np.errstate(all="ignore"):
        x = np.minimum(
            np.where(np.isnan(original), np.float32(0), np.abs(original)),
            np.float32(4),
        ).astype(np.float32)
        sx = np.minimum(x, np.float32(0.5))
        z = np.multiply(sx, sx, dtype=np.float32)
        small = np.full(x.shape, ERF_SMALL[-1], dtype=np.float32)
        for c in ERF_SMALL[-2::-1]:
            small = np.add(
                np.multiply(small, z, dtype=np.float32), c, dtype=np.float32
            )
        small = np.multiply(sx, small, dtype=np.float32)
        lx = np.maximum(x, np.float32(0.5))
        segment = np.minimum(
            ((lx - np.float32(0.5)) * np.float32(2)).astype(np.int32), 6
        )
        h = np.subtract(
            lx, np.array(CENTERS, dtype=np.float32)[segment], dtype=np.float32
        )
        large = ERF_COEFF[segment, -1]
        for n in range(10, -1, -1):
            large = np.add(
                np.multiply(large, h, dtype=np.float32),
                ERF_COEFF[segment, n],
                dtype=np.float32,
            )
        result = np.where(x < np.float32(0.5), small, large)
        result = np.where(x >= np.float32(4), np.float32(1), result)
        result = np.copysign(result, original)
        result = np.where(original == 0, original, result)
        return np.where(np.isnan(original), original, result).astype(np.float32)


def neighbors(value, count=4):
    result = [np.float32(value)]
    lo = hi = result[0]
    for _ in range(count):
        lo = np.nextafter(lo, np.float32(-np.inf))
        hi = np.nextafter(hi, np.float32(np.inf))
        result.extend((lo, hi))
    return result


def samples(count):
    import mpmath as mp
    from scipy import special

    erf_x = np.linspace(-8, 8, count, dtype=np.float32)
    erf_x = np.unique(
        np.concatenate(
            (
                erf_x,
                np.array(
                    [v for b in np.arange(-4, 4.25, 0.5) for v in neighbors(b)],
                    dtype=np.float32,
                ),
            )
        )
    )
    exp_x = np.linspace(
        -104, np.float32(88.72283172607422), count, dtype=np.float32
    )
    critical = [
        v for k in range(-150, 129) for v in neighbors((k + 0.5) * math.log(2))
    ]
    critical += (
        neighbors(-103.97207708399179, 16)
        + neighbors(-126 * math.log(2), 16)
        + neighbors(88.72283905206835, 16)
    )
    exp_x = np.unique(
        np.concatenate((exp_x, np.array(critical, dtype=np.float32)))
    )
    erf_y = f32_erf(erf_x)
    erf_ref = special.erf(erf_x.astype(np.float64))
    erf_error = np.abs(erf_y.astype(np.float64) - erf_ref)
    exp_y = f32_exp(exp_x)
    with np.errstate(all="ignore"):
        exp_ref = np.exp(exp_x.astype(np.float64))
    finite = exp_ref <= np.finfo(np.float32).max
    normal = finite & (exp_ref >= np.finfo(np.float32).tiny)
    subnormal = finite & ~normal
    relative = np.abs(exp_y[normal].astype(np.float64) / exp_ref[normal] - 1)
    suberror = np.abs(exp_y[subnormal].astype(np.float64) - exp_ref[subnormal])
    mp.mp.dps = 90
    boundaries = []
    for kind, values, implementation, reference in (
        (
            "erf",
            [v for b in np.arange(0, 4.25, 0.5) for v in neighbors(b)],
            f32_erf,
            mp.erf,
        ),
        ("exp", critical, f32_exp, mp.exp),
    ):
        maximum = mp.mpf(0)
        for x in values:
            exact = reference(mp.mpf(float(x)))
            actual = float(implementation(np.array([x], dtype=np.float32))[0])
            if math.isfinite(actual):
                maximum = max(
                    maximum,
                    abs(mp.mpf(actual) - exact)
                    / (
                        abs(exact)
                        if kind == "exp" and exact >= mp.mpf(2) ** -126
                        else 1
                    ),
                )
        boundaries.append(
            {
                "function": kind,
                "count": len(values),
                "mpmath_digits": 90,
                "metric": "exp normal relative, otherwise absolute",
                "maximum": float(maximum),
            }
        )
    special_x = np.array(
        [
            np.nan,
            -np.inf,
            np.inf,
            -0.0,
            0.0,
            -np.nextafter(np.float32(0), np.float32(1)),
            np.nextafter(np.float32(0), np.float32(1)),
            -np.finfo(np.float32).max,
            np.finfo(np.float32).max,
        ],
        dtype=np.float32,
    )
    for function in (f32_erf, f32_exp):
        output = function(special_x)
        assert np.isnan(output[0])
        assert not np.isnan(output[1:]).any()
    assert np.signbit(f32_erf(special_x)[3])
    assert np.array_equal(
        f32_exp(special_x)[1:5], np.array([0, np.inf, 1, 1], dtype=np.float32)
    )
    assert np.array_equal(
        f32_erf(special_x)[1:5], np.array([-1, 1, -0.0, 0.0], dtype=np.float32)
    )
    assert float(np.max(erf_error)) < 2e-6
    assert float(np.max(relative)) < 2e-6
    assert float(np.max(suberror)) < float(8 * Q(1, 2**149))
    return {
        "kind": "dense numerical evidence, not a strict bound",
        "erf": {
            "samples": int(erf_x.size),
            "reference": "scipy.special.erf evaluated in binary64",
            "max_abs": float(np.max(erf_error)),
            "at": float(erf_x[np.argmax(erf_error)]),
        },
        "exp": {
            "samples": int(exp_x.size),
            "reference": "NumPy binary64 exp",
            "normal_max_rel": float(np.max(relative)),
            "subnormal_max_abs": float(np.max(suberror)),
            "subnormal_max_min_subnormal_units": float(np.max(suberror))
            / float(np.nextafter(np.float32(0), np.float32(1))),
        },
        "high_precision_boundaries": boundaries,
        "special_cases": "NaN, +/-Inf, +/-0, +/-minimum subnormal, +/-maximum finite passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-header", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--samples", type=int, default=1000001)
    parser.add_argument(
        "--quick", action="store_true", help="certificate and constants only"
    )
    args = parser.parse_args()
    header = generated_header()
    if args.write_header:
        HEADER.write_text(header)
    elif not HEADER.exists() or HEADER.read_text() != header:
        raise SystemExit(
            "generated coefficients differ; run with --write-header"
        )
    report = {
        "coefficient_sha256": hashlib.sha256(header.encode()).hexdigest(),
        "certificate": certificate(),
    }
    if not args.quick:
        report["samples"] = samples(args.samples)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
