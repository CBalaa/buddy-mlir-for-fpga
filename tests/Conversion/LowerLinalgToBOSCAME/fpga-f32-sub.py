# RUN: %PYTHON %s | FileCheck %s
"""Keep unsupported forms rejected; check the opt-in f32 subtraction expansion."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from restrict_fpga_assembly import transform  # noqa: E402


def rejected(source, **options):
    try:
        transform(source, **options)
    except ValueError:
        return
    raise AssertionError("unsupported assembly accepted")


source = ".text\nentry:\nvsetivli zero, 16, e32, m4, ta, ma\n"
sub = "vfsub.vf v8, v8, fs0\n"
rejected(source + sub)
output = transform(source + sub, expand_f32_sub=True)
assert "vfsub.vf" not in output
assert output.count("fneg.s\tfs0, fs0") == 2
assert "vfadd.vf\tv8, v8, fs0" in output
masked = transform(source + sub.rstrip() + ", v0.t\n", expand_f32_sub=True)
assert "vfadd.vf\tv8, v8, fs0, v0.t" in masked
for invalid in (
    sub,
    source.replace("e32", "e64") + sub,
    source + "next:\n" + sub,
    source + "call other\n" + sub,
    source + "j next\n" + sub,
    source + "vfsub.vf v32, v8, fs0\n",
):
    rejected(invalid, expand_f32_sub=True)
data = '.section .rodata\n.ascii "vfsub.vf v8, v8, fs0"\n'
assert transform(data, expand_f32_sub=True) == data

# Independently compare subtraction with sign-bit inversion followed by add.
# Include subnormals, extrema, both zeros, infinities and NaNs. NumPy uses
# host IEEE round-to-nearest; RISC-V canonicalizes NaN results for both forms.
bits = np.array(
    [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x3F800000,
        0xBF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0x7F801234,
    ],
    dtype=np.uint32,
)
rng = np.random.default_rng(42)
bits = np.concatenate((bits, rng.integers(0, 2**32, 128, dtype=np.uint32)))
x, y = bits.view(np.float32)[:, None], bits.view(np.float32)[None, :]
neg = np.bitwise_xor(bits, np.uint32(0x80000000)).view(np.float32)[None, :]
assert np.array_equal(
    np.bitwise_xor(neg.view(np.uint32), 0x80000000), bits[None, :]
)
with np.errstate(all="ignore"):
    expected, actual = x - y, x + neg
assert np.array_equal(np.isnan(actual), np.isnan(expected))
finite = ~np.isnan(expected)
assert np.array_equal(
    actual.view(np.uint32)[finite], expected.view(np.uint32)[finite]
)

# CHECK: PASS FPGA f32 subtraction expansion and unsupported-form checks
print("PASS FPGA f32 subtraction expansion and unsupported-form checks")
