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


source = ".text\n.Lentry:\nvsetivli zero, 16, e32, m4, ta, ma\n"
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

# Prove e32 across local blocks only when every reachable predecessor agrees.
cfg = """.text
.globl kernel
.type kernel,@function
kernel:
  vsetivli zero, 16, e32, m4, ta, ma
  beq a0, zero, .Lright
.Lleft:
  j .Lmerge
.Lright:
  j .Lmerge
.Lmerge:
  vfsub.vf v8, v8, fs0
  vmv.s.x v12, zero
  vfredmax.vs v8, v8, v12
  ret
"""
expanded = transform(cfg, expand_f32_sub=True, expand_f32_scalar_moves=True)
assert "vfsub.vf" not in expanded and "vmv.s.x" not in expanded
assert expanded.count("fneg.s\tfs0, fs0") == 2
assert "fmv.w.x\tft11, zero" in expanded
assert "vfmv.s.f\tv12, ft11" in expanded
assert "fsd\tft11, 0(sp)" in expanded
assert "fld\tft11, 0(sp)" in expanded


def rejects_vector_expansion(text, *, scalar_move=False, subtract=False):
    try:
        transform(
            text, expand_f32_scalar_moves=scalar_move, expand_f32_sub=subtract
        )
    except ValueError:
        return
    raise AssertionError("unproved vector SEW accepted")


for vtype in (
    "vsetivli zero, 16, e16, m2, ta, ma",
    "vsetivli zero, 16, e64, m1, ta, ma",
    "vsetvl zero, a0, a1",
):
    rejects_vector_expansion(
        f".text\n.Lentry:\n{vtype}\nvmv.s.x v8, zero\n",
        scalar_move=True,
    )

# A function entry remains unknown even if its lexical predecessor set e32.
rejects_vector_expansion(
    ".text\n.Lsetup:\nvsetivli zero, 16, e32, m4, ta, ma\n"
    ".globl kernel\n.type kernel,@function\nkernel:\nvmv.s.x v8, zero\n",
    scalar_move=True,
)
# Join disagreement, calls, and opaque indirect control flow invalidate proof.
rejects_vector_expansion(
    cfg.replace(
        ".Lright:\n  j .Lmerge",
        ".Lright:\n  vsetivli zero, 16, e16, m2, ta, ma\n  j .Lmerge",
    ),
    scalar_move=True,
)
for opaque in (
    "call helper",
    "jalr a0",
    "jr a0",
    "c.jr a0",
    "vsetvl zero, a0, a1",
    "mret",
    "sret",
    "uret",
    "dret",
    "ecall",
    "ebreak",
    "c.ebreak",
    "cm.jt a0",
    "cm.popret",
    "c.jal .Lmerge",
    "c.ret",
    "tail helper",
    "c.bltz a0, .Lmerge",
    ".word 0x0000006f",
):
    rejects_vector_expansion(
        cfg.replace(
            ".Lright:\n  j .Lmerge", ".Lright:\n  " + opaque + "\n  j .Lmerge"
        ),
        scalar_move=True,
    )
rejects_vector_expansion(
    source + "ret\n.Ldead:\nvmv.s.x v8, zero\n",
    scalar_move=True,
)
rejects_vector_expansion(
    source + "j .Lmissing\n.Ldead:\nvmv.s.x v8, zero\n",
    scalar_move=True,
)
rejects_vector_expansion(source + "vmv.s.x v8, sp\n", scalar_move=True)
rejects_vector_expansion(source + "vmv.s.x v32, zero\n", scalar_move=True)
rejects_vector_expansion(source + "vmv.s.x v8, zero, v0.t\n", scalar_move=True)
# Calls/opaque edges also prevent subtraction expansion across the join.
rejects_vector_expansion(
    cfg.replace(
        ".Lright:\n  j .Lmerge", ".Lright:\n  call helper\n  j .Lmerge"
    ),
    subtract=True,
)

# A loop header's backedge agrees with the entry path; a later vset to a
# different SEW must make the entire header unknown on the next iteration.
loop = source + ".Lloop:\nvmv.s.x v8, a0\nbnez a1, .Lloop\nret\n"
assert "vfmv.s.f\tv8, ft11" in transform(loop, expand_f32_scalar_moves=True)
assert "vfmv.s.f\tv8, ft11" in transform(loop, expand_f32_sub=True)
rejects_vector_expansion(
    loop.replace("bnez", "vsetivli zero, 16, e64, m1, ta, ma\nbnez"),
    scalar_move=True,
)

# Both moves copy the low 32 bits, including arbitrary NaN payloads and sign
# bits. fmv.w.x NaN-boxes its scalar result; vfmv.s.f does not canonicalize it.
# Neither form modifies VL/vtype or lanes beyond element zero, even at VL=0.
patterns = (
    0,
    0x80000000,
    1,
    0x7F800000,
    0xFF800000,
    0x7FC01234,
    0x7F801234,
    0xFFFFFFFF,
)
for vl in (0, 1, 16):
    for pattern in patterns:
        before = [0xCDCDCDCD] * 16
        direct, expanded = before.copy(), before.copy()
        boxed = 0xFFFFFFFF00000000 | pattern
        if vl:
            direct[0] = pattern
            expanded[0] = boxed & 0xFFFFFFFF
        assert direct == expanded

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
