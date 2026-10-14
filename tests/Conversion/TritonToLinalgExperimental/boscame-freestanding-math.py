# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""The scalar math.rsqrt libcall links and matches rounded f32 sqrt/divide."""

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
clang = REPO / "llvm/build/bin/clang"
if not clang.is_file():
    clang = shutil.which("clang")
library = work / "math.so"
subprocess.run(
    [
        str(clang),
        "-O2",
        "-shared",
        "-fPIC",
        "-fno-builtin",
        "-ffp-contract=off",
        "-Wl,-Bsymbolic",
        str(REPO / "runtime/boscame/nr/nr_math.c"),
        "-lm",
        "-o",
        str(library),
    ],
    check=True,
)
math = ctypes.CDLL(str(library))
math.rsqrtf.argtypes = [ctypes.c_float]
math.rsqrtf.restype = ctypes.c_float
rng = np.random.default_rng(971)
words = np.concatenate(
    [
        np.array(
            [0, 0x80000000, 1, 0x7FFFFF, 0x800000, 0x7F7FFFFF], dtype=np.uint32
        ),
        rng.integers(1, 0x7F800000, 10000, dtype=np.uint32),
        np.array(
            [0x7F800000, 0xFF800000, 0x7FC01234, 0xBF800000], dtype=np.uint32
        ),
    ]
)
values = words.view(np.float32)
actual = np.array(
    [math.rsqrtf(float(value)) for value in values], dtype=np.float32
)
with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
    expected = np.divide(
        np.float32(1), np.sqrt(values, dtype=np.float32), dtype=np.float32
    )
finite_or_inf = ~np.isnan(expected)
np.testing.assert_array_equal(
    actual[finite_or_inf].view(np.uint32),
    expected[finite_or_inf].view(np.uint32),
)
assert np.isnan(actual[~finite_or_inf]).all()
# CHECK: PASS freestanding rsqrtf: rounded f32 operations and special values
print("PASS freestanding rsqrtf: rounded f32 operations and special values")
