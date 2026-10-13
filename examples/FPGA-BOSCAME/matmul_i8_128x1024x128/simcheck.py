#!/usr/bin/env python3
"""Bit-exact DDR-level check of the generated `--triton-to-boscame` kernel.

`make host` verifies the Triton-to-linalg path, which does not contain the
DMA/SPM/AME datapath at all.  This script runs the *AME* output instead:
every program of the launch grid is executed by `tools/boscame_sim.py` (the
functional NH/E6 model -- real DMA contract, two 128 KiB SPM banks, AME tile
semantics) and the resulting C is compared against an independent NumPy INT32
oracle.  The scratch window starts poisoned, so the check also proves the pass
initializes it before use.

    python3 simcheck.py [--module PATH] [--programs N]

The shape, grid and default module path come from `build.py`, so there is one
place where the problem size is written down.  Requires numpy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
import build as example  # noqa: E402  (frontend owner of the fixed shape)

sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
import boscame_sim  # noqa: E402


def operands():
    """The deterministic A, B and initial C the launcher fills (launch.c)."""
    a = (
        np.arange(example.M)[:, None] * 13 + np.arange(example.K)[None, :] * 7
    ) % 31 - 15
    b = (
        np.arange(example.N)[:, None] * 11 + np.arange(example.K)[None, :] * 5
    ) % 29 - 14
    c = (
        np.arange(example.M)[:, None] * 3 + np.arange(example.N)[None, :]
    ) % 7 - 3
    return a.astype(np.int8), b.astype(np.int8), c.astype("<i4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        type=Path,
        default=example.OUT / "kernel.boscame.mlir",
        help="pass output (default: build/<name>/kernel.boscame.mlir)",
    )
    parser.add_argument(
        "--programs",
        type=int,
        default=0,
        help="stop after N programs (default: the whole grid)",
    )
    args = parser.parse_args()
    if not args.module.is_file():
        raise SystemExit("missing " + str(args.module) + "; run `make boscame`")

    a, b, c0 = operands()
    buffers = {
        "%arg0": bytearray(a.tobytes()),
        "%arg1": bytearray(b.tobytes()),
        "%arg2": bytearray(c0.tobytes()),
        # Deliberately poisoned: the pass initializes scratch every program.
        "%arg3": bytearray([0xAA] * example.SCRATCH_BYTES),
    }
    total = example.GRID[0] * example.GRID[1] * example.GRID[2]
    limit = args.programs or total
    launched = 0
    for x in range(example.GRID[0]):
        for y in range(example.GRID[1]):
            for z in range(example.GRID[2]):
                if launched >= limit:
                    break
                boscame_sim.simulate(
                    args.module,
                    example.SYMBOL,
                    buffers,
                    {
                        "%arg4": example.GRID[0],
                        "%arg5": example.GRID[1],
                        "%arg6": example.GRID[2],
                        "%arg7": x,
                        "%arg8": y,
                        "%arg9": z,
                    },
                )
                launched += 1

    want = c0.astype(np.int64) + a.astype(np.int64) @ b.astype(np.int64).T
    got = np.frombuffer(bytes(buffers["%arg2"]), dtype="<i4").reshape(
        example.M, example.N
    )
    max_abs = int(np.abs(got.astype(np.int64) - want).max())
    print(f"simulated {launched}/{total} programs, max_abs={max_abs}")
    if launched != total or max_abs:
        print("SIM CHECK FAIL")
        return 1
    print(
        f"SIM CHECK PASS: {example.M * example.N} outputs match the NumPy "
        f"INT32 oracle after {total} programs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
