# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Execute the legacy lowering's real C wrapper against an expanded-ABI stub.

The migrated examples expose only experimental typed-unranked adapters; this
test exercises the legacy ranked ABI directly, independent of those examples.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_check import (  # noqa: E402
    Structure,
    buddy_opt_path,
    run_lower,
    run_pass,
)
from boscame_sim import op_name, read_module, split_result  # noqa: E402


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    fixture = (
        Path(__file__).parent / "Inputs/reference-matmul-i8-196x768x768.ttir"
    )
    boscame = work / "kernel.mlir"
    lower = work / "kernel.llvm.mlir"
    llvm = work / "kernel.ll"
    code, stderr = run_pass(buddy_opt_path(), fixture, boscame)
    assert code == 0, stderr
    code, stderr = run_lower(buddy_opt_path(), boscame, lower)
    assert code == 0, stderr
    structure = Structure(boscame)
    acc_live = False
    policy_calls = {"read": 0, "write": 0}

    def check_cpu(ops):
        nonlocal acc_live
        for op in ops:
            name = op_name(op.text)
            body = split_result(op.text)[1]
            if name == "bosc_ame.tile.load_acc":
                acc_live = True
            elif name == "bosc_ame.tile.store_acc":
                acc_live = False
            if name in ("func.call", "call"):
                assert not acc_live, op.text
                policy = re.search(r"@bosc_ame_ddr_(read|write)_begin", body)[1]
                policy_calls[policy] += 1
            pointer = None
            if name == "memref.load":
                pointer = re.search(r"memref.load (%\S+)\[", body)[1]
            elif name == "memref.store":
                pointer = re.search(r"memref.store %\S+, (%\S+)\[", body)[1]
            elif name == "linalg.fill":
                pointer = re.search(r"outs\((%\S+)", body)[1]
            if pointer is not None:
                assert structure.roots.get(pointer) != "spm", op.text
            for block in op.blocks:
                check_cpu(block)

    check_cpu(read_module(boscame))
    lowered = lower.read_text()
    starts = len(re.findall(r"llvm.call @bosc_ame_dma_start\(", lowered))
    waits = len(re.findall(r"llvm.call @bosc_ame_dma_wait\(", lowered))
    assert starts == waits == len(structure.dma), (
        starts,
        waits,
        len(structure.dma),
    )
    assert structure.dma_in_loop == 0
    for policy, count in policy_calls.items():
        assert count > 0
        assert (
            len(
                re.findall(
                    rf"llvm.call @bosc_ame_ddr_{policy}_begin\(", lowered
                )
            )
            == count
        )
    # CHECK: PASS: all SPM CPU accesses absent; DMA lowering preserves start/wait counts
    print(
        "PASS: all SPM CPU accesses absent; DMA lowering preserves start/wait counts"
    )
    # CHECK: PASS: DDR policy calls preserved outside AME register live ranges
    print("PASS: DDR policy calls preserved outside AME register live ranges")

    translate = Path(buddy_opt_path()).with_name("buddy-translate")
    run([translate, "--buddy-to-llvmir", lower, "-o", llvm])
    text = llvm.read_text()
    symbol = "triton_matmul_i8_196x768x768"
    wrapper = re.search(
        rf"^define void @_mlir_ciface_{symbol}\(.*?^\}}", text, re.M | re.S
    )[0]
    raw = re.search(rf"^define void @{symbol}\([^\n]+", text, re.M)[0]
    wrapper_llvm = work / "wrapper.ll"
    wrapper_llvm.write_text(
        raw.replace("define", "declare", 1).removesuffix(" {")
        + "\n"
        + wrapper
        + "\n"
    )
    probe = work / "probe.c"
    probe.write_text(r"""
#include <stdint.h>
typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor;
static unsigned char a[64], b[64];
static int32_t c[16];
static unsigned char scratch[4096];
static int calls, failures;
void triton_matmul_i8_196x768x768(
    void *aa, void *ap, int64_t ao, int64_t as, int64_t ast,
    void *ba, void *bp, int64_t bo, int64_t bs, int64_t bst,
    void *ca, void *cp, int64_t co, int64_t cs, int64_t cst,
    void *sa, void *sp, int64_t so, int64_t ss, int64_t sst,
    int32_t gx, int32_t gy, int32_t gz, int32_t x, int32_t y, int32_t z) {
  failures += aa != a || ap != a + 3 || ao != 7 || as != 196 * 768 || ast != 1;
  failures += ba != b || bp != b + 5 || bo != 11 || bs != 768 * 768 || bst != 1;
  failures += ca != c || cp != c + 2 || co != 13 || cs != 196 * 768 * 4 || cst != 1;
  failures += sa != scratch || sp != scratch || so != 17 || ss != 4096 || sst != 1;
  failures += gx != 13 || gy != 48 || gz != 1 || x != calls / 48 || y != calls % 48 || z != 0;
  ++calls;
}
extern void _mlir_ciface_triton_matmul_i8_196x768x768(Descriptor *, Descriptor *, Descriptor *, Descriptor *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
int main(void) {
  Descriptor ma = {a, a + 3, 7, 196 * 768, 1};
  Descriptor mb = {b, b + 5, 11, 768 * 768, 1};
  Descriptor mc = {c, c + 2, 13, 196 * 768 * 4, 1};
  Descriptor ms = {scratch, scratch, 17, 4096, 1};
  for (int32_t x = 0; x < 13; ++x)
    for (int32_t y = 0; y < 48; ++y)
      _mlir_ciface_triton_matmul_i8_196x768x768(&ma, &mb, &mc, &ms, 13, 48, 1, x, y, 0);
  return failures != 0 || calls != 13 * 48;
}
""")
    clang = REPO / "llvm/build/bin/clang"
    if not clang.exists():
        clang = shutil.which("clang")
    assert clang, "clang is needed for the native ABI probe"
    executable = work / "probe"
    run([clang, "-O2", wrapper_llvm, probe, "-o", executable])
    run([executable])
    # CHECK: PASS: generated C wrapper passes all descriptor fields and grid IDs for 624 calls
    print(
        "PASS: generated C wrapper passes all descriptor fields and grid IDs for 624 calls"
    )


if __name__ == "__main__":
    main()
