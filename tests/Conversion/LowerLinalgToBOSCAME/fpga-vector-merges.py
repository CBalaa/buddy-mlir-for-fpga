# RUN: %PYTHON %s | FileCheck %s
"""Preserve merge lanes and all GPRs using only the allowed scalar merge."""

import random
import re
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from nr_isa import RVV_ALLOWED  # noqa: E402
from restrict_fpga_assembly import transform  # noqa: E402


def rejected(source, **options):
    try:
        transform(source, **options)
    except ValueError:
        return
    raise AssertionError("unsupported assembly accepted: " + source)


def check_execution(immediate, destination, source, sew, vl):
    """Interpret every expanded instruction, with no assumed free registers."""
    assembly = transform(
        f"vmerge.vim v{destination}, v{source}, {immediate}, v0\n",
        expand_vector_merges=True,
    )
    rng = random.Random(42)
    gprs = [rng.getrandbits(64) for _ in range(32)]
    gprs[0], gprs[2] = 0, 0x80001000
    before_gprs = list(gprs)
    # Guards include live caller data starting at sp and bytes below the
    # temporary slot. Only the newly allocated [sp-16,sp) slot may change.
    memory = {
        address: rng.randrange(256)
        for address in range(gprs[2] - 32, gprs[2] + 32)
    }
    before_memory = dict(memory)
    vectors = [rng.getrandbits(256) for _ in range(32)]
    before_vectors = list(vectors)
    expected = list(vectors)
    lane_mask = (1 << sew) - 1
    for lane in range(vl):
        # An independent immediate-merge reference reads all inputs before
        # writing vd; this also tests vd==vs2 and vs2==v0.
        value = (
            immediate & lane_mask
            if (before_vectors[0] >> lane) & 1
            else (before_vectors[source] >> (lane * sew)) & lane_mask
        )
        expected[destination] &= ~(lane_mask << (lane * sew))
        expected[destination] |= value << (lane * sew)

    for line in assembly.splitlines():
        fields = re.split(r"[\s,()]+", line.strip())
        instruction = fields[0]
        if instruction == "addi":
            assert fields[1:3] == ["sp", "sp"]
            gprs[2] += int(fields[3])
            assert gprs[2] % 16 == 0
        elif instruction == "sd":
            assert fields[1:4] == ["t6", "0", "sp"]
            for offset in range(8):
                memory[gprs[2] + offset] = (gprs[31] >> (8 * offset)) & 255
        elif instruction == "li":
            assert fields[1] == "t6"
            gprs[31] = int(fields[2]) & ((1 << 64) - 1)
        elif instruction == "vmerge.vxm":
            assert fields[1:] == [f"v{destination}", f"v{source}", "t6", "v0"]
            old_source, mask = vectors[source], vectors[0]
            for lane in range(vl):
                value = (
                    gprs[31] & lane_mask
                    if (mask >> lane) & 1
                    else (old_source >> (lane * sew)) & lane_mask
                )
                vectors[destination] &= ~(lane_mask << (lane * sew))
                vectors[destination] |= value << (lane * sew)
        elif instruction == "ld":
            assert fields[1:4] == ["t6", "0", "sp"]
            gprs[31] = sum(
                memory[gprs[2] + offset] << (8 * offset) for offset in range(8)
            )
        else:
            raise AssertionError("unexpected expansion: " + line)
    assert vectors == expected
    assert gprs == before_gprs
    assert all(
        memory[address] == value
        for address, value in before_memory.items()
        if not before_gprs[2] - 16 <= address < before_gprs[2]
    )


def main():
    assert "vmerge.vxm" in RVV_ALLOWED and "vmerge.vim" not in RVV_ALLOWED
    for immediate in range(-16, 16):
        for destination, source in ((8, 12), (8, 8), (8, 0), (31, 1)):
            for sew in (8, 16, 32, 64):
                for vl in (0, 1, 256 // sew):
                    check_execution(immediate, destination, source, sew, vl)
    for literal, value in (
        ("-0x10", -16),
        ("+0Xf", 15),
        ("+0", 0),
        ("015", 15),
    ):
        assert f"li\tt6, {value}" in transform(
            f"vmerge.vim v8, v0, {literal}, v0\n", expand_vector_merges=True
        )
    valid = "vmerge.vim v8, v8, -1, v0\n"
    rejected(valid)
    for invalid in (
        "vmerge.vim v0, v8, 1, v0\n",
        "vmerge.vim v32, v8, 1, v0\n",
        "vmerge.vim v8, v32, 1, v0\n",
        "vmerge.vim v08, v8, 1, v0\n",
        "vmerge.vim v8, v8, 16, v0\n",
        "vmerge.vim v8, v8, -17, v0\n",
        "vmerge.vim v8, v8, 31, v0\n",
        "vmerge.vim v8, v8, 1, v1\n",
        "vmerge.vim v8, v8, 1, v0.t\n",
        "vmerge.vim v8, v8, 1\n",
        "vmerge.vim v8, v8, 1, v0, v0\n",
        "vmerge.vim v8, v8, 1+1, v0\n",
        "vmerge.vim v8, v8, symbol, v0\n",
        "vmerge.vim v8, v8, 1.0, v0\n",
        valid.rstrip() + "; ret\n",
        "entry: vmerge.vim v8, v8, 16, v0\n",
        "vfsub.vf v8, v8, fs0\n",
    ):
        rejected(invalid, expand_vector_merges=True)
    expanded = transform(valid, expand_vector_merges=True)
    assert len(expanded.splitlines()) == 6
    assert "vmerge.vim" not in expanded
    assert not re.search(r"\bv(?:s[1248]r|l[1248]r|lenb)\b", expanded)
    for options in ({"insert_fences": False}, {"coalesce_fences": True}):
        assert (
            transform(valid, expand_vector_merges=True, **options) == expanded
        )
    assert (
        transform("entry: " + valid, expand_vector_merges=True)
        == "entry:\n" + expanded
    )
    data = '.section .rodata\n.ascii "vmerge.vim v8, v8, 1, v0"\n'
    assert transform(data, expand_vector_merges=True) == data
    nested = '.text\n.pushsection .data,"aw",@progbits\n'
    assert transform(
        nested + valid + ".popsection\n" + valid, expand_vector_merges=True
    ) == (nested + valid + ".popsection\n" + expanded)
    cli = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "restrict_fpga_assembly.py"),
            "--expand-vector-merges",
        ],
        input=valid,
        capture_output=True,
        text=True,
        check=True,
    )
    assert cli.stdout == expanded
    # CHECK: PASS FPGA immediate merges: signed5, aliases, mask, GPR and stack
    print("PASS FPGA immediate merges: signed5, aliases, mask, GPR and stack")


if __name__ == "__main__":
    main()
