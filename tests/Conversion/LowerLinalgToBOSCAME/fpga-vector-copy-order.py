# RUN: %PYTHON %s 2>&1 | FileCheck %s
"""Check the NR byte-copy schedule and its register/control-flow boundaries."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from restrict_fpga_assembly import transform  # noqa: E402

LOAD = "vle8.v v8, (a5) # source bytes\n"
ADD = "addi a4, a4, 64 # next column\n"
STORE = "vse8.v v8, (a6) # destination bytes\n"
COPY = LOAD + ADD + STORE
ORDERED = LOAD + STORE + ADD


def ordered(source):
    return transform(source, order_vector_copies=True)


def unchanged(source):
    try:
        expected = transform(source)
    except ValueError as original:
        try:
            ordered(source)
        except ValueError as actual:
            assert str(actual) == str(original)
        else:
            raise AssertionError("copy ordering bypassed validation")
    else:
        assert ordered(source) == expected, source


def main():
    # The opt-in changes only the independent counter's position. Existing
    # load/store fencing and each original inline comment remain intact.
    assert ordered(COPY) == transform(ORDERED)
    assert transform(COPY) != transform(ORDERED)
    commented = LOAD + "# counter comment\n\n" + ADD + "// copy\n" + STORE
    expected = LOAD + "# counter comment\n\n" + STORE + "// copy\n" + ADD
    assert ordered(commented) == transform(expected)
    aliases = COPY.replace("addi a4, a4", "addi x14, a4")
    assert ordered(aliases) == transform(
        LOAD + STORE + "addi x14, a4, 64 # next column\n"
    )

    # Address aliases matter: moving a write to the destination address would
    # change the actual bytes written, even when its spelling differs.
    for source in (
        COPY.replace("addi a4, a4", "addi x16, a6"),
        COPY.replace("addi a4, a4", "addi x15, a5"),
        COPY.replace("(a6)", "(fp)").replace("addi a4, a4", "addi s0, x8"),
        COPY.replace("addi a4, a4", "addi a4, a3"),
        COPY.replace("addi a4, a4", "addi x0, zero"),
        COPY.replace("addi a4, a4", "addi x2, sp"),
        COPY.replace("64 # next", "4096 # next"),
        COPY.replace("vle8.v v8", "vle8.v v12"),
        COPY.replace("vse8.v v8", "vse8.v v12"),
        COPY.replace("(a5) #", "(a5), v0.t #"),
        COPY.replace("(a6) #", "(a6), v0.t #"),
        COPY.replace("vle8.v", "vle16.v").replace("vse8.v", "vse16.v"),
        COPY.replace("vle8.v", "vle32.v").replace("vse8.v", "vse32.v"),
    ):
        unchanged(source)
    for barrier in (
        "label:\n",
        ".cfi_def_cfa_offset 32\n",
        ".loc 1 2 3\n",
        ".p2align 2\n",
        "fence rw, rw\n",
        "ld t0, 0(sp)\n",
        "sd t0, 0(sp)\n",
        "call helper\n",
        "beq a0, a1, label\n",
        "vsetivli zero, 16, e8, m1, ta, ma\n",
        "vadd.vv v8, v8, v8\n",
        "addi t0, t0, 1\n",
    ):
        unchanged(LOAD + barrier + ADD + STORE)
        unchanged(LOAD + ADD + barrier + STORE)
    unchanged("label: " + COPY)
    unchanged(LOAD + ADD + "label: " + STORE)

    # Reuse the transform's section state; instruction-looking data, nested
    # section switches, and .previous must never trigger this rewrite.
    for header in (
        ".data\n",
        ".rodata\n",
        '.section .copy_data,"aw",@progbits\n',
        ".data\n.text\n.previous\n",
    ):
        assert ordered(header + COPY) == header + COPY
    nested = '.text\n.pushsection .copy_data,"aw",@progbits\n'
    assert ordered(nested + COPY + ".popsection\n" + COPY) == (
        nested + COPY + ".popsection\n" + transform(ORDERED)
    )
    # CHECK: PASS NR copy ordering: independent counter, aliases and section barriers
    print(
        "PASS NR copy ordering: independent counter, aliases and section barriers"
    )


if __name__ == "__main__":
    main()
