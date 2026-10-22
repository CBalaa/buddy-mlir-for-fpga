# RUN: %PYTHON %s 2>&1 | FileCheck %s
"""Restrict the NR add scheduling workaround to the measured i32 loop."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from restrict_fpga_assembly import transform  # noqa: E402

CONFIGURATION = "vsetivli zero, 16, e32, m4, ta, ma\n"
LEFT = "vle32.v v8, (a5) # accumulator\n"
RIGHT = "vle32.v v12, (a4) # product\n"
ADD = "vadd.vv v8, v8, v12\n"
COUNTER = "addi a3, a3, 16 # element counter\n"
STORE = "vse32.v v8, (a5) # accumulator result\n"
STEPS = "addi a5, a5, 64\naddi a4, a4, 64\nbge a2, a3, loop\n"
LOOP = "loop:\n" + CONFIGURATION + LEFT + RIGHT + ADD + COUNTER + STORE + STEPS
EXPECTED = LOOP.replace(COUNTER, "\tc.nop\n").replace(STORE, STORE + COUNTER)


def ordered(source, **options):
    return transform(source, order_vector_adds=True, **options)


def unchanged(source):
    try:
        expected = transform(source)
    except ValueError as original:
        try:
            ordered(source)
        except ValueError as actual:
            assert str(actual) == str(original)
        else:
            raise AssertionError("add ordering bypassed validation")
    else:
        assert ordered(source) == expected, source


def main():
    # Preserve the arithmetic, original comments, store address, and both
    # adjacent fences. The counter must follow the store's trailing fence.
    assert ordered(LOOP) == transform(EXPECTED)
    assert transform(LOOP) != transform(EXPECTED)
    assert ADD + "\tc.nop\n\tfence\trw, rw\n" + STORE in ordered(LOOP)
    assert STORE + "\tfence\trw, rw\n" + COUNTER in ordered(LOOP)
    assert ordered(LOOP).count(COUNTER) == 1
    assert ordered(LOOP, coalesce_fences=True) == transform(
        EXPECTED, coalesce_fences=True
    )
    aliases = LOOP.replace("zero", "x0").replace("a3", "x13")
    assert ordered(aliases) == transform(
        EXPECTED.replace("zero", "x0").replace("a3", "x13")
    )
    aliases = LOOP.replace("(a5)", "(x15)").replace("(a4)", "(x14)")
    assert ordered(aliases) == transform(
        EXPECTED.replace("(a5)", "(x15)").replace("(a4)", "(x14)")
    )
    commented = LOOP.replace(COUNTER, "# next column\n\n" + COUNTER)
    assert ordered(commented) == transform(
        EXPECTED.replace("\tc.nop\n", "# next column\n\n\tc.nop\n")
    )

    # No other arithmetic, vector shape, masked form, or register dataflow
    # is covered by the board measurement.
    for source in (
        LOOP.replace(CONFIGURATION, ""),
        LOOP.replace("16, e32", "8, e32"),
        LOOP.replace("e32, m4", "e64, m4"),
        LOOP.replace("e32, m4", "e32, m2"),
        LOOP.replace("ta, ma", "tu, ma"),
        LOOP.replace("vsetivli zero", "vsetivli a0"),
        LOOP.replace(ADD, "vfadd.vv v8, v8, v12\n"),
        LOOP.replace(ADD, "vsub.vx v8, v8, a0\n"),
        LOOP.replace(ADD, "vmul.vv v8, v8, v12\n"),
        LOOP.replace(ADD, ADD.rstrip() + ", v0.t\n"),
        LOOP.replace(LEFT, "vle32.v v8, (a5), v0.t\n"),
        LOOP.replace(STORE, "vse32.v v8, (a5), v0.t\n"),
        LOOP.replace("vadd.vv v8, v8", "vadd.vv v16, v8"),
        LOOP.replace("vadd.vv v8, v8, v12", "vadd.vv v8, v12, v8"),
        LOOP.replace("vse32.v v8", "vse32.v v12"),
        LOOP.replace("v12", "v10"),
        LOOP.replace("v12", "v8"),
        LOOP.replace(COUNTER, "addi a3, a2, 16\n"),
        LOOP.replace(COUNTER, "addi x15, a5, 16\n"),
        LOOP.replace(COUNTER, "addi x14, a4, 16\n"),
        LOOP.replace(COUNTER, "addi x0, zero, 16\n"),
        LOOP.replace(COUNTER, "addi x2, sp, 16\n"),
        LOOP.replace(COUNTER, "addi a3, a3, 8\n"),
        LOOP.replace(COUNTER, "addi a3, a3, 4096\n"),
        LOOP.replace("vse32.v v8, (a5)", "vse32.v v8, (a6)"),
        LOOP.replace("addi a5, a5, 64", "addi a5, a5, 32"),
        LOOP.replace("addi a4, a4, 64", "addi a4, a4, 32"),
        LOOP.replace("bge a2, a3", "bge a2, a0"),
        LOOP.replace("bge a2, a3", "bge a3, a3"),
        LOOP.replace("bge a2, a3, loop", "bge a2, a3, other"),
        LOOP.replace("loop:\n", "loop: "),
        LOOP.replace("loop:\n", ""),
    ):
        unchanged(source)
    for barrier in (
        "other:\n",
        ".cfi_def_cfa_offset 32\n",
        ".loc 1 2 3\n",
        ".p2align 2\n",
        "fence rw, rw\n",
        "call helper\n",
        "beq a0, a1, other\n",
        "addi t0, t0, 1\n",
        "vsetivli zero, 16, e32, m4, ta, ma\n",
    ):
        for statement in (CONFIGURATION, LEFT, RIGHT, ADD, COUNTER, STORE):
            unchanged(LOOP.replace(statement, statement + barrier))

    # The executable-section parser and byte-copy handling remain independent.
    for header in (
        ".data\n",
        ".rodata\n",
        '.section .add_data,"aw",@progbits\n',
        ".data\n.text\n.previous\n",
    ):
        assert ordered(header + LOOP) == header + LOOP
    nested = '.text\n.pushsection .add_data,"aw",@progbits\n'
    assert ordered(nested + LOOP + ".popsection\n" + LOOP) == (
        nested + LOOP + ".popsection\n" + transform(EXPECTED)
    )
    byte_copy = "vle8.v v8, (a5)\naddi a4, a4, 64\nvse8.v v8, (a6)\n"
    unchanged(byte_copy)
    assert ordered(byte_copy, order_vector_copies=True) == transform(
        byte_copy, order_vector_copies=True
    )
    assert ordered(LOOP, order_vector_copies=True) == transform(EXPECTED)
    # CHECK: PASS NR i32 add ordering: exact loop, fences, aliases and barriers
    print("PASS NR i32 add ordering: exact loop, fences, aliases and barriers")


if __name__ == "__main__":
    main()
