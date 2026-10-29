# RUN: %PYTHON %s | FileCheck %s
"""Check the default ordered-sum fallback and execute its complete expansion."""

import random
import re
import struct
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from check_nr_elf import audit  # noqa: E402
from restrict_fpga_assembly import transform  # noqa: E402

MASK64 = (1 << 64) - 1
GPRS = {
    name: number
    for number, name in enumerate(
        [
            "zero",
            "ra",
            "sp",
            "gp",
            "tp",
            "t0",
            "t1",
            "t2",
            "s0",
            "s1",
            "a0",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
            "s8",
            "s9",
            "s10",
            "s11",
            "t3",
            "t4",
            "t5",
            "t6",
        ]
    )
}
FPRS = {
    name: number
    for number, name in enumerate(
        [
            "ft0",
            "ft1",
            "ft2",
            "ft3",
            "ft4",
            "ft5",
            "ft6",
            "ft7",
            "fs0",
            "fs1",
            "fa0",
            "fa1",
            "fa2",
            "fa3",
            "fa4",
            "fa5",
            "fa6",
            "fa7",
            "fs2",
            "fs3",
            "fs4",
            "fs5",
            "fs6",
            "fs7",
            "fs8",
            "fs9",
            "fs10",
            "fs11",
            "ft8",
            "ft9",
            "ft10",
            "ft11",
        ]
    )
}


def rejected(source, **options):
    try:
        transform(source, **options)
    except ValueError:
        return
    raise AssertionError("unproved ordered reduction accepted: " + source)


def fadd32(left, right, mode):
    """Integer IEEE binary32 oracle; return canonical RISC-V bits and flags."""
    parts = []
    invalid = False
    for bits in (left, right):
        sign, exponent, fraction = (
            bits >> 31,
            (bits >> 23) & 255,
            bits & 0x7FFFFF,
        )
        parts.append((sign, exponent, fraction))
        invalid |= exponent == 255 and 0 < fraction < 0x400000
    if any(exponent == 255 and fraction for _, exponent, fraction in parts):
        return 0x7FC00000, 16 if invalid else 0
    if all(exponent == 255 for _, exponent, _ in parts):
        if parts[0][0] != parts[1][0]:
            return 0x7FC00000, 16
        return left, 0
    for bits, (_, exponent, _) in zip((left, right), parts):
        if exponent == 255:
            return bits, 0
    integers, powers = [], []
    for sign, exponent, fraction in parts:
        integers.append(
            (-1 if sign else 1) * (fraction | (0x800000 if exponent else 0))
        )
        powers.append(exponent - 150 if exponent else -149)
    power = min(powers)
    exact = sum(
        value << (scale - power) for value, scale in zip(integers, powers)
    )
    if not exact:
        # Same-signed zeros keep their sign; cancellation rounds to -0 only
        # under RDN. This catches erroneously skipped additions of +0.
        sign = (
            parts[0][0] if left == right == (left & 0x80000000) else mode == 2
        )
        return int(sign) << 31, 0
    sign = exact < 0
    magnitude = abs(exact)
    target = max(magnitude.bit_length() - 1 + power - 23, -149)
    shift = target - power
    if shift > 0:
        rounded, remainder = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        increment = (
            (
                mode == 0
                and (
                    remainder > halfway
                    or (remainder == halfway and rounded & 1)
                )
            )
            or (mode == 2 and sign and remainder)
            or (mode == 3 and not sign and remainder)
            or (mode == 4 and remainder >= halfway)
        )
        rounded += bool(increment)
    else:
        rounded, remainder = magnitude << -shift, 0
    if rounded >= 1 << 24:
        rounded >>= 1
        target += 1
    exponent = target + 150 if rounded >= 1 << 23 else 0
    if exponent >= 255:
        infinity = (
            mode in (0, 4) or (mode == 2 and sign) or (mode == 3 and not sign)
        )
        return (int(sign) << 31) | (0x7F800000 if infinity else 0x7F7FFFFF), 5
    flags = int(bool(remainder)) | (2 if remainder and exponent == 0 else 0)
    return (int(sign) << 31) | (exponent << 23) | (rounded & 0x7FFFFF), flags


def statements(assembly):
    instructions, labels = [], {}
    for line in assembly.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"([\w.$]+):\s*(.*)", line)
        if match:
            assert match[1] not in labels, (
                "duplicate generated label: " + match[1]
            )
            labels[match[1]] = len(instructions)
            line = match[2]
        if line and not line.startswith("."):
            instructions.append(re.split(r"[\s,()]+", line))
    return instructions, labels


def reduction_elf():
    """An independently assembled vfredosum word in a minimal ELF64 image."""
    # llvm-mc -triple=riscv64 -mattr=+v,+f -show-encoding confirms this word.
    code = struct.pack("<I", 0x0EC81457)
    names = b"\0.text\0.shstrtab\0"
    section_offset = (64 + len(code) + len(names) + 7) & ~7
    data = bytearray(section_offset + 3 * 64)
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        data,
        0,
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        2,
        243,
        1,
        0x80000000,
        0,
        section_offset,
        0,
        64,
        0,
        0,
        64,
        3,
        2,
    )
    data[64:68] = code
    data[68 : 68 + len(names)] = names
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 64,
        1,
        1,
        6,
        0x80000000,
        64,
        4,
        0,
        0,
        4,
        0,
    )
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 128,
        7,
        3,
        0,
        0,
        68,
        len(names),
        0,
        0,
        1,
        0,
    )
    dump = (
        "Disassembly of section .text:\n"
        "0000000080000000 <kernel>:\n"
        "80000000: 0ec81457 vfredosum.vs v8, v12, v16\n"
    )
    return bytes(data), dump


def check_execution(assembly, avl, lmul, aliases, vlen, mode, policy, pattern):
    """Interpret actual emitted instructions, including spills, flags and VL."""
    instructions, labels = statements(assembly)
    rng = random.Random(712)
    gprs = [rng.getrandbits(64) for _ in range(32)]
    gprs[0], gprs[2] = 0, 0x80002000
    fprs = [rng.getrandbits(64) for _ in range(32)]
    words = vlen // 32
    vectors = [[rng.getrandbits(32) for _ in range(words)] for _ in range(32)]
    destination, source, seed = aliases
    for lane in range(lmul * words):
        vectors[source + lane // words][lane % words] = pattern[
            lane % len(pattern)
        ]
    vectors[seed][0] = pattern[-1]
    old_gprs, old_fprs = gprs.copy(), fprs.copy()
    old_vectors = [register.copy() for register in vectors]
    memory = {
        address: rng.randrange(256)
        for address in range(gprs[2] - 192, gprs[2] + 32)
    }
    old_memory = memory.copy()
    vlmax = words * lmul
    actual = min(avl, vlmax)
    if policy and vlmax < avl < 2 * vlmax:
        actual = (avl + 1) // 2
    expected_gprs = old_gprs.copy()
    if instructions[0][0] == "vsetivli" and instructions[0][1] != "zero":
        expected_gprs[GPRS[instructions[0][1]]] = actual
    expected, expected_flags = old_vectors[seed][0], 8
    if actual:
        for lane in range(actual):
            expected, flags = fadd32(
                expected,
                old_vectors[source + lane // words][lane % words],
                mode,
            )
            expected_flags |= flags
    flags, vl, pc, steps = 8, None, 0, 0
    frame_size = 32 + ((4 * avl + 15) // 16) * 16

    def integer(name):
        return gprs[GPRS[name]] if name in GPRS else int(name, 0)

    def set_integer(name, value):
        if name != "zero":
            gprs[GPRS[name]] = value & MASK64

    def read(address, size):
        return sum(
            memory[address + offset] << (8 * offset) for offset in range(size)
        )

    def write(address, value, size):
        assert gprs[2] <= address and address + size <= old_gprs[2], (
            "store outside temporary stack"
        )
        for offset in range(size):
            memory[address + offset] = (value >> (8 * offset)) & 255

    while pc < len(instructions):
        fields = instructions[pc]
        op, operands = fields[0], fields[1:]
        pc += 1
        steps += 1
        assert steps < 1024, "scalar reduction failed to terminate"
        if op == "vsetivli":
            assert operands[1:] == [
                str(avl),
                "e32",
                f"m{lmul}",
                operands[4],
                operands[5],
            ]
            assert operands[4] in ("ta", "tu") and operands[5] in ("ma", "mu")
            vl = actual
            set_integer(operands[0], vl)
        elif op in ("addi", "add", "sub", "slli"):
            dest, left, right = operands
            left, right = integer(left), integer(right)
            if op in ("addi", "add"):
                result = left + right
            elif op == "sub":
                result = left - right
            else:
                result = left << right
            set_integer(dest, result)
            assert gprs[2] % 16 == 0, "temporary stack is misaligned"
        elif op in ("mv", "li"):
            set_integer(operands[0], integer(operands[1]))
        elif op in ("sd", "fsd", "sw", "fsw"):
            register, offset, base = operands[:3]
            value = (
                fprs[FPRS[register]]
                if op.startswith("f")
                else integer(register)
            )
            write(
                integer(base) + int(offset), value, 8 if op.endswith("d") else 4
            )
        elif op in ("ld", "fld", "flw"):
            register, offset, base = operands[:3]
            value = read(integer(base) + int(offset), 4 if op == "flw" else 8)
            if op.startswith("f"):
                fprs[FPRS[register]] = value | (
                    0xFFFFFFFF00000000 if op == "flw" else 0
                )
            else:
                set_integer(register, value)
        elif op == "vse32.v":
            assert vl is not None
            vector, base = operands[:2]
            start = int(vector[1:])
            for lane in range(vl):
                write(
                    integer(base) + 4 * lane,
                    vectors[start + lane // words][lane % words],
                    4,
                )
        elif op == "vfmv.f.s":
            register, vector = operands
            fprs[FPRS[register]] = (
                vectors[int(vector[1:])][0] | 0xFFFFFFFF00000000
            )
        elif op == "vfmv.s.f":
            vector, register = operands
            if vl:
                vectors[int(vector[1:])][0] = fprs[FPRS[register]] & 0xFFFFFFFF
        elif op == "fadd.s":
            dest, left, right = operands[:3]
            assert len(operands) == 3 or operands[3] == "dyn"
            left, right = fprs[FPRS[left]], fprs[FPRS[right]]
            left = left & 0xFFFFFFFF if left >> 32 == 0xFFFFFFFF else 0x7FC00000
            right = (
                right & 0xFFFFFFFF if right >> 32 == 0xFFFFFFFF else 0x7FC00000
            )
            value, raised = fadd32(left, right, mode)
            fprs[FPRS[dest]] = value | 0xFFFFFFFF00000000
            flags |= raised
        elif op in ("beqz", "bnez", "beq", "bne", "j"):
            if op == "j":
                take, label = True, operands[0]
            elif op in ("beqz", "bnez"):
                take, label = (
                    (integer(operands[0]) == 0) == (op == "beqz"),
                    operands[1],
                )
            else:
                take, label = (
                    (integer(operands[0]) == integer(operands[1]))
                    == (op == "beq"),
                    operands[2],
                )
            if take:
                pc = labels[label]
        elif op == "ret":
            break
        elif op != "fence":
            raise AssertionError(
                "uninterpreted expansion instruction: " + str(fields)
            )
    assert vl == actual, "vector state changed"
    assert flags == expected_flags, (flags, expected_flags, mode, pattern)
    assert gprs == expected_gprs, "live GPR or stack pointer corrupted"
    assert fprs == old_fprs, "live floating register corrupted"
    if actual:
        old_vectors[destination][0] = expected
    assert vectors == old_vectors, (
        "wrong result, source overlap, or live vector corruption"
    )
    assert all(
        memory[address] == value
        for address, value in old_memory.items()
        if not old_gprs[2] - frame_size <= address < old_gprs[2]
    ), "caller stack or guard corrupted"


def main():
    image, disassembly = reduction_elf()
    report = audit(image, disassembly)
    assert report["status"] == "FAIL"
    assert any(
        "vector ordered reduction is disabled" in error
        for error in report["errors"]
    )
    assert (
        audit(image, disassembly, allow_vector_ordered_reductions=True)[
            "status"
        ]
        == "PASS"
    )
    for dump in ("", disassembly.replace("0ec81457", "00000013")):
        assert any(
            "vector ordered reduction is disabled" in error
            for error in audit(image, dump)["errors"]
        )
    for left, right, mode, value, flags in (
        (0x80000000, 0, 0, 0, 0),
        (0x80000000, 0, 2, 0x80000000, 0),
        (0x80000000, 0x80000000, 0, 0x80000000, 0),
        (0x3F800000, 0x33800000, 0, 0x3F800000, 1),
        (0x3F800000, 0x33800000, 4, 0x3F800001, 1),
        (0x3F800000, 0x33800000, 3, 0x3F800001, 1),
        (0x7F7FFFFF, 0x7F7FFFFF, 1, 0x7F7FFFFF, 5),
        (0xFF7FFFFF, 0xFF7FFFFF, 2, 0xFF800000, 5),
        (0x7F801234, 0, 0, 0x7FC00000, 16),
        (0x7FC01234, 0, 0, 0x7FC00000, 0),
        (0x7F800000, 0xFF800000, 0, 0x7FC00000, 16),
    ):
        assert fadd32(left, right, mode) == (value, flags)
    prefix = ".text\n.Lentry:\nvsetivli zero, 16, e32, m4, ta, ma\n"
    reduction = "vfredosum.vs v8, v12, v16\n"
    valid = prefix + reduction + "ret\n"
    expanded = transform(valid)
    assert "vfredosum.vs" not in expanded and "fadd.s" in expanded
    assert not re.search(
        r"\b(?:vlenb|vl|vtype|vs[1248]r\.v|vl[1248]re\d+\.v)\b", expanded
    )
    assert transform(valid, scalarize_f32_ordered_reductions=False) == valid
    # Production builds must not fall back to the generic vector/FPR
    # roundtrip: its ISA semantics pass here, but FPGA execution is faulty.
    try:
        transform(valid, reject_vector_ordered_reductions=True)
    except ValueError as error:
        assert "scalar SSA lowering" in str(error)
    else:
        raise AssertionError("production policy accepted a residual reduction")
    for options in ({}, {"coalesce_fences": True}, {"insert_fences": False}):
        output = transform(valid, **options)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        stores = [
            index
            for index, line in enumerate(lines)
            if line.startswith("vse32.v")
        ]
        assert len(stores) == 1
        if options.get("insert_fences", True):
            assert (
                lines[stores[0] - 1] == lines[stores[0] + 1] == "fence\trw, rw"
            )
        else:
            assert not any(line.startswith("fence") for line in lines)
    # Generated labels must also avoid definitions appearing after the use.
    generated = set(re.findall(r"^([\w.$]+):", expanded, re.M)) - {".Lentry"}
    assert generated
    collision = valid + "".join(label + ":\n" for label in sorted(generated))
    statements(transform(collision))
    data = '.section .rodata\n.ascii "vfredosum.vs v8, v12, v16"\n'
    assert transform(data) == data
    for invalid in (
        reduction,
        valid.replace("e32", "e64"),
        valid.replace("m4", "mf2"),
        valid.replace("16, e32", "32, e32"),
        valid.replace("16, e32", "-1, e32"),
        valid.replace("vsetivli zero, 16", "vsetvli zero, a0"),
        valid.replace(reduction, reduction.rstrip() + ", v0.t\n"),
        valid.replace("v8, v12", "v32, v12"),
        valid.replace("v8, v12", "v8, v14"),
        valid.replace("v16\n", "v32\n"),
        prefix + "ret\n.Ldead:\n" + reduction,
        prefix + ".globl kernel\n.type kernel,@function\nkernel:\n" + reduction,
    ):
        rejected(invalid)
    cfg = (
        prefix
        + """beqz a0, .Lright
.Lleft:
j .Lmerge
.Lright:
j .Lmerge
.Lmerge:
"""
        + reduction
        + "ret\n"
    )
    assert "vfredosum.vs" not in transform(cfg)
    for change in (
        "8, e32, m4, ta, ma",
        "16, e32, m2, ta, ma",
        "16, e32, m4, tu, ma",
        "16, e32, m4, ta, mu",
        "16, e64, m4, ta, ma",
    ):
        rejected(
            cfg.replace(
                ".Lright:\nj", ".Lright:\nvsetivli zero, " + change + "\nj"
            )
        )
    for opaque in (
        "call helper",
        "tail helper",
        "jalr a0",
        "jr a0",
        "c.jr a0",
        "mret",
        "ecall",
        "c.ebreak",
        "cm.jt a0",
        ".word 0x0000006f",
        "vsetvl zero, a0, a1",
        "csrr a0, vl",
        "csrw vtype, a0",
        "csrw vstart, a0",
    ):
        rejected(cfg.replace(".Lright:\nj", ".Lright:\n" + opaque + "\nj"))
    loop = prefix + ".Lloop:\n" + reduction + "bnez a0, .Lloop\nret\n"
    assert "vfredosum.vs" not in transform(loop)
    rejected(loop.replace("bnez", "vsetivli zero, 8, e32, m4, ta, ma\nbnez"))
    # Bypass is explicit and does not need a configuration proof. Other
    # supported reductions retain their existing lowering policy.
    assert (
        transform(reduction, scalarize_f32_ordered_reductions=False)
        == reduction
    )
    for mnemonic in ("vfredusum.vs", "vfredmax.vs"):
        assert mnemonic in transform(valid.replace("vfredosum.vs", mnemonic))
    for args in ([], ["--allow-vector-ordered-reductions"]):
        cli = subprocess.run(
            [sys.executable, str(TOOLS / "restrict_fpga_assembly.py"), *args],
            input=valid,
            capture_output=True,
            text=True,
            check=True,
        )
        assert cli.stdout == (valid if args else expanded)
    strict_cli = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "restrict_fpga_assembly.py"),
            "--reject-vector-ordered-reductions",
        ],
        input=valid,
        capture_output=True,
        text=True,
    )
    assert strict_cli.returncode == 1
    assert not strict_cli.stdout and "scalar SSA lowering" in strict_cli.stderr
    assert "fadd.s" in transform(
        expanded, reject_vector_ordered_reductions=True
    )
    patterns = (
        [0x3F800000, 0x40000000, 0x40400000, 0x40800000],
        [0x60AD78EC, 0x3F800000, 0xE0AD78EC, 0x3F800000],
        [0, 0x80000000, 0, 0x80000000],
        [1, 0x80000001, 0x007FFFFF, 0x00800000],
        [0x7F7FFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000],
        [0x7F800000, 0xFF800000, 0x3F800000, 0],
        [0x7FC01234, 0x7F801234, 0xFFC01234, 0x3F800000],
        [0x3F800000, 0x33800000, 0xB3800000, 0x3F800001],
    )
    cases = 0
    for lmul in (1, 2, 4, 8):
        for avl in (0, 1, 16, 31):
            for tail in ("ta", "tu"):
                for aliases in (
                    (8, 16, 24),
                    (16, 16, 24),
                    (24, 16, 24),
                    (17, 16, 17),
                    (0, 0, 0),
                ):
                    destination, source, seed = aliases
                    text = f".text\n.Lentry:\nvsetivli zero, {avl}, e32, m{lmul}, {tail}, mu\nvfredosum.vs v{destination}, v{source}, v{seed}\nret\n"
                    output = transform(text)
                    for index, pattern in enumerate(patterns):
                        for mode in range(5):
                            check_execution(
                                output,
                                avl,
                                lmul,
                                aliases,
                                (128, 256, 512)[index % 3],
                                mode,
                                index % 2,
                                pattern,
                            )
                            cases += 1
    assert cases == 6400
    for register in ("t5", "t6", "a0"):
        check_execution(
            transform(valid.replace("vsetivli zero", "vsetivli " + register)),
            16,
            4,
            (8, 12, 16),
            128,
            0,
            False,
            patterns[0],
        )
    # CHECK: PASS FPGA scalar ordered reductions: CFG, aliases, IEEE flags, VL and live state
    print(
        "PASS FPGA scalar ordered reductions: CFG, aliases, IEEE flags, VL and live state"
    )


if __name__ == "__main__":
    main()
