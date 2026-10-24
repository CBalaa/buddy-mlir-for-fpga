# RUN: %PYTHON %s
"""Exercise conservative stack CFG traversal and its failure conditions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
)
from check_e6_stack import AuditError, Auditor, Function  # noqa: E402


def auditor(blocks, indirect=None):
    functions, instructions = {}, {}
    for name, base, operations in blocks:
        functions[name] = Function(name, base, 4 * len(operations))
        for i, text in enumerate(operations):
            address = base + 4 * i
            instructions[address] = {
                "address": address,
                "instruction": text,
                "mnemonic": text.split()[0],
                "size": 4,
            }
    return Auditor(functions, instructions, 0x1000, indirect or {})


class StackAuditTest(unittest.TestCase):
    def test_conditional_call_and_transient_save(self):
        check = auditor(
            [
                (
                    "entry",
                    0x100,
                    [
                        "addi sp, sp, -32",
                        "beqz a0, 0x10c",
                        "jal 0x200 <leaf>",
                        "addi sp, sp, 32",
                        "ret",
                    ],
                ),
                (
                    "leaf",
                    0x200,
                    [
                        "addi sp, sp, -64",
                        "addi sp, sp, -128",
                        "addi sp, sp, 128",
                        "addi sp, sp, 64",
                        "ret",
                    ],
                ),
            ]
        )
        result = check.run("entry", 0x1000)
        self.assertEqual(result["peak"], 224)
        self.assertEqual(result["path"], ["entry", "leaf"])

    def test_aligned_allocas_frame_restore_and_nested_call(self):
        check = auditor(
            [
                (
                    "entry",
                    0x100,
                    [
                        "addi sp, sp, -64",
                        "addi s0, sp, 64",
                        "andi sp, sp, -64",
                        "addi a0, sp, -16",
                        "andi a0, a0, -64",
                        "mv sp, a0",
                        "addi a1, sp, -16",
                        "andi a2, a1, -64",
                        "mv sp, a2",
                        "jal 0x200 <wait>",
                        "addi sp, s0, -64",
                        "addi sp, sp, 64",
                        "ret",
                    ],
                ),
                ("wait", 0x200, ["addi sp, sp, -16", "addi sp, sp, 16", "ret"]),
            ]
        )
        self.assertEqual(check.run("entry", 0x1000)["peak"], 208)

    def test_tail_call_does_not_add_restored_frame(self):
        check = auditor(
            [
                (
                    "entry",
                    0x100,
                    ["addi sp, sp, -32", "addi sp, sp, 32", "j 0x200 <leaf>"],
                ),
                ("leaf", 0x200, ["addi sp, sp, -64", "addi sp, sp, 64", "ret"]),
            ]
        )
        self.assertEqual(check.run("entry", 0x1000)["peak"], 64)

    def test_dma_failure_hooks_are_both_audited(self):
        blocks = [
            ("wait", 0x100, ["addi sp, sp, -16", "jalr a4"]),
            ("installed", 0x200, ["addi sp, sp, -64", "unimp"]),
            ("default", 0x300, ["unimp"]),
        ]
        with self.assertRaisesRegex(AuditError, "unmapped indirect"):
            auditor(blocks).run("wait", 0x1000)
        check = auditor(blocks, {"wait": ["installed", "default"]})
        result = check.run("wait", 0x1000)
        self.assertEqual(result["peak"], 80)
        self.assertFalse(result["returns"])
        self.assertEqual(
            {call[2] for call in check.visited_calls}, {"installed", "default"}
        )

    def test_unknown_sp_and_unbounded_loop_fail_closed(self):
        for operations, message in [
            (["sub sp, sp, a0", "ret"], "unknown SP write"),
            (["addi sp, sp, -16", "j 0x100"], "inconsistent/unbounded SP"),
            (["addi sp, sp, -16", "ret"], "unrestored SP"),
        ]:
            with (
                self.subTest(operations=operations),
                self.assertRaisesRegex(AuditError, message),
            ):
                auditor([("entry", 0x100, operations)]).run("entry", 0x1000)

    def test_recursive_frame_growth_fails_closed(self):
        check = auditor(
            [("entry", 0x100, ["addi sp, sp, -16", "jal 0x100 <entry>", "ret"])]
        )
        with self.assertRaisesRegex(AuditError, "recursive call"):
            check.run("entry", 0x1000)


if __name__ == "__main__":
    unittest.main()
