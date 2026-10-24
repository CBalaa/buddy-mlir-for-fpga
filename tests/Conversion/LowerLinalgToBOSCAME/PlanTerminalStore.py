# RUN: %PYTHON %s %t
"""Check terminal forwarding, old outs and actual private window separation."""

import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def source(
    old_out=False, partial=False, extra=False, observer=False, noalias=True
):
    count = 32 if partial else 64
    target = f"memref<{count}xi32, strided<[1], offset: 16>>"
    old_init = (
        "linalg.fill ins(%seven : i32) outs(%c : memref<64xi32>)"
        if old_out
        else ""
    )
    add = "%previous" if old_out else "%right"
    prefix = (
        "%prefix = memref.subview %c[0] [32] [1] : memref<64xi32> to memref<32xi32, strided<[1]>>"
        if partial
        else ""
    )
    value = "%prefix" if partial else "%c"
    value_type = "memref<32xi32, strided<[1]>>" if partial else "memref<64xi32>"
    observation = ""
    if extra:
        observation = (
            "memref.copy %c, %audit : memref<64xi32> to memref<64xi32>"
        )
    if observer:
        observation = f"""
    %observed = memref.alloc() alignment = 64 : memref<64xi32>
    linalg.generic {{indexing_maps = [#id, #id], iterator_types = ["parallel"]}}
        ins(%dest : {target}) outs(%observed : memref<64xi32>) {{
      ^bb0(%v: i32, %unused: i32):
        %double = arith.addi %v, %v : i32
        linalg.yield %double : i32
    }}
    memref.copy %observed, %audit : memref<64xi32> to memref<64xi32>
    memref.dealloc %observed : memref<64xi32>
"""
    return f"""
#id = affine_map<(d0) -> (d0)>
module attributes {{contract.abi = {{args_aligned_bytes = 64 : i64,
  args_disjoint = {str(noalias).lower()}}}}} {{
  func.func @terminal(%a: memref<64xi32>, %b: memref<64xi32>,
      %out: memref<96xi32>, %audit: memref<64xi32>) {{
    %seven = arith.constant 7 : i32
    %c = memref.alloc() alignment = 64 : memref<64xi32>
    {old_init}
    linalg.generic {{indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}}
        ins(%a, %b : memref<64xi32>, memref<64xi32>) outs(%c : memref<64xi32>) {{
      ^bb0(%left: i32, %right: i32, %previous: i32):
        %sum = arith.addi %left, {add} : i32
        linalg.yield %sum : i32
    }}
    // The address-only view is deliberately defined after the producer.
    %dest = memref.subview %out[16] [{count}] [1] : memref<96xi32> to {target}
    {observation}
    {prefix}
    memref.copy {value}, %dest : {value_type} to {target}
    memref.dealloc %c : memref<64xi32>
    return
  }}
}}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def invoke(name, text, *passes):
        path = work / f"{name}.mlir"
        path.write_text(text)
        result = subprocess.run(
            [str(buddy_opt_path()), str(path), *passes],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (name, result.stderr)
        return result.stdout, result.stderr

    for name, options, dma, folded in (
        ("fresh", {}, 0, True),
        ("old_out", {"old_out": True}, 0, True),
        ("partial", {"partial": True}, 1, False),
        ("extra_consumer", {"extra": True}, 2, False),
        ("intervening_observation", {"observer": True}, 1, True),
        ("unknown_alias", {"noalias": False}, 1, False),
    ):
        planned, report = invoke(
            name,
            source(**options),
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-report=true",
            "--bosc-ame-plan-verify",
        )
        assert ('action = "folded-store"' in planned) == folded, name
        if folded:
            assert "safe-terminal-forwarding" in report, (name, report)
        if name == "fresh":
            assert "plan.buffer" not in planned
            assert planned.count("contract.task =") == 1
        if name == "old_out":
            assert 'level = "spm"' in planned
            assert "linalg.fill" in planned
            assert "plan.transfer" not in planned
            # Old C=7 is an explicit input, while the DDR output starts at 9.
            assert re.search(r"ins\([^)]*,[^)]*,[^)]* :", planned), planned
        emitted, _ = invoke(name + ".planned", planned, "--bosc-ame-emit")
        path = work / f"{name}.emitted.mlir"
        path.write_text(emitted)
        a_values = [0x7FFFFFF0 + i % 7 for i in range(64)]
        b_values = [i * 3 + 17 for i in range(64)]
        a = bytearray(struct.pack("<64i", *a_values))
        b = bytearray(struct.pack("<64i", *b_values))
        output = bytearray(struct.pack("<96i", *([9] * 96)))
        audit = bytearray(struct.pack("<64i", *([-1] * 64)))
        saved_a, saved_b = bytes(a), bytes(b)
        machine = Machine(path, "terminal")
        machine.launch(
            {"%arg0": a, "%arg1": b, "%arg2": output, "%arg3": audit}, {}
        )
        expected = [
            (left + (7 if options.get("old_out") else right) + (1 << 31))
            % (1 << 32)
            - (1 << 31)
            for left, right in zip(a_values, b_values)
        ]
        count = 32 if options.get("partial") else 64
        assert struct.unpack("<96i", output) == tuple(
            [9] * 16 + expected[:count] + [9] * (80 - count)
        ), name
        assert (bytes(a), bytes(b)) == (saved_a, saved_b), name
        if options.get("extra"):
            assert struct.unpack("<64i", audit) == tuple(expected), name
        elif options.get("observer"):
            assert struct.unpack("<64i", audit) == (18,) * 64, name
        else:
            assert struct.unpack("<64i", audit) == (-1,) * 64, name
        assert len(machine.dma_log) == dma, (name, machine.dma_log)

    nested = """
module {
  func.func @sequential_regions() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i32
    scf.for %i = %c0 to %c2 step %c1 {
      %a = memref.alloc() alignment = 64 : memref<64xi32>
      linalg.fill ins(%zero : i32) outs(%a : memref<64xi32>)
      memref.dealloc %a : memref<64xi32>
    }
    %b = memref.alloc() alignment = 64 : memref<64xi32>
    linalg.fill ins(%zero : i32) outs(%b : memref<64xi32>)
    memref.dealloc %b : memref<64xi32>
    return
  }
}
"""
    planned, _ = invoke(
        "sequential_regions",
        nested,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
        "--bosc-ame-plan-verify",
    )
    windows = re.findall(r"plan.buffer = \{([^}]*)\}", planned)
    assert len(windows) == 2
    assert windows[0] != windows[1], planned

    # An extracted address escapes through the return value. The caller must
    # still see the producer's private result through that address.
    escaped = source().replace(
        "%audit: memref<64xi32>) {", "%audit: memref<64xi32>) -> index {"
    )
    escaped = escaped.replace(
        "    linalg.generic",
        "    %address = memref.extract_aligned_pointer_as_index %c "
        ": memref<64xi32> -> index\n"
        "    linalg.generic",
        1,
    ).replace("    memref.dealloc %c : memref<64xi32>", "")
    escaped = escaped.replace("    return", "    return %address : index")
    escaped_path = work / "escaped_private.mlir"
    escaped_path.write_text(escaped)
    result = subprocess.run(
        [
            str(buddy_opt_path()),
            str(escaped_path),
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "non-escaping" in result.stderr, result.stderr
    print(
        "PASS terminal DDR writes, preserved old outs, observations, guards and physical windows"
    )


if __name__ == "__main__":
    main()
