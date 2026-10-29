# RUN: %PYTHON %s %t
"""V2 uses bounded DDR frames unless a verified optimization removes them."""

import importlib.util
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def source(shape, serial=False):
    rank = len(shape)
    typ = "memref<" + "x".join(map(str, shape)) + "xi32>"
    dims = ",".join("d" + str(i) for i in range(rank))
    mapping = "affine_map<(" + dims + ")->(" + dims + ")>"
    iterators = ",".join('"parallel"' for _ in shape)
    body = f"""
    %tile=memref.alloc() : {typ}
    %bias=memref.alloca() : {typ}
    linalg.fill ins(%bias_value:i32) outs(%bias:{typ})
    memref.copy %arg0,%tile:{typ} to {typ}
    linalg.generic {{indexing_maps=[#id,#id,#id],iterator_types=[{iterators}]}}
      ins(%tile,%bias:{typ},{typ}) outs(%tile:{typ}) {{
      ^bb0(%value:i32,%addend:i32,%old:i32):
        %sum=arith.addi %value,%addend:i32
        linalg.yield %sum:i32
    }}
    memref.copy %tile,%arg1:{typ} to {typ}
    memref.dealloc %tile:{typ}
"""
    if serial:
        body = (
            """
    %c0=arith.constant 0:index
    %c1=arith.constant 1:index
    %c3=arith.constant 3:index
    scf.for %i=%c0 to %c3 step %c1 {
      %bias_value=arith.index_cast %i:index to i32
"""
            + body
            + "\n    }\n"
        )
    else:
        body = "\n    %bias_value=arith.constant 7:i32\n" + body
    return f"""#id={mapping}
module attributes {{contract.abi={{args_aligned_bytes=64:i64,args_disjoint=true}}}} {{
  func.func @ddr(%arg0:{typ},%arg1:{typ}) {{
{body}
    return
  }}
}}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def invoke(name, text, *passes):
        path = work / (name + ".mlir")
        path.write_text(text)
        result = subprocess.run(
            [str(buddy_opt_path()), str(path), *passes],
            text=True,
            capture_output=True,
        )
        assert (
            "LLVM ERROR" not in result.stderr
            and "PLEASE submit" not in result.stderr
        )
        return result

    def rejected(name, text, diagnostic, *passes):
        result = invoke(name, text, *passes)
        assert result.returncode != 0, (name, result.stdout)
        assert diagnostic in result.stderr, (name, result.stderr)
        return result

    plans = {}
    for name, shape, serial in [
        ("rank2", (3, 5), False),
        ("rank3", (2, 3, 5), False),
        ("serial", (3, 5), True),
    ]:
        original = source(shape, serial)
        planned = invoke(
            name + ".source",
            original,
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
            "--bosc-ame-plan-verify",
        )
        assert planned.returncode == 0, planned.stderr
        assert re.search(r"plan\.version = 2 : i(?:32|64)", planned.stdout)
        assert planned.stdout.count('level = "ddr"') == 2, planned.stdout
        assert planned.stdout.count('reason = "reuse-benefit-unproven"') == 2
        assert all(
            f'kind = "{kind}"' in planned.stdout
            for kind in ("copy", "fill", "pointwise")
        )
        assert not any(
            token in planned.stdout
            for token in ('level = "spm"', 'level = "virtual"', "plan.stream")
        )
        plans[name] = planned.stdout
        emitted = invoke(name + ".planned", planned.stdout, "--bosc-ame-emit")
        assert emitted.returncode == 0, emitted.stderr
        assert emitted.stdout.count("memref.alloca") == 2
        assert not any(
            token in emitted.stdout
            for token in (
                "memref.alloc()",
                "linalg.",
                "bosc_ame.spm.view",
                "bosc_ame.dma.start",
            )
        )
        first_loop = emitted.stdout.index("scf.for")
        assert emitted.stdout.rindex("memref.alloca") < first_loop
        count = 1
        for dim in shape:
            count *= dim
        values = [((i * 71) % 257) - 128 for i in range(count)]
        guard_in, guard_out = bytes([0xA1]) * 64, bytes([0xC3]) * 64
        inputs = bytearray(struct.pack("<" + "i" * count, *values) + guard_in)
        outputs = bytearray(bytes([0xDD]) * (count * 4) + guard_out)
        original_input = bytes(inputs)
        (work / (name + ".emitted.mlir")).write_text(emitted.stdout)
        machine = Machine(work / (name + ".emitted.mlir"), "ddr")
        machine.launch({"%arg0": inputs, "%arg1": outputs})
        addition = 2 if serial else 7
        expected = (
            struct.pack("<" + "i" * count, *[v + addition for v in values])
            + guard_out
        )
        assert bytes(inputs) == original_input and bytes(outputs) == expected
        assert not machine.dma_log
        assert all(
            0 <= begin < end <= count * 4
            for _, begin, end, _ in machine.accesses
        )

    # The compatibility adapter carries a fixed physical window into V2.
    spec = importlib.util.spec_from_file_location(
        "physical_adapter", Path(__file__).with_name("physical-adapter.py")
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    fixed = invoke(
        "fixed",
        adapter.SOURCE,
        "--linalg-to-boscame=buffer-placement=false plan-report=true",
    )
    assert fixed.returncode == 0, fixed.stderr
    assert 'role = "fixed-input"' in fixed.stderr
    assert "offset = 4160 : i64" in fixed.stdout
    assert fixed.stdout.count("bosc_ame.spm.view") == 1
    fixed_path = work / "fixed.emitted.mlir"
    fixed_path.write_text(fixed.stdout)
    payload = struct.pack("<32i", *range(-16, 16))
    guard = bytes([0xA7]) * 64
    incoming, outgoing = (
        bytearray(payload + guard),
        bytearray(bytes(128) + guard),
    )
    machine = Machine(fixed_path, "copy")
    machine.spm[1][:] = bytes([0x5A]) * len(machine.spm[1])
    machine.launch({"%arg0": incoming, "%arg1": outgoing})
    assert incoming == payload + guard and outgoing == payload + guard
    assert machine.spm[1][4096:4160] == bytes([0x5A]) * 64
    assert machine.spm[1][4288:4352] == bytes([0x5A]) * 64

    # A parsed execution recipe cannot secretly materialize its virtual roots.
    virtual = invoke(
        "virtual.source",
        source((16,)),
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
    )
    assert virtual.returncode == 0 and "plan.stream_virtual" in virtual.stdout
    forged, count = re.subn(
        r'plan\.buffer = \{capacity = 64 : i64, level = "virtual", role = "logical-value"\}',
        'plan.buffer = {bank = 0 : i64, capacity = 64 : i64, level = "spm", offset = 0 : i64, role = "fixed-input"}',
        virtual.stdout,
        count=1,
    )
    assert count == 1
    for entry in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        rejected(
            "forged_spm." + entry[2:],
            forged,
            "stream virtual buffer lacks a complete non-escaping content-version proof",
            entry,
        )

    orphan, count = re.subn(
        r"plan\.cpu = \{",
        "plan.register_member = {schedule = 999 : i64}, plan.cpu = {",
        plans["rank2"],
        count=1,
    )
    assert count == 1
    for entry in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        rejected(
            "orphan_member." + entry[2:],
            orphan,
            "orphaned register schedule decision",
            entry,
        )

    # Hoisting one private frame slot cannot replace concurrent allocations,
    # or shared accesses whose enclosing scope has no serial ownership proof.
    scope_header = """module {
  func.func @scope() {
    %c0=arith.constant 0:index
    %c1=arith.constant 1:index
    %c2=arith.constant 2:index
    %zero=arith.constant 0:i32
"""
    allocation = """    %a=memref.alloca():memref<2x2xi32>
    linalg.fill ins(%zero:i32) outs(%a:memref<2x2xi32>)
"""
    access = """      %v=memref.load %a[%c0,%c0]:memref<2x2xi32>
      memref.store %v,%a[%c0,%c0]:memref<2x2xi32>
"""
    cases = {
        "parallel_allocated": scope_header
        + "    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {\n"
        + allocation
        + "      scf.reduce\n    }\n    return\n  }\n}\n",
        "parallel_captured": scope_header
        + allocation
        + "    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {\n"
        + access
        + "      scf.reduce\n    }\n    return\n  }\n}\n",
        "unknown_scope_captured": scope_header
        + allocation
        + "    scf.execute_region {\n"
        + access
        + "      scf.yield\n    }\n    return\n  }\n}\n",
    }
    for name, text in cases.items():
        result = rejected(
            name,
            text,
            "private DDR frame slot requires serial",
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
        )
        expected_scope = (
            "scf.execute_region"
            if name.startswith("unknown")
            else "scf.parallel"
        )
        assert "unsupported ancestor " + expected_scope in result.stderr

    print(
        "PASS default V2 DDR frames, exact values/guards, serial hoisting, fixed SPM, forged ownership and concurrent-scope rejection"
    )


if __name__ == "__main__":
    main()
