# RUN: %PYTHON %s %t
"""Check deferred zero initialization through planning, emission and execution.

The production planner must prove that the private C has no observable zero
contents before its first matmul. Every tile is then cleared on the first K
iteration and reloaded on subsequent iterations, with no register lifetime
across DMA. Repeated launches retain dirty SPM and must still overwrite C.
"""

import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine, op_name  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402
from PlanBoundaryTransfers import (  # noqa: E402
    dictionaries,
    remove_field,
    replace_field,
    string,
)


class TracedMachine(Machine):
    def __init__(self, path):
        super().__init__(path, "product")
        self.trace = []

    def run(self, op):
        operation = op_name(op.text)
        for name in (
            "bosc_ame.tile.zero_acc",
            "bosc_ame.tile.load_acc",
            "bosc_ame.tile.mma",
            "bosc_ame.tile.store_acc",
            "bosc_ame.dma.start",
            "bosc_ame.dma.wait",
        ):
            if name == operation:
                self.trace.append(name)
                break
        return super().run(op)


def source(
    *,
    blocks=1,
    loop=False,
    rows=32,
    columns=64,
    initial=0,
    observe="",
    carried_ab=False,
    start=0,
):
    inner = max(1, blocks + start) * 64
    private_columns = max(64, columns)
    a = f"memref<{rows}x{inner}xi8>"
    b = f"memref<{columns}x{inner}xi8>"
    c = f"memref<{rows}x{columns}xi32>"
    ap = "memref<32x64xi8>"
    bp = f"memref<{private_columns}x64xi8>"
    bt = f"memref<64x{private_columns}xi8, strided<[1, 64]>>"
    cp = f"memref<32x{private_columns}xi32>"
    av = f"memref<{rows}x64xi8, strided<[64, 1]>>"
    bv = f"memref<{columns}x64xi8, strided<[64, 1]>>"
    cv = f"memref<{rows}x{columns}xi32, strided<[{private_columns}, 1]>>"
    observed_argument = f", %observed: {c}" if observe else ""
    lines = [
        "module attributes {",
        "  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}",
        "} {",
        f"  func.func @product(%ain: memref<*xi8>, %bin: memref<*xi8>, %out: {c}{observed_argument}) {{",
        f"    %a = memref.reinterpret_cast %ain to offset: [0], sizes: [{rows}, {inner}], strides: [{inner}, 1] : memref<*xi8> to {a}",
        f"    %b = memref.reinterpret_cast %bin to offset: [0], sizes: [{columns}, {inner}], strides: [{inner}, 1] : memref<*xi8> to {b}",
        "    %zero = arith.constant 0 : i8",
        f"    %initial = arith.constant {initial} : i32",
        f"    %lb = arith.constant {start * 64} : index",
        f"    %ub = arith.constant {(blocks + start) * 64} : index",
        "    %step = arith.constant 64 : index",
        f"    %cc = memref.alloc() alignment = 64 : {cp}",
        f"    linalg.fill ins(%initial : i32) outs(%cc : {cp})",
        f"    %cp = memref.subview %cc[0, 0] [{rows}, {columns}] [1, 1] : {cp} to {cv}",
    ]
    inputs = [
        f"    %aa = memref.alloc() alignment = 64 : {ap}",
        f"    %bb = memref.alloc() alignment = 64 : {bp}",
        f"    linalg.fill ins(%zero : i8) outs(%aa : {ap})",
        f"    linalg.fill ins(%zero : i8) outs(%bb : {bp})",
        f"    %ap = memref.subview %aa[0, 0] [{rows}, 64] [1, 1] : {ap} to {av}",
        f"    %bp = memref.subview %bb[0, 0] [{columns}, 64] [1, 1] : {bp} to {bv}",
        f"    %bt = memref.transpose %bb (d0, d1) -> (d1, d0) : {bp} to {bt}",
    ]
    if observe == "whole":
        assert rows == 32 and columns == 64
        lines.append(f"    memref.copy %cc, %observed : {cp} to {c}")
    elif observe == "alias":
        lines.append(f"    memref.copy %cp, %observed : {cv} to {c}")
    if carried_ab:
        lines.extend(inputs)
    if loop:
        lines.append("    scf.for %k = %lb to %ub step %step {")
        offset, suffix = "%k", "?"
    else:
        assert blocks == 1
        offset, suffix = str(start * 64), str(start * 64)
    if not carried_ab:
        lines.extend(inputs)
    ae = f"memref<{rows}x64xi8, strided<[{inner}, 1], offset: {suffix}>>"
    be = f"memref<{columns}x64xi8, strided<[{inner}, 1], offset: {suffix}>>"
    lines.extend(
        [
            f"      %ae = memref.subview %a[0, {offset}] [{rows}, 64] [1, 1] : {a} to {ae}",
            f"      %be = memref.subview %b[0, {offset}] [{columns}, 64] [1, 1] : {b} to {be}",
            f"      memref.copy %ae, %ap : {ae} to {av}",
            f"      memref.copy %be, %bp : {be} to {bv}",
            "      linalg.matmul {cast = #linalg.type_fn<cast_signed>}",
            f"          ins(%aa, %bt : {ap}, {bt})",
            f"          outs(%cc : {cp})",
        ]
    )
    if not carried_ab:
        lines.extend(
            [f"    memref.dealloc %aa : {ap}", f"    memref.dealloc %bb : {bp}"]
        )
    if loop:
        lines.append("    }")
    if carried_ab:
        lines.extend(
            [f"    memref.dealloc %aa : {ap}", f"    memref.dealloc %bb : {bp}"]
        )
    lines.extend(
        [
            f"    memref.copy %cp, %out : {cv} to {c}",
            f"    memref.dealloc %cc : {cp}",
            "    return",
            "  }",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def signed(value):
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(text, *passes):
        result = subprocess.run(
            [opt, *passes], input=text, text=True, capture_output=True
        )
        assert "LLVM ERROR" not in result.stderr, result.stderr
        assert "PLEASE submit" not in result.stderr, result.stderr
        return result

    def compile_case(name, text, initialization):
        (work / f"{name}.mlir").write_text(text)
        result = invoke(
            text,
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-report=true",
            "--bosc-ame-plan-verify",
        )
        assert result.returncode == 0, (name, result.stderr)
        planned = result.stdout
        (work / f"{name}.planned.mlir").write_text(planned)
        initializations = dictionaries(planned, "initialization =")
        deferred = [
            entry
            for entry in dictionaries(planned, "plan.fill =")
            if 'kind = "deferred-zero"' in entry[2]
        ]
        if initialization:
            assert len(initializations) == len(deferred) == 1, (name, planned)
            assert string(initializations[0][2], "kind") == initialization
            assert string(deferred[0][2], "engine") == "ame"
        else:
            assert not initializations and not deferred, (name, planned)
        emitted = invoke(planned, "--bosc-ame-emit")
        assert emitted.returncode == 0, (name, emitted.stderr)
        assert "plan." not in emitted.stdout and "linalg." not in emitted.stdout
        target = invoke(
            emitted.stdout, "--bosc-ame-spm-dma-check", "--lower-bosc-ame"
        )
        assert target.returncode == 0, (name, target.stderr)
        path = work / f"{name}.emitted.mlir"
        path.write_text(emitted.stdout)
        if initialization == "zero":
            assert "bosc_ame.tile.zero_acc" in emitted.stdout
            assert "bosc_ame.tile.load_acc" not in emitted.stdout
        elif initialization:
            assert "scf.if" in emitted.stdout
            assert "bosc_ame.tile.zero_acc" in emitted.stdout
            assert "bosc_ame.tile.load_acc" in emitted.stdout
        else:
            assert "bosc_ame.tile.zero_acc" not in emitted.stdout
        return planned, path

    def simulate(
        name,
        path,
        *,
        blocks=1,
        rows=32,
        columns=64,
        initial=0,
        observe="",
        start=0,
    ):
        inner = max(1, blocks + start) * 64
        extremes = (-128, 127, -1, 1, 0, -127, 126)
        left = [
            extremes[(row * 3 + k) % len(extremes)]
            for row in range(rows)
            for k in range(inner)
        ]
        right = [
            extremes[(column + k * 3) % len(extremes)]
            for column in range(columns)
            for k in range(inner)
        ]
        a = bytes(value & 255 for value in left)
        b = bytes(value & 255 for value in right)
        expected = bytearray()
        for row in range(rows):
            for column in range(columns):
                value = initial + sum(
                    left[row * inner + k] * right[column * inner + k]
                    for k in range(start * 64, (blocks + start) * 64)
                )
                expected.extend(struct.pack("<i", signed(value)))
        guard = bytes([0xD7]) * 64
        machine = TracedMachine(path)
        machine.spm[0][:] = bytes([0x86]) * len(machine.spm[0])
        machine.spm[1][:] = bytes([0x39]) * len(machine.spm[1])
        for repeat in range(2):
            buffers = {
                "%arg0": bytearray(a + guard),
                "%arg1": bytearray(b + guard),
                "%arg2": bytearray(
                    bytes([0xA6 + repeat]) * len(expected) + guard
                ),
            }
            if observe:
                buffers["%arg3"] = bytearray(
                    bytes([0xCE]) * len(expected) + guard
                )
            access_begin, trace_begin = (
                len(machine.accesses),
                len(machine.trace),
            )
            machine.launch(buffers)
            assert buffers["%arg0"] == a + guard, name
            assert buffers["%arg1"] == b + guard, name
            assert buffers["%arg2"] == expected + guard, (name, repeat)
            if observe:
                assert (
                    buffers["%arg3"]
                    == struct.pack("<i", initial) * (rows * columns) + guard
                ), name
            output_accesses = [
                (begin, end, action)
                for argument, begin, end, action in machine.accesses[
                    access_begin:
                ]
                if argument == "%arg2"
            ]
            coverage = Counter()
            for begin, end, action in output_accesses:
                assert action == "write", (name, action)
                coverage.update(range(begin, end))
            assert coverage == Counter(range(len(expected))), name
            trace = machine.trace[trace_begin:]
            clears = trace.count("bosc_ame.tile.zero_acc")
            loads = trace.count("bosc_ame.tile.load_acc")
            stores = trace.count("bosc_ame.tile.store_acc")
            if initial or observe or not blocks:
                assert clears == 0, (name, trace)
            else:
                assert clears > 0, (name, trace)
                assert stores == clears * blocks, (name, trace)
                assert loads == clears * (blocks - 1), (name, trace)
            assert not machine.pending and not machine.live_matrix, name

    single, single_path = compile_case("single_block", source(), "zero")
    simulate("single_block", single_path)
    _, multi_path = compile_case(
        "multiple_n_tiles", source(columns=128), "zero"
    )
    simulate("multiple_n_tiles", multi_path, columns=128)
    _, loop_path = compile_case(
        "static_k_loop",
        source(blocks=3, loop=True, start=1),
        "first-iteration-zero",
    )
    simulate("static_k_loop", loop_path, blocks=3, start=1)
    _, tail_path = compile_case(
        "padded_tail",
        source(blocks=2, loop=True, rows=4, columns=40),
        "first-iteration-zero",
    )
    simulate("padded_tail", tail_path, blocks=2, rows=4, columns=40)

    for name, arguments in (
        ("nonzero_initial", {"initial": 2147483600}),
        ("extra_consumer", {"observe": "whole"}),
        ("alias_observer", {"rows": 4, "columns": 40, "observe": "alias"}),
        ("zero_trip_loop", {"blocks": 0, "loop": True}),
    ):
        _, path = compile_case(name, source(**arguments), None)
        simulate(
            name,
            path,
            **{key: value for key, value in arguments.items() if key != "loop"},
        )

    # An opaque loop whose A/B storage also crosses iterations has no complete
    # task coverage. Zero-init planning must not dereference a missing task or
    # turn that incomplete graph into a deferred memory initialization.
    incomplete_source = source(blocks=3, loop=True, carried_ab=True)
    (work / "incomplete_loop.mlir").write_text(incomplete_source)
    incomplete = invoke(incomplete_source, "--kernel-contract-annotate")
    assert incomplete.returncode == 0, incomplete.stderr
    assert "complete = false" in incomplete.stdout
    incomplete_plan = invoke(incomplete.stdout, "--bosc-ame-plan")
    if incomplete_plan.returncode == 0:
        assert "initialization =" not in incomplete_plan.stdout
        assert 'kind = "deferred-zero"' not in incomplete_plan.stdout
        incomplete_emitted = invoke(incomplete_plan.stdout, "--bosc-ame-emit")
        assert incomplete_emitted.returncode == 0, incomplete_emitted.stderr
        assert "bosc_ame.tile.zero_acc" not in incomplete_emitted.stdout
    else:
        assert "error:" in incomplete_plan.stderr, incomplete_plan.stderr

    # Both public verification and the emitter's internal gate must re-prove
    # the source fill, coverage, task links, and first-iteration lifetime.
    begin, end, initialization = dictionaries(single, "initialization =")[0]
    malformed = {
        "wrong_kind": initialization.replace(
            'kind = "zero"', 'kind = "first-iteration-zero"'
        ),
        "unknown_kind": initialization.replace(
            'kind = "zero"', 'kind = "memory"'
        ),
        "wrong_producer": replace_field(initialization, "task", 99999),
        "missing_kind": remove_field(initialization, "kind"),
        "missing_producer": remove_field(initialization, "task"),
    }
    invalid_plans = {
        name: single[:begin] + replacement + single[end:]
        for name, replacement in malformed.items()
    }
    fill_begin, fill_end, fill = next(
        entry
        for entry in dictionaries(single, "plan.fill =")
        if 'kind = "deferred-zero"' in entry[2]
    )
    invalid_plans["wrong_consumer"] = (
        single[:fill_begin]
        + replace_field(fill, "consumer", 99999)
        + single[fill_end:]
    )
    invalid_plans["changed_fill"] = single.replace(
        "arith.constant 0 : i32", "arith.constant 1 : i32"
    )
    invalid_plans["wrong_init_type"] = single[:begin] + '"zero"' + single[end:]
    for name, invalid in invalid_plans.items():
        assert invalid != single, name
        for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            rejected = invoke(invalid, gate)
            assert rejected.returncode != 0, (name, gate)
            assert "error:" in rejected.stderr, (name, gate, rejected.stderr)

    print(
        "PASS deferred zero: private products, first-K clear, polluted/repeated launches, exact tails and proof rejection"
    )


if __name__ == "__main__":
    main()
