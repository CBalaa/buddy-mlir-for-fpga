# RUN: %PYTHON %s %t
"""Default V2 must not defer an accumulator's selected real DDR fill."""

import struct
import subprocess
import sys
from pathlib import Path

from PlanZeroAccumulator import source

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def check_execution(path, options):
    """Poisoned private storage and repeated calls expose missing real fills."""
    rows, columns = options["rows"], options["columns"]
    blocks = options.get("blocks", 1)
    inner = max(1, blocks) * 64
    initial = options.get("initial", 0)
    extremes = (-128, 127, -1, 0, 1, -127, 126)
    left = [extremes[(i * 3 + 1) % len(extremes)] for i in range(rows * inner)]
    right = [
        extremes[(i * 5 + 2) % len(extremes)] for i in range(columns * inner)
    ]
    expected = bytearray()
    for row in range(rows):
        for column in range(columns):
            result = initial + sum(
                left[row * inner + k] * right[column * inner + k]
                for k in range(blocks * 64)
            )
            wrapped = (result + (1 << 31)) % (1 << 32) - (1 << 31)
            expected.extend(struct.pack("<i", wrapped))
    guard = bytes([0xD7]) * 64
    a = bytes(value & 255 for value in left) + guard
    b = bytes(value & 255 for value in right) + guard
    machine = Machine(path, "product")
    machine.spm[0][:] = bytes([0x86]) * len(machine.spm[0])
    machine.spm[1][:] = bytes([0x39]) * len(machine.spm[1])
    for repeat in range(2):
        buffers = {
            "%arg0": bytearray(a),
            "%arg1": bytearray(b),
            "%arg2": bytearray(bytes([0xA6 + repeat]) * len(expected) + guard),
        }
        if options.get("observe"):
            buffers["%arg3"] = bytearray(bytes([0xCE]) * len(expected) + guard)
        machine.launch(buffers)
        assert buffers["%arg0"] == a and buffers["%arg1"] == b
        assert buffers["%arg2"] == expected + guard, (path, repeat)
        if options.get("observe"):
            assert buffers["%arg3"] == (
                struct.pack("<i", initial) * (rows * columns) + guard
            )
        assert not machine.pending and not machine.live_matrix


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    for name, options in (
        ("classifier", {"rows": 1, "columns": 40, "blocks": 6, "loop": True}),
        ("observed", {"rows": 4, "columns": 40, "observe": "alias"}),
        ("nonzero", {"rows": 1, "columns": 40, "initial": 2147483600}),
        ("empty_loop", {"rows": 1, "columns": 40, "blocks": 0, "loop": True}),
    ):
        text = source(**options)
        path = work / f"{name}.mlir"
        path.write_text(text)
        result = subprocess.run(
            [
                str(buddy_opt_path()),
                str(path),
                "--kernel-contract-annotate",
                "--bosc-ame-plan",
                "--bosc-ame-plan-verify",
            ],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, (name, result.stderr)
        planned = result.stdout
        (work / f"{name}.planned.mlir").write_text(planned)
        assert "plan.version = 2 : i32" in planned
        # If a complete register recipe eliminates the fill, it owns both
        # ends. Otherwise a materialized DDR fill cannot claim deferred zero.
        for line in planned.splitlines():
            if "linalg.fill" in line and "plan.cpu" in line:
                assert "plan.fill" not in line, (name, line)
        # Placement may legally select the paired SPM deferred-zero recipe.
        # The observed/nonzero/empty cases still require real initialization.
        if name != "classifier" or 'kind = "deferred-zero"' not in planned:
            assert "initialization =" not in planned, (name, planned)
            assert 'kind = "deferred-zero"' not in planned, (name, planned)
        emitted = subprocess.run(
            [str(buddy_opt_path()), "--bosc-ame-emit"],
            input=planned,
            text=True,
            capture_output=True,
        )
        assert emitted.returncode == 0, (name, emitted.stderr)
        emitted_path = work / f"{name}.emitted.mlir"
        emitted_path.write_text(emitted.stdout)
        lowered = subprocess.run(
            [
                str(buddy_opt_path()),
                str(emitted_path),
                "--bosc-ame-spm-dma-check",
                "--lower-bosc-ame",
            ],
            text=True,
            capture_output=True,
        )
        assert lowered.returncode == 0, (name, lowered.stderr)
        check_execution(emitted_path, options)
    print("PASS default V2 joint initialization and repeated exact execution")


if __name__ == "__main__":
    main()
