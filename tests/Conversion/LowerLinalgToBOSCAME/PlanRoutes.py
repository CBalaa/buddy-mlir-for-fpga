# RUN: %PYTHON %s %t
"""Exercise production route selection independently of any source frontend."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_check import buddy_opt_path  # noqa: E402


def number(text, field):
    return int(re.search(rf"\b{field} = (\d+) : i64", text)[1])


def selection(line, operand):
    return re.search(rf"\b{operand} = \{{([^}}]*)\}}", line)[1]


def windows(text):
    result = set()
    for buffer in re.findall(r"plan\.buffer = \{([^}]*)\}", text):
        if 'level = "spm"' in buffer:
            result.add(
                tuple(
                    number(buffer, key)
                    for key in ("bank", "offset", "capacity")
                )
            )
    return result


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(path, *passes):
        return subprocess.run(
            [opt, str(path), *passes], text=True, capture_output=True
        )

    def raw_case(name, source):
        path = work / f"{name}.mlir"
        path.write_text(source)
        return invoke(
            path,
            "--kernel-contract-annotate",
            "--bosc-ame-plan=plan-version=1 plan-report=true",
            "--bosc-ame-plan-verify",
        )

    # The expected plans are produced by the actual planner, rather than read
    # from the hand-authored *.planned.mlir design references.
    for name, count, window_count, workspace, source_count in (
        ("kernel", 10, 2, 20480, 12),
        ("matmul128", 1, 0, 0, 5),
        ("layernorm", 21, 6, 24576, 21),
    ):
        result = invoke(
            ROOT
            / "tests/Conversion/LowerLinalgToBOSCAME/Inputs/contracts"
            / f"{name}.contract.mlir",
            "--bosc-ame-plan=plan-version=1 plan-report=true",
            "--bosc-ame-plan-verify",
        )
        assert result.returncode == 0, (name, result.stderr)
        planned = result.stdout
        assert planned.count("contract.task =") == count, name
        assert number(planned, "plan.source_task_count") == source_count
        placed = windows(planned)
        assert len(placed) == window_count, (name, placed)
        assert sum(capacity for _, _, capacity in placed) == workspace, (
            name,
            placed,
        )
        high_water = tuple(
            int(value)
            for value in re.search(
                r"high_water = array<i64: ([^>]*)>", planned
            )[1].split(",")
        )
        assert sum(high_water) == workspace, (name, high_water)
        routes = [
            line for line in planned.splitlines() if "plan.route =" in line
        ]
        if name == "kernel":
            assert len(routes) == 2
            for route in routes:
                assert 'level = "spm"' in selection(route, "a")
                assert 'level = "ddr"' in selection(route, "b")
                assert 'level = "spm"' in selection(route, "accumulator")
            assert "private-accumulator" in result.stderr
        elif name == "matmul128":
            assert len(routes) == 1
            for operand in ("a", "b", "accumulator", "output"):
                assert 'level = "ddr"' in selection(routes[0], operand)
            assert planned.count('action = "folded-load"') == 3
            assert planned.count('action = "folded-store"') == 1
            assert "plan.transfer =" not in planned
        else:
            assert not routes
            assert planned.count('level = "host", role = "reduction"') == 2

        planned_path = work / f"{name}.planned.mlir"
        planned_path.write_text(planned)
        result = invoke(planned_path, "--bosc-ame-emit")
        assert result.returncode == 0, (name, result.stderr)
        assert "plan." not in result.stdout and "linalg." not in result.stdout
        assert result.stdout.count("contract.task =") == count
        assert (
            number(result.stdout, "bosc_ame.source_task_count") == source_count
        )
        assert "bosc_ame.task_map =" in result.stdout
        if name == "matmul128":
            assert "bosc_ame.spm.view" not in result.stdout
            assert "bosc_ame.dma.start" not in result.stdout
            assert result.stdout.count("bosc_ame.tile.mma") == 2
        elif name == "kernel":
            assert result.stdout.count("bosc_ame.tile.mma") == 8
            assert result.stdout.count("bosc_ame.spm.view") == 4
            for line in result.stdout.splitlines():
                if "bosc_ame.tile.load_b" in line:
                    assert "bosc_ame.direct_ddr" in line
        else:
            assert result.stdout.count("bosc_ame.spm.view") == 6
            assert result.stdout.count("memref.alloca") == 2

    source = Path(__file__).with_suffix(".mlir").read_text()
    result = raw_case("capacity_retry", source)
    assert result.returncode == 0, result.stderr
    assert "capacity-fallback-safe-direct" in result.stderr
    assert "capacity-fallback-safe-direct" in result.stdout
    assert windows(result.stdout) == {(0, 0, 131072), (1, 0, 131072)}
    assert "plan.transfer =" not in result.stdout
    route = next(
        line for line in result.stdout.splitlines() if "plan.route =" in line
    )
    assert 'level = "ddr"' in selection(route, "a")

    # With space available, A is staged once because two N subtiles otherwise
    # read it twice. B and C remain direct in the very same matmul.
    spacious = source.replace("memref<131072xi8>", "memref<64xi8>")
    result = raw_case("traffic_prefers_staging", spacious)
    assert result.returncode == 0, result.stderr
    route = next(
        line for line in result.stdout.splitlines() if "plan.route =" in line
    )
    assert 'level = "spm"' in selection(route, "a")
    assert 'level = "ddr"' in selection(route, "b")
    assert 'level = "ddr"' in selection(route, "accumulator")
    assert "plan.transfer =" in result.stdout
    assert "capacity-fallback-safe-direct" not in result.stdout

    # Non-cache-line A pitches use exact RVV rows in the selected staging
    # window; B and C retain their independently legal direct DDR routes.
    result = raw_case(
        "unaligned_pitch_staging",
        spacious.replace(
            "memref<32x128xi8>", "memref<32x128xi8, strided<[129, 1]>>"
        ),
    )
    assert result.returncode == 0, result.stderr
    planned = result.stdout
    route = next(
        line for line in planned.splitlines() if "plan.route =" in line
    )
    assert 'level = "spm"' in selection(route, "a")
    assert 'level = "ddr"' in selection(route, "b")
    assert 'level = "ddr"' in selection(route, "accumulator")
    assert 'engine = "segmented"' in planned
    assert 'engine = "rvv"' in planned
    assert "dma-endpoint-residues-differ" in planned
    planned_path = work / "unaligned_pitch_staging.planned.mlir"
    planned_path.write_text(planned)
    verified = invoke(planned_path, "--bosc-ame-plan-verify")
    assert verified.returncode == 0, verified.stderr
    emitted = invoke(planned_path, "--bosc-ame-emit")
    assert emitted.returncode == 0, emitted.stderr
    assert "bosc_ame.spm.view" in emitted.stdout
    assert "vector.load" in emitted.stdout
    assert "bosc_ame.direct_ddr" in emitted.stdout
    emitted_path = work / "unaligned_pitch_staging.emitted.mlir"
    emitted_path.write_text(emitted.stdout)
    checked = invoke(
        emitted_path, "--bosc-ame-spm-dma-check", "--lower-bosc-ame"
    )
    assert checked.returncode == 0, checked.stderr

    for name, bad_source, message in (
        (
            "external_scalar_store",
            """module {
  func.func @scalar_store(%out: memref<1xi32>) {
    %zero = arith.constant 0 : index
    %value = arith.constant 7 : i32
    memref.store %value, %out[%zero] : memref<1xi32>
    return
  }
}""",
            "no independent external or SPM scalar-store recipe",
        ),
        (
            "capacity_still_exceeded",
            source.replace("memref<131072xi8>", "memref<131136xi8>"),
            "SPM workspace capacity exceeded after legal direct/staging alternatives",
        ),
    ):
        result = raw_case(name, bad_source)
        assert result.returncode != 0, name
        assert message in result.stderr, (name, result.stderr)
        assert "store policy" not in result.stderr, (name, result.stderr)
        assert (
            "LLVM ERROR" not in result.stderr
            and "PLEASE submit" not in result.stderr
        )

    print(
        "PASS production planner: three contracts, mixed routes, capacity retry, and bounded diagnostics"
    )


if __name__ == "__main__":
    main()
