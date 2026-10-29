"""Check retained memory fixtures through contracts and explicit plan limits.

The old pair/access report is retired. Each original source function remains
independently checked, including target/pointer inputs outside the neutral ABI.
Derived store/read probes test alias safety instead of inspecting report text.
"""

import re
import subprocess
import sys
from pathlib import Path

SOURCE = Path(sys.argv[1])
TOOL = sys.argv[2]
SCALAR_STORE = "v1 has no independent external or SPM scalar-store recipe"
SCALAR_LOAD = "scalar snapshot load requires a private SPM singleton"
HOST_SCALAR_LOAD = "scalar load must access planned host rank-zero storage"


def functions(source):
    result = {}
    for found in re.finditer(r"func\.func @([\w]+)\(", source):
        begin = source.index("{", found.end())
        depth, end = 1, begin + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        result[found[1]] = source[found.start() : end]
    return result


def module(function, abi=False):
    declarations = ""
    if "func.call @opaque" in function:
        declarations += "func.func private @opaque()\n"
    if "func.call @called_helper" in function:
        declarations += "func.func private @called_helper(memref<16xf32>, memref<16xf32>, f32) -> f32\n"
    attrs = (
        "contract.abi = {args_disjoint = true, args_aligned_bytes = 64 : i64}"
        if abi
        else ""
    )
    return f"module attributes {{{attrs}}} {{\n{declarations}{function}\n}}"


def run(source, *passes):
    result = subprocess.run(
        [TOOL, *passes], input=source, text=True, capture_output=True
    )
    assert (
        "LLVM ERROR" not in result.stderr
        and "PLEASE submit" not in result.stderr
    ), result.stderr
    return result


def reject(source, diagnostic, *passes):
    result = run(source, *passes)
    assert result.returncode != 0, (diagnostic, result.stdout)
    assert diagnostic in result.stderr, (diagnostic, result.stderr)


def annotated(function, abi=False):
    result = run(
        module(function, abi),
        "--kernel-contract-annotate",
        "--kernel-contract-verify",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def graph(output):
    return re.search(r"contract.compute_graph = \{([^}]*)\}", output)[1]


def tasks(output):
    return re.findall(r"contract.task = \{([^}]*)\}", output)


def ids(task, field):
    found = re.search(rf"\b{field} = array<i64(?:: ([^>]*))?>", task)
    assert found, task
    return list(map(int, found[1].split(","))) if found[1] else []


def memory_cases(cases):
    invalid = {
        "raw": "V23/V24",
        "execute_region_order": "V23/V24",
        "nested_scf": "V23/V24",
        "allocation_lifetime": "v1 private scalar stores require rank-zero storage",
        "empty_allocation_lifetime": "buffer must describe a static rank-0/1/2 allocation",
        "mixed_pointer": "V3/V14: nonstandard pointer or target types",
    }
    incomplete = {
        "unknown_barrier",
        "caller",
        "nested_execute_region",
        "multiblock_function",
        "multiblock_execute_region",
    }
    complete = {
        "war",
        "waw",
        "read_read",
        "disjoint_static_views",
        "distinct_arguments",
        "called_helper",
    }
    assert set(cases) == invalid.keys() | incomplete | complete
    for name, source in cases.items():
        if name in invalid:
            reject(
                module(source),
                invalid[name],
                "--kernel-contract-annotate",
                "--kernel-contract-verify",
            )
            continue
        output = annotated(source)
        expected = "false" if name in incomplete else "true"
        assert re.search(rf"\bcomplete = {expected}\b", graph(output)), (
            name,
            output,
        )
        records = tasks(output)
        if name in {"war", "waw", "distinct_arguments", "called_helper"}:
            assert len(records) == 2, (name, output)
            assert ids(records[1], "predecessors") == [0], (name, output)
            reject(
                output.replace(
                    "predecessors = array<i64: 0>", "predecessors = array<i64>"
                ),
                "V23/V24: missing RAW, WAR, WAW or scalar SSA dependency",
                "--kernel-contract-verify",
            )
        if name in {"read_read", "disjoint_static_views"}:
            assert len(records) == 2
            assert all(ids(task, "predecessors") == [] for task in records), (
                output
            )
        if name == "read_read":
            assert ids(records[0], "input_values") == ids(
                records[1], "input_values"
            )
        if name in {
            "war",
            "waw",
            "distinct_arguments",
            "called_helper",
            "disjoint_static_views",
            "unknown_barrier",
        }:
            reject(
                output,
                SCALAR_STORE
                if name in {"waw", "disjoint_static_views"}
                else SCALAR_LOAD,
                "--bosc-ame-plan=plan-version=1",
            )
        elif name == "read_read":
            reject(output, SCALAR_LOAD, "--bosc-ame-plan=plan-version=1")
        if name in {"caller", "multiblock_function"}:
            assert not records, output
    # An entry ABI can prove distinct descriptors independent. It is not
    # silently attached to a helper which receives the same actual argument.
    output = annotated(cases["distinct_arguments"], abi=True)
    assert ids(tasks(output)[1], "predecessors") == []
    cross_waw = (
        cases["distinct_arguments"]
        .replace(
            "%read = memref.load %left[%c0] : memref<16xf32>",
            "memref.store %value, %left[%c0] : memref<16xf32>",
        )
        .replace("return %read : f32", "return %value : f32")
    )
    assert ids(tasks(annotated(cross_waw))[1], "predecessors") == [0]
    assert ids(tasks(annotated(cross_waw, abi=True))[1], "predecessors") == []
    # The supported full-buffer equivalent retains allocation/fill/deallocation
    # and carries an actual initialized content version into explicit placement.
    initialized = (
        cases["allocation_lifetime"]
        .replace(
            "memref.store %value, %buffer[%c0] : memref<16xf32>",
            "linalg.fill ins(%value : f32) outs(%buffer : memref<16xf32>)",
        )
        .replace("memref.alloc() :", "memref.alloc() alignment = 64 :")
    )
    output = annotated(initialized)
    assert ids(tasks(output)[0], "output_values") == [0]
    private_raw = initialized.replace(
        "memref.dealloc %buffer : memref<16xf32>",
        "%read = memref.load %buffer[%c0] : memref<16xf32>\n"
        "memref.dealloc %buffer : memref<16xf32>",
    )
    private_output = annotated(private_raw)
    records = tasks(private_output)
    assert ids(records[1], "input_values") == ids(records[0], "output_values")
    assert ids(records[1], "predecessors") == [0]
    reject(private_output, SCALAR_LOAD, "--bosc-ame-plan=plan-version=1")
    planned = run(
        output,
        "--bosc-ame-plan=plan-version=1 plan-report=true",
        "--bosc-ame-plan-verify",
    )
    assert planned.returncode == 0, planned.stderr
    assert "plan.buffer" in planned.stdout and "plan.fill" in planned.stdout
    assert "plan.buffer" in planned.stderr and "plan.fill" in planned.stderr


LOAD = re.compile(r"%[\w]+ = memref\.load ([^\n]+)\s*:\s*([^\n]+)")


def buffer_cases(cases):
    target = {
        "memory_space_cast_disjoint_views",
        "memory_space_cast_overlapping_views",
    }
    selects = {"select_different_views", "select_same_view"}
    static_disjoint = {"shared_root_views", "reshape_disjoint_siblings"}
    for name, source in cases.items():
        if name in target:
            reject(
                module(source),
                "target memory spaces are forbidden in the neutral contract",
                "--kernel-contract-annotate",
            )
            continue
        if name == "direct_ddr_copy":
            reject(
                module(source),
                "target attributes are forbidden in the neutral contract",
                "--kernel-contract-annotate",
            )
            neutral = re.sub(
                r"bosc_ame.memory = \{[^}]*\}",
                'contract.access = {boundary = "external_ddr", kind = "store", origin_id = 77 : i64, mask = {kind = "none"}}',
                source,
            )
            output = annotated(neutral, abi=True)
            assert "origin_id = 77 : i64" in output
            assert ids(tasks(output)[0], "output_values") == []
            reject(
                output,
                "v1 has no independent DDR-to-DDR copy recipe",
                "--bosc-ame-plan=plan-version=1",
            )
            continue
        if name in selects:
            reject(
                module(source),
                "V22/V23: read or partial inheritance has no initialized content version",
                "--kernel-contract-annotate",
            )
            if name == "select_same_view":
                result = run(
                    module(source),
                    "--canonicalize",
                    "--kernel-contract-annotate",
                    "--kernel-contract-verify",
                )
                assert result.returncode == 0, result.stderr
            # Unknown descriptor alternatives cannot establish disjointness.
        else:
            output = annotated(source)
            expected = "false" if name == "shared_root_views" else "true"
            assert re.search(rf"\bcomplete = {expected}\b", graph(output)), (
                name,
                output,
            )
            records = tasks(output)
            assert len(records) >= 2, (name, output)
            assert ids(records[0], "input_values") == ids(
                records[1], "input_values"
            ), (name, output)
            indices = LOAD.search(source)[1].rsplit("[", 1)[1].split("]", 1)[0]
            reject(
                output,
                SCALAR_LOAD if "," not in indices else HOST_SCALAR_LOAD,
                "--bosc-ame-plan=plan-version=1",
            )
        # Convert the first original load into a write, preserving its exact
        # descriptor/index/type. Unknown/dynamic/overflowing spans must fail
        # conservatively; retained static byte proofs permit a later read.
        found = LOAD.search(source)
        assert found, name
        probe = (
            source[: found.start()]
            + f"memref.store %probe, {found[1]} : {found[2]}"
            + source[found.end() :]
        )
        begin = probe.index("{") + 1
        probe = (
            probe[:begin]
            + "\n%probe = arith.constant 0.0 : f32\n"
            + probe[begin:]
        )
        if name in static_disjoint:
            annotated(probe)
        else:
            reject(module(probe), "V23/V24", "--kernel-contract-annotate")


cases = functions(SOURCE.read_text())
if SOURCE.stem == "memory-dependencies":
    memory_cases(cases)
else:
    assert SOURCE.stem == "buffer-facts"
    assert len(cases) == 32, len(cases)
    buffer_cases(cases)
print(
    f"PASS {SOURCE.stem}: {len(cases)} original source cases, contract safety and explicit plan limits"
)
