# RUN: %PYTHON %s buddy-opt %S/Inputs/layernorm-197x768-bufferized.mlir
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Root placement preserves the graph after removing dead prefix-mask tasks."""

import collections
import re
import subprocess
import sys
from pathlib import Path

CLOSING_BRACKETS = {">": "<", "}": "{", "]": "[", ")": "("}


def dictionaries(text):
    """Read attribute dictionaries, including nested logical-user arrays."""
    stack = []
    quoted = escaped = False
    for offset, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            stack.append(offset)
        elif char == "}":
            begin = stack.pop()
            content = text[begin + 1 : offset]
            # Operation regions also use braces. A dictionary starts with a
            # named field followed by '=' or a comma; a region starts with an
            # SSA definition, block label or operation with operands.
            if not re.match(r"\s*[A-Za-z_][\w.]*\s*(?:=|,|$)", content):
                continue
            fields = []
            brackets = []
            field = 0
            in_string = escape = False
            for index, token in enumerate(content):
                if in_string:
                    if escape:
                        escape = False
                    elif token == "\\":
                        escape = True
                    elif token == '"':
                        in_string = False
                elif token == '"':
                    in_string = True
                elif token in "<{[(":
                    brackets.append(token)
                elif token in ">}])":
                    if brackets and brackets[-1] == CLOSING_BRACKETS[token]:
                        brackets.pop()
                elif token == "," and not brackets:
                    fields.append(content[field:index])
                    field = index + 1
            fields.append(content[field:])
            attributes = {}
            for entry in fields:
                match = re.fullmatch(
                    r"\s*(bosc_ame\.[\w.]+)(?:\s*=\s*(.*?))?\s*",
                    entry,
                    re.DOTALL,
                )
                if match:
                    attributes[match[1]] = (
                        re.sub(r"\s+", " ", match[2]).strip()
                        if match[2] is not None
                        else None
                    )
            if attributes:
                prefix = text[text.rfind("\n", 0, begin) + 1 : begin]
                yield prefix, attributes


def graph(text):
    tasks = {}
    logical_descriptors = collections.Counter()
    counts = []
    dma_tasks = set()
    for prefix, attributes in dictionaries(text):
        if "bosc_ame.compute_graph_node_count" in attributes:
            counts.append(attributes["bosc_ame.compute_graph_node_count"])
        task = attributes.get("bosc_ame.compute_id")
        if task is not None:
            number = int(task.split()[0])
            assert number not in tasks, f"duplicate task {number}"
            tasks[number] = attributes
            if "bosc_ame.dma.start" in prefix:
                dma_tasks.add(number)
        elif "bosc_ame.logical_value_ids" in attributes:
            logical_descriptors[
                (
                    attributes["bosc_ame.logical_value_ids"],
                    attributes.get("bosc_ame.logical_value_users"),
                )
            ] += 1
    for number, attributes in tasks.items():
        predecessors = attributes["bosc_ame.compute_predecessors"]
        ids = predecessors.removeprefix("array<i64").rstrip("> ")
        if ids.startswith(":"):
            for predecessor in map(int, ids[1:].split(",")):
                assert predecessor in tasks, (number, predecessor)
                assert predecessor < number, (number, predecessor)
    return tasks, logical_descriptors, counts, dma_tasks


def contract_tasks(text):
    tasks = {}
    for body in re.findall(r"contract\.task = \{([^{}]*)\}", text):
        task = int(re.search(r"\bid = (\d+) : i64", body)[1])
        assert task not in tasks, task
        tasks[task] = body
    for task, body in tasks.items():
        predecessors = re.search(
            r"predecessors = array<i64(?:: ([^>]+))?>", body
        )
        assert predecessors, body
        for predecessor in map(
            int, (predecessors[1] or "").split(",") if predecessors[1] else []
        ):
            assert predecessor in tasks and predecessor < task, (
                task,
                predecessor,
            )
    return tasks


def main():
    binary = sys.argv[1]
    historical = Path(sys.argv[2]).read_text()
    before, _, counts, _ = graph(historical)
    assert len(before) == 22 and counts == ["22 : i64"]
    rejected = subprocess.run(
        [
            binary,
            sys.argv[2],
            "--linalg-to-boscame=plan-version=1 buffer-placement=true",
        ],
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "obsolete geometry attribute bosc_ame.bm" in rejected.stderr, (
        rejected.stderr
    )
    # The old fixture remains a migration case. The neutral LayerNorm graph
    # includes captured rank-zero loads and explicit broadcasts instead of
    # relying on retired prefix-mask/scalar-forwarding metadata.
    fixture = (
        Path(__file__).resolve().parents[3]
        / "tests/Conversion/LowerLinalgToBOSCAME/Inputs/contracts/layernorm.contract.mlir"
    )
    planned = subprocess.run(
        [
            binary,
            str(fixture),
            "--bosc-ame-plan=plan-version=1",
            "--bosc-ame-plan-verify",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    result = subprocess.run(
        [binary, "-", "--bosc-ame-emit"],
        input=planned,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    source_tasks = contract_tasks(fixture.read_text())
    planned_tasks, emitted_tasks = (
        contract_tasks(planned),
        contract_tasks(result),
    )
    assert len(source_tasks) == len(planned_tasks) == 21
    assert planned_tasks == emitted_tasks
    assert result.count("bosc_ame.dma.start ") == 4
    assert result.count("bosc_ame.dma.wait ") == 4
    assert result.count("bosc_ame.spm.view ") == 6
    assert result.count("vector.reduction <add>") == 2
    assert result.count('kind = "scalar-load"') == 2
    assert "bosc_ame.source_task_count = 21 : i64" in result
    assert result.count('action = "retained"') == 21
    dma_tasks = [
        line
        for line in result.splitlines()
        if "bosc_ame.dma.wait " in line and "contract.task" in line
    ]
    assert len(dma_tasks) == 4
    assert all(
        "contract.task" not in line
        for line in result.splitlines()
        if "bosc_ame.dma.start " in line
    )
    assert "linalg." not in result and "plan." not in result
    print(
        "PASS placement: 21 neutral tasks, captured scalars and four DMA completion events"
    )


if __name__ == "__main__":
    main()
