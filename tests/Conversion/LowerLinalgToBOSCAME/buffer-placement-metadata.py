# RUN: %PYTHON %s buddy-opt %S/Inputs/layernorm-197x768-bufferized.mlir
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


def main():
    source = Path(sys.argv[2]).read_text()
    result = subprocess.run(
        [
            sys.argv[1],
            sys.argv[2],
            "--linalg-to-boscame=buffer-placement=true store-policy=spm-dma",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    before, descriptors, counts, _ = graph(source)
    after, placed_descriptors, placed_counts, dma_tasks = graph(result.stdout)
    assert set(before) == set(range(22)), before
    expected = {
        task: dict(attributes)
        for task, attributes in before.items()
        if task not in {1, 2, 3}
    }
    # Fixed-prefix lowering consumes only the two data inputs. Its original
    # predicate and the now-dead width/range/compare tasks must leave the graph.
    expected[14]["bosc_ame.compute_input_values"] = "array<i64: 26, 1>"
    expected[14]["bosc_ame.compute_predecessors"] = "array<i64: 0, 13>"
    assert set(after) == set(expected), (set(expected), set(after))
    for task in expected:
        assert after[task] == expected[task], (
            task,
            expected[task],
            after[task],
        )
    assert descriptors == placed_descriptors, (
        descriptors - placed_descriptors,
        placed_descriptors - descriptors,
    )
    assert counts == ["22 : i64"]
    assert placed_counts == ["19 : i64"]
    assert dma_tasks == {5, 7, 9, 21}, dma_tasks
    assert result.stdout.count("bosc_ame.dma.start") == 4
    assert "memref<1024xi32>" not in result.stdout
    assert "memref<1024xi1>" not in result.stdout
    assert not re.search(r"memref\.alloca?[^\n]*: memref<f32>", result.stdout)
    assert not re.search(r"memref\.(?:load|store|copy)\b", result.stdout)
    assert result.stdout.count("vector.reduction <add>") == 2
    assert "vector<1xf32>" not in result.stdout
    print("PASS root placement: 19 surviving graph tasks and four DMA events")


if __name__ == "__main__":
    main()
