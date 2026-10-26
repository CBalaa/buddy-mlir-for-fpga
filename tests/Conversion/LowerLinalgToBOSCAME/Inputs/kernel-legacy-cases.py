"""Retain original buffer programs while checking the explicit v1 boundary.

Retired reports and global policy choices are covered by PlanRoutes.py,
PlanAccumulatorFusion.py, PlanTerminalStore.py, physical-adapter.py and the
KernelContract tests. Each source function below remains an independent input;
called functions stay together so helper ABI promises are never invented.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

COPY = "v1 has no independent DDR-to-DDR copy recipe"
LOAD = "scalar snapshot load requires a private SPM singleton"
HOST_LOAD = "scalar load must access planned host rank-zero storage"
UNKNOWN = (
    "V22/V23: read or partial inheritance has no initialized content version"
)
OVERLAP = "V25: overlapping views must have identical element addresses"
TERMINAL = "V23/V24: v1 does not support reading an external region after its terminal write"
ESCAPE = (
    "SPM root requires non-escaping aliases and completed local memory accesses"
)
PREFIX = "V4: static valid shape must match the access rank and be positive"
FRONTEND = (
    "compatibility input contains a non-neutral dialect or bridge operation"
)
TENSOR = "V3: contract must be buffer form, without tensor types"

# Entries with None are supported programs, not expected rejections. Keep this
# inventory explicit so adding/removing a source function cannot skip coverage.
EXPECTED = {
    "aliasing": {
        "overlap": OVERLAP,
        "disjoint": COPY,
        "same_global": UNKNOWN,
        "unknown": OVERLAP,
        "readonly": None,
        "same_buffer": None,
        "dynamic_disjoint": OVERLAP,
        "dynamic_overlap": OVERLAP,
        "dynamic_bases": OVERLAP,
        "dynamic_size": OVERLAP,
    },
    "buffer-alias-noabi": {
        "distinct_arguments": UNKNOWN,
        "called_helper": LOAD,
        "custom_space_arguments": "target memory must be rooted in a verified SPM window",
    },
    "buffer-alias-reinterpret": {
        name: HOST_LOAD
        if name == "reinterpret_rank2_dynamic_disjoint"
        else LOAD
        for name in [
            "reinterpret_disjoint",
            "reinterpret_overlap",
            "reinterpret_dynamic",
            "reinterpret_dynamic_base_disjoint",
            "reinterpret_distinct_dynamic_bases",
            "reinterpret_dynamic_disjoint",
            "reinterpret_dynamic_overlap",
            "reinterpret_rank2_dynamic_disjoint",
            "reinterpret_resets_source_offset",
            "reinterpret_shifted_pointer_unknown",
            "reinterpret_offset_overflow",
            "reinterpret_nonzero_root_offset",
            "reinterpret_nonzero_root_offset_disjoint",
            "reinterpret_nonzero_root_offset_overlap",
            "mixed_root_reinterpret_disjoint",
            "mixed_root_reinterpret_overlap",
            "mixed_root_reinterpret_equal",
            "reinterpret_dynamic_root_offset",
            "reinterpret_rank3",
        ]
    },
    "buffer-placement-ame-budget": {"ame_staging_is_in_the_budget": None},
    "buffer-placement-scf-legality": {
        "escaping_branch": UNKNOWN,
        "alias_through_iter_arg": UNKNOWN,
        "escaping_call": ESCAPE,
        "multiple_blocks": "unsupported ancestor scf.execute_region",
        "single_block_region": "unsupported ancestor scf.execute_region",
        "parallel_private_root": "unsupported ancestor scf.parallel",
    },
    "canonical-gemm": {"gemm": None},
    "dataflow-control-flow-alias": {
        "yielded_memref_alias": TERMINAL,
        "iterated_memref_alias": TERMINAL,
        **dict.fromkeys(
            [
                "divergent_memref_alias",
                "changed_view_memref_alias",
                "forwarded_view_intervals",
                "while_memref_alias",
                "while_changed_view_alias",
                "index_switch_memref_alias",
                "index_switch_divergent_alias",
                "select_memref_alias",
                "select_divergent_alias",
            ],
            UNKNOWN,
        ),
    },
    "direct-store-legality": {"masked": COPY},
    "direct-vector-store": {"copy": COPY},
    "direct-vector-transfer-fusion": {
        "copy": ESCAPE,
        "pitched_copy": "SPM root requires non-escaping aliases and completed local memory accesses",
        "destination_read": ESCAPE,
    },
    "direct-vector-transfer-gap": {"separated": ESCAPE},
    "dynamic-offset": {"aligned": COPY, "unknown": COPY},
    "dynamic-prefix-alignment": dict.fromkeys(
        [
            "aligned_mask",
            "reversed_clamp",
            "wrong_upper_bound",
            "selected_clamp",
            "selected_wrong_upper_bound",
            "unaligned_mask",
        ],
        PREFIX,
    ),
    "dynamic-prefix-errors": dict.fromkeys(
        [
            "strided_store",
            "overlap_prefix_helper",
            "global_peer_store",
            "oversized_prefix",
        ],
        PREFIX,
    ),
    "errors": {"scalar": COPY},
    "exit-contract": {
        "residual": (
            "Dialect `tt' not found",
            "type created with unregistered dialect",
            FRONTEND,
        ),
        "gather": "V3: arguments must be memrefs or ordinary scalars",
        "scatter": "V3: arguments must be memrefs or ordinary scalars",
        "module5": "V14: target attributes are forbidden in the neutral contract",
        "ordinary": "unplanned external fill",
    },
    "input-contract": {
        "tensor": TENSOR,
        "bridge": FRONTEND,
        "unused_tensor": TENSOR,
        "unrealized": FRONTEND,
        "module4": TENSOR,
    },
    "launch-grid-axis-invalid": {
        "bad_axes": "invalid legacy program-axis metadata"
    },
    "launch-grid-invalid": {
        "bad_extent": "expected three positive i32 extents",
        "bad_rank": "gpu.known_grid_size must contain exactly 3 elements",
        "bad_type": "gpu.known_grid_size must be a dense i32 array",
    },
    "matmul-example-golden": {
        "triton_matmul_i8_128x1024x128": "obsolete geometry attribute bosc_ame.bm",
    },
    "overflow": dict.fromkeys(
        ["shape", "bytes", "span", "span_bytes", "offset_span"],
        "unplanned external fill",
    ),
    "spm-dma-copies": {
        **dict.fromkeys(
            [
                "copy_i8",
                "copy_i16",
                "copy_i32",
                "copy_i64",
                "copy_f16",
                "copy_bf16",
                "copy_f32",
                "copy_f64",
                "different_row_pitches",
                "same_bank_disjoint_subviews",
                "shifted_typed_view",
                "masked_valid_subview",
            ],
            UNKNOWN,
        ),
        "staged_identity_generic": COPY,
    },
    "vector-computations": {
        "fill": "unplanned external fill",
        "copy": COPY,
        "accumulate": None,
        "scalar_broadcast": "V25: pointwise update requires parallel identity indexing",
    },
}


def matching(text, begin, opening="{", closing="}"):
    depth = 1
    quoted = escaped = False
    for end in range(begin + 1, len(text)):
        char = text[end]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if not depth:
                return end + 1
    raise AssertionError("unbalanced original source")


def modules(source):
    source = re.sub(r"//[^\n]*", "", source)
    aliases = "\n".join(
        line for line in source.splitlines() if line.startswith(("#", "!"))
    )
    for found in re.finditer(r"^module\b", source, re.M):
        opening = source.index("{", found.end())
        if source[found.end() : opening].strip() == "attributes":
            opening = source.index("{", matching(source, opening))
        prefix = aliases + "\n" if aliases else ""
        yield prefix + source[found.start() : matching(source, opening)]


def cases(source):
    for module_index, module in enumerate(modules(source)):
        functions = {}
        declarations = []
        starts = list(
            re.finditer(r"^  func.func (?:private )?@([\w]+)\(", module, re.M)
        )
        for index, found in enumerate(starts):
            args_end = matching(module, found.end() - 1, "(", ")")
            line_end = module.find("\n", args_end)
            first_brace = module.find("{", args_end)
            next_function = (
                starts[index + 1].start()
                if index + 1 < len(starts)
                else len(module)
            )
            if first_brace == -1 or first_brace >= next_function:
                declarations.append(module[found.start() : line_end])
                continue
            if "attributes" in module[args_end:first_brace]:
                first_brace = module.index("{", matching(module, first_brace))
            end = matching(module, first_brace)
            functions[found[1]] = module[found.start() : end]
        if not functions:
            yield f"module{module_index}", module
            continue
        first = module.index("  func.func ")
        prefix = module[:first]
        # Keep every declaration once, including declarations after a body.
        prefix = prefix.rstrip() + "\n" + "\n".join(declarations) + "\n"
        remaining = set(functions)
        for name in functions:
            if name not in remaining:
                continue
            connected = {name}
            while True:
                previous = len(connected)
                for candidate, body in functions.items():
                    references = set(re.findall(r"@([\w]+)", body))
                    if candidate in connected or references & connected:
                        connected.add(candidate)
                        connected |= references & functions.keys()
                if len(connected) == previous:
                    break
            remaining -= connected
            body = "\n".join(
                body
                for candidate, body in functions.items()
                if candidate in connected
            )
            yield name, prefix + body + "\n}\n"


def invoke(tool, source, *passes):
    result = subprocess.run(
        [tool, *passes], input=source, text=True, capture_output=True
    )
    assert "LLVM ERROR" not in result.stderr, result.stderr
    assert "PLEASE submit" not in result.stderr, result.stderr
    assert result.returncode >= 0, result.stderr
    return result


def check_result(result, diagnostic):
    if diagnostic is None:
        assert result.returncode == 0, result.stderr
        assert "linalg." not in result.stdout
        assert "plan." not in result.stdout
    else:
        assert result.returncode != 0, (diagnostic, result.stdout)
        alternatives = (
            diagnostic if isinstance(diagnostic, tuple) else (diagnostic,)
        )
        assert any(item in result.stderr for item in alternatives), (
            diagnostic,
            result.stderr,
        )


def supported(tool, filename, name, result):
    output, report = result.stdout, result.stderr
    if filename in {"canonical-gemm", "buffer-placement-ame-budget"}:
        large = filename == "canonical-gemm"
        assert output.count("bosc_ame.tile.mma") == (32 if large else 16)
        assert output.count("bosc_ame.tile.load_acc") == (2 if large else 1)
        assert output.count("bosc_ame.tile.store_acc") == (2 if large else 1)
        assert "plan.route" in report and "plan.budget" in report
        assert 'reason = "safe-terminal-forwarding"' in report
        if large:
            assert "high_water = array<i64: 0, 65536>" in report
            assert output.count("bosc_ame.dma.start") == 64
        else:
            assert "high_water = array<i64: 131072, 0>" in report
            assert "bosc_ame.dma.start" not in output
        for chain in re.findall(
            r"bosc_ame.tile.load_acc(.*?)bosc_ame.tile.store_acc", output, re.S
        ):
            assert "bosc_ame.dma." not in chain
            assert "func.call" not in chain
        lowered = invoke(tool, output, "--lower-bosc-ame")
        assert lowered.returncode == 0, lowered.stderr
        assert lowered.stdout.count("mqma.b.mm") == (32 if large else 16)
    else:
        assert filename in {"aliasing", "vector-computations"}, (filename, name)
        assert "vector.load" in output and "vector.store" in output
        assert "arith.addf" in output
        assert "bosc_ame.dma.start" not in output
        assert "high_water = array<i64: 0, 0>" in report


def reinterpret_proof(tool, name, program):
    annotated = invoke(
        tool, program, "--kernel-contract-annotate", "--kernel-contract-verify"
    )
    assert annotated.returncode == 0, annotated.stderr
    assert annotated.stdout.count("contract.task") == 2
    # Preserve descriptor arithmetic, changing the original first access into
    # a write to test actual alias safety instead of a retired pair report.
    load = re.search(r"%[\w]+ = memref.load ([^\n]+)\s*:\s*([^\n]+)", program)
    assert load, name
    probe = (
        program[: load.start()]
        + f"memref.store %probe, {load[1]} : {load[2]}"
        + program[load.end() :]
    )
    begin = probe.index("{", probe.index("func.func")) + 1
    probe = (
        probe[:begin] + "\n%probe = arith.constant 0.0 : f32\n" + probe[begin:]
    )
    result = invoke(
        tool, probe, "--kernel-contract-annotate", "--kernel-contract-verify"
    )
    if name == "reinterpret_disjoint":
        assert result.returncode == 0, result.stderr
    else:
        check_result(result, TERMINAL)


def staged_dynamic_offset(tool, name, program):
    """Alignment alone does not prove that an unbounded row is in storage."""
    begin = program.index("{", program.index("func.func")) + 1
    derived = (
        program[:begin]
        + "\n    %local = memref.alloc() alignment = 64 : memref<16xf32>\n"
        + program[begin:]
    )
    derived = derived.replace("outs(%out :", "outs(%local :").replace(
        "    return",
        "    memref.copy %local, %out : memref<16xf32> to memref<16xf32>\n    return",
    )
    result = invoke(
        tool, derived, "--linalg-to-boscame=dma-abi=args_aligned64_nonalias"
    )
    check_result(
        result, "DDR view exceeds its statically known argument storage"
    )
    if name != "unknown":
        bounded = derived.replace(
            "%offset = arith.muli %row, %c16 : index",
            "%offset = arith.constant 16 : index",
        )
        result = invoke(
            tool, bounded, "--linalg-to-boscame=dma-abi=args_aligned64_nonalias"
        )
        check_result(result, None)
        assert result.stdout.count("bosc_ame.dma.start") == 2
        verified = invoke(tool, result.stdout, "--bosc-ame-spm-dma-check")
        assert verified.returncode == 0, verified.stderr


def dma_roundtrip(tool, source, name, program):
    """Exercise initialized byte copies using the v1 DDR/SPM boundary form."""
    if name == "staged_identity_generic":
        return
    source_type = ""
    if name.startswith("copy_"):
        element = name[len("copy_") :]
        width = {
            "i8": 1,
            "i16": 2,
            "i32": 4,
            "i64": 8,
            "f16": 2,
            "bf16": 2,
            "f32": 4,
            "f64": 8,
        }[element]
        external_type = f"memref<64x{element}>"
        source_type = f"memref<64x{element}, #bosc_ame.spm<0>>"
        signature = f"(%input: {external_type}, %output: {external_type})"
        initial = (
            f"memref.copy %input, %source : {external_type} to {source_type}"
        )
        final = (
            f"memref.copy %source, %output : {source_type} to {external_type}"
        )
        input_bytes = output_bytes = 64 * width
        slices = [(0, 0, input_bytes)]
        dma_count = 2
    elif name == "different_row_pitches":
        signature = "(%input: memref<192xi8>, %output: memref<2x64xi8, strided<[192, 1]>>)"
        initial = "memref.copy %input, %source : memref<192xi8> to memref<192xi8, #bosc_ame.spm<0>>"
        final = "memref.copy %source_rows, %output : memref<2x64xi8, strided<[128, 1]>, #bosc_ame.spm<0>> to memref<2x64xi8, strided<[192, 1]>>"
        input_bytes, output_bytes = 192, 256
        slices = [(0, 0, 64), (128, 192, 64)]
        dma_count = 3
    elif name == "same_bank_disjoint_subviews":
        signature = "(%input: memref<256xi8>, %output: memref<64xi8>)"
        initial = "memref.copy %input, %bank : memref<256xi8> to memref<256xi8, #bosc_ame.spm<0>>"
        final = "memref.copy %source, %output : memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>> to memref<64xi8>"
        input_bytes, output_bytes = 256, 64
        slices = [(64, 0, 64)]
        dma_count = 2
    elif name == "shifted_typed_view":
        signature = "(%input: memref<128xi8>, %output: memref<16xi32>)"
        initial = "memref.copy %input, %source : memref<128xi8> to memref<128xi8, #bosc_ame.spm<0>>"
        final = "memref.copy %source_typed, %output : memref<16xi32, #bosc_ame.spm<0>> to memref<16xi32>"
        input_bytes, output_bytes = 128, 64
        slices = [(64, 0, 64)]
        dma_count = 2
    else:
        assert name == "masked_valid_subview", name
        signature = "(%input: memref<1024xf32>, %output: memref<1024xf32>)"
        initial = "memref.copy %input, %source : memref<1024xf32> to memref<1024xf32, #bosc_ame.spm<0>>"
        final = """
        %output_valid = memref.subview %output[0] [768] [1]
          : memref<1024xf32> to memref<768xf32, strided<[1]>>
        memref.copy %source_valid, %output_valid {bosc_ame.memory = {
          kind = "store", logical_shape = array<i64: 1024>,
          masked = true, valid_shape = array<i64: 768>}}
          : memref<768xf32, strided<[1]>, #bosc_ame.spm<0>>
            to memref<768xf32, strided<[1]>>
        """
        input_bytes = output_bytes = 4096
        slices = [(0, 0, 3072)]
        dma_count = 2
    derived = program.replace(f"@{name}()", f"@{name}{signature}")
    begin = derived.index("    memref.copy")
    end = derived.index("    return", begin)
    # The original descriptors remain. Only the external initialization and
    # destination replace an implicit assumption about the SPM entry contents.
    initialized = derived[:begin] + "    " + initial + "\n" + derived[begin:]
    if source_type:
        check_result(
            invoke(
                tool,
                initialized,
                "--linalg-to-boscame=dma-abi=args_aligned64_nonalias",
            ),
            "requires matching static rank-two SPM storage",
        )
    derived = (
        derived[:begin]
        + "    "
        + initial
        + "\n    "
        + final
        + "\n"
        + derived[end:]
    )
    result = invoke(
        tool, derived, "--linalg-to-boscame=dma-abi=args_aligned64_nonalias"
    )
    check_result(result, None)
    assert result.stdout.count("bosc_ame.dma.start") == dma_count
    assert result.stdout.count("bosc_ame.dma.wait") == dma_count
    assert (
        "vector.load" not in result.stdout
        and "vector.store" not in result.stdout
    )
    verified = invoke(tool, result.stdout, "--bosc-ame-spm-dma-check")
    assert verified.returncode == 0, verified.stderr
    if name in {"copy_f16", "copy_bf16"}:
        # The strict buffer simulator has no half-precision descriptor parser.
        # Keep their typed DMA generation/checker coverage without changing it.
        return
    sys.path.insert(
        0, str(source.resolve().parents[3] / "examples/FPGA-BOSCAME/tools")
    )
    from boscame_buffer_sim import Machine

    payload = bytes((index * 17 + 11) % 256 for index in range(input_bytes))
    guard = b"\xa7" * 64
    expected = bytearray(b"\xcc" * output_bytes)
    for read_offset, write_offset, count in slices:
        expected[write_offset : write_offset + count] = payload[
            read_offset : read_offset + count
        ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.mlir"
        path.write_text(result.stdout)
        machine = Machine(path, name)
        for bank in machine.spm:
            bank[:] = b"\x5a" * len(bank)
        incoming = bytearray(payload + guard)
        outgoing = bytearray(b"\xcc" * output_bytes + guard)
        machine.launch({"%arg0": incoming, "%arg1": outgoing})
        assert incoming == payload + guard
        assert outgoing == expected + guard
        assert machine.spm[0][:input_bytes] == payload


def main():
    source, tool = Path(sys.argv[1]), sys.argv[2]
    original = source.read_text()
    if source.stem == "matmul-example-golden":
        original = (
            source.parent / "Inputs/matmul-example-golden.mlir"
        ).read_text()
        original = original.split("//--- kernel.boscame.mlir")[0]
    programs = list(cases(original))
    assert {name for name, _ in programs} == EXPECTED[source.stem].keys()
    for name, program in programs:
        result = invoke(
            tool,
            program,
            "--linalg-to-boscame=dma-abi=args_aligned64_nonalias plan-report=true",
        )
        if "--probe" in sys.argv:
            errors = re.findall(r"error: ([^\n]+)", result.stderr)
            print(source.stem, name, result.returncode, errors[:1])
        else:
            diagnostic = EXPECTED[source.stem][name]
            check_result(result, diagnostic)
            if diagnostic is None:
                supported(tool, source.stem, name, result)
            if source.stem == "buffer-alias-reinterpret":
                reinterpret_proof(tool, name, program)
            if source.stem == "spm-dma-copies":
                dma_roundtrip(tool, source, name, program)
            if source.stem == "dynamic-offset":
                staged_dynamic_offset(tool, name, program)
    if "--probe" not in sys.argv:
        print(f"PASS {source.stem}: {len(programs)} original program groups")


if __name__ == "__main__":
    main()
