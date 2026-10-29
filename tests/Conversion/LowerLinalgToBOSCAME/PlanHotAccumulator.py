# RUN: %PYTHON %s %t
"""Measured hot private P selects SPM; unmeasured routes remain legal DDR."""

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

_spec = importlib.util.spec_from_file_location(
    "register_fixture", Path(__file__).with_name("PlanRegisterSchedule.py")
)
_register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_register)

MEASURED = {
    (False, 6): (1275161, 177632, 980, "573a4d65db6847b690ee5c29f37b4f0b"),
    (False, 24): (
        1628473,
        556539,
        1712,
        "3217f5dc739740a58ef3e3a13ef40240+a82c64e45ec0473c8368bd10e62b2674",
    ),
    (True, 6): (2453211, 677892, 814, "1cf30d811eaa4bcb9f947041d3d18f0b"),
    (True, 24): (
        2807472,
        1047294,
        1314,
        "6a7ff0d0ec1d40cd9d42c3e6c2a43df4+6c0b5c40159246488f0afd8c50bca1b2",
    ),
}
INTEGRATION = (
    "--linalg-to-boscame=plan-version=2 enable-spm-dma=true "
    "dma-abi=args_aligned64_nonalias target=nr-fpga"
)


def hot_buffer(planned):
    """Read the single flat hot decision, preserving its exact text for forgeries."""
    choices = [
        match
        for match in re.finditer(r"plan.buffer = \{([^{}]+)\}", planned)
        if 'role = "hot-accumulator"' in match[1]
    ]
    assert len(choices) == 1, planned
    fields = {}
    for entry in choices[0][1].split(","):
        key, value = entry.strip().split(" = ", 1)
        if value.startswith('"'):
            fields[key] = value.strip('"')
        else:
            assert value.endswith(" : i64"), entry
            fields[key] = int(value.removesuffix(" : i64"))
    return choices[0][0], fields


def source(
    trips=6,
    *,
    capacity=False,
    observe=True,
    post_add=False,
    rows=32,
    columns=64,
    extra_observer=False,
):
    """Two final copies observe the same full 8 KiB product.

    This ordinary source feature deliberately prevents unique-store register
    fusion. It is not a policy-disable flag or corrupt scheduling metadata.
    Every A row is loaded through a private full tile, B through private
    physical-NxK storage, and P survives all K iterations.
    """
    text = _register.source(
        rows,
        columns,
        trips * 128,
        trips,
        initial_zero=not post_add,
        post_add=post_add,
    )
    arguments = "%arg2: memref<*xi32>, %arg3: memref<*xi32>"
    if extra_observer:
        arguments += ", %arg4: memref<*xi32>"
    text = text.replace(
        "%arg2: memref<*xi32>)",
        arguments + ")",
    )
    private_type = f"memref<{rows}x{columns}xi32>"
    insertion = ""
    if observe:
        insertion = f"""
    %snapshot = memref.reinterpret_cast %arg3 to offset: [0], sizes: [{rows}, {columns}], strides: [{columns}, 1]
      : memref<*xi32> to {private_type}
    memref.copy %c, %snapshot : {private_type} to {private_type}
"""
    if extra_observer:
        insertion += f"""
    %extra = memref.reinterpret_cast %arg4 to offset: [0], sizes: [{rows}, {columns}], strides: [{columns}, 1]
      : memref<*xi32> to {private_type}
    memref.copy %c, %extra : {private_type} to {private_type}
"""
    text = text.replace(
        "    memref.dealloc %c : " + private_type,
        insertion + "    memref.dealloc %c : " + private_type,
    )
    if capacity:
        # The target-input compatibility adapter reserves both live fixed
        # windows. They are observable after the hot P lifetime and cannot
        # disappear or be overlaid with P.
        text = text.replace(
            "    %c = memref.alloc()",
            """
    %fixed_zero = arith.constant 0 : i8
    %fixed0 = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<131072xi8, #bosc_ame.spm<0>>
    %fixed1 = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<131072xi8, #bosc_ame.spm<1>>
    linalg.fill ins(%fixed_zero : i8) outs(%fixed0 : memref<131072xi8, #bosc_ame.spm<0>>)
    linalg.fill ins(%fixed_zero : i8) outs(%fixed1 : memref<131072xi8, #bosc_ame.spm<1>>)
    %c = memref.alloc()""",
        )
        text = text.replace(
            "    return",
            """
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%fixed0, %fixed1 : memref<131072xi8, #bosc_ame.spm<0>>, memref<131072xi8, #bosc_ame.spm<1>>)
      outs(%fixed0 : memref<131072xi8, #bosc_ame.spm<0>>) {
    ^bb0(%x: i8, %y: i8, %fixed_old: i8):
      %sum = arith.addi %x, %y : i8
      linalg.yield %sum : i8
    }
    return""",
        )
    return text


def fixed_spm(text):
    """Constrain P to a physical bank window using the supported input API."""
    private_type = "memref<32x64xi32, #bosc_ame.spm<0>>"
    text = text.replace(
        "%c = memref.alloc() alignment = 64 : memref<32x64xi32>",
        "%c = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : "
        + private_type,
    )
    text = text.replace("%c : memref<32x64xi32>", "%c : " + private_type)
    text = text.replace(
        "ins(%c, %old : memref<32x64xi32>, memref<32x64xi32>)",
        "ins(%c, %old : " + private_type + ", memref<32x64xi32>)",
    )
    for endpoint in ("%out_full", "%snapshot"):
        text = text.replace(
            "memref.copy %c, " + endpoint + " : memref<32x64xi32>",
            "memref.copy %c, " + endpoint + " : " + private_type,
        )
    return text.replace("    memref.dealloc %c : " + private_type, "")


def execute_outputs(
    path,
    trips,
    *,
    post_add=True,
    repeats=1,
    full_banks=False,
    rows=32,
    columns=64,
    extra_observer=False,
    storage=None,
    initial=0,
    c_offset=0,
    k_trips=None,
):
    """Check both terminal observations, dirty storage and memory guards."""
    pitch = trips * 128
    k_trips = trips if k_trips is None else k_trips
    tile_bytes = rows * columns * 4
    a = bytearray((i * 37 + 13) % 256 for i in range(rows * pitch))
    b = bytearray((i * 17 + 211) % 256 for i in range(columns * pitch))
    a.extend(bytes([0xD1]) * 64)
    b.extend(bytes([0xE2]) * 64)
    c = bytearray().join(
        struct.pack("<i", _register.wrapped(0x7FFFFF00 + i))
        for i in range(rows * pitch)
    )
    c.extend(bytes([0xF3]) * 64)
    snapshot = bytearray([0xA5]) * tile_bytes + bytearray([0xB6]) * 64
    extra = bytearray([0x74]) * tile_bytes + bytearray([0x85]) * 64
    expected_c, expected_snapshot = bytearray(c), bytearray(snapshot)
    for row in range(rows):
        for column in range(columns):
            old = (
                struct.unpack_from(
                    "<i", c, 4 * (c_offset + row * pitch + column)
                )[0]
                if post_add
                else 0
            )
            product = sum(
                _register.signed(a[row * pitch + q])
                * _register.signed(b[column * pitch + q])
                for q in range(pitch)
            )
            result = _register.wrapped(old + initial + product)
            struct.pack_into(
                "<i", expected_c, 4 * (c_offset + row * pitch + column), result
            )
            struct.pack_into(
                "<i", expected_snapshot, 4 * (row * columns + column), result
            )
    saved_a, saved_b = bytes(a), bytes(b)
    initial_c, initial_snapshot, initial_extra = (
        bytes(c),
        bytes(snapshot),
        bytes(extra),
    )
    machine = Machine(path, "matrix")
    for repetition in range(repeats):
        c[:], snapshot[:] = initial_c, initial_snapshot
        extra[:] = initial_extra
        for bank in machine.spm:
            bank[:] = bytes([0x6B ^ repetition]) * len(bank)
        machine.private_poison = 0xC7 ^ repetition
        buffers = {"%arg0": a, "%arg1": b, "%arg2": c, "%arg3": snapshot}
        if extra_observer:
            buffers["%arg4"] = extra
        machine.launch(buffers, {})
        assert c == expected_c and snapshot == expected_snapshot, path
        assert bytes(a) == saved_a and bytes(b) == saved_b, path
        if extra_observer:
            assert (
                extra == expected_snapshot[:tile_bytes] + bytes([0x85]) * 64
            ), path
        assert machine.private_top == 0, path
        if full_banks:
            assert all(not any(bank) for bank in machine.spm), path
    if storage in ("spm", "acc"):
        assert not machine.frame_log and not machine.ame_frame_accesses, path
    if full_banks or storage == "ddr":
        assert [entry[0] for entry in machine.frame_log] == [
            "acquire",
            "release",
        ] * repeats
        traffic = {}
        for _, begin, end, kind in machine.ame_frame_accesses:
            traffic[kind] = traffic.get(kind, 0) + end - begin
        assert traffic == {
            "read": repeats * k_trips * tile_bytes,
            "write": repeats * k_trips * tile_bytes,
        }, traffic


def buffer_attribute(fields):
    entries = []
    for key, value in sorted(fields.items()):
        literal = f'"{value}"' if isinstance(value, str) else f"{value} : i64"
        entries.append(f"{key} = {literal}")
    return "plan.buffer = {" + ", ".join(entries) + "}"


def expected_hot(
    trips,
    *,
    post_add=False,
    rows=32,
    extra_observer=False,
    reason="hot-accumulator-measured-profitable",
):
    capacity = rows * 64 * 4
    fields = {
        "alignment": 64,
        "capacity": capacity,
        "role": "hot-accumulator",
        "level": "ddr",
        "reason": reason,
        "trips": trips,
        "loop_bytes": trips * 2 * capacity,
        "boundary_bytes": capacity * (6 if post_add else 3),
    }
    if extra_observer:
        fields["boundary_bytes"] += capacity
    if reason == "hot-accumulator-measured-profitable":
        ddr, spm, margin, evidence = MEASURED[post_add, trips]
        fields.update(
            level="spm",
            bank=0,
            offset=0,
            profile="nr-e6-hot-p-v1",
            evidence=evidence,
            mode="normal-post-add" if post_add else "overwrite",
            pitch=trips * 128,
            ddr_cycles_x2=ddr,
            spm_cycles_x2=spm,
            margin_cycles_x2=margin,
            saving_cycles_x2=ddr - spm,
        )
    return fields


def fixed_old_spm(text):
    """Only old-C has an explicit bank1 constraint; P must stay legal DDR."""
    ordinary = "memref<32x64xi32>"
    physical = "memref<32x64xi32, #bosc_ame.spm<1>>"
    text = text.replace(
        "%old = memref.alloc() alignment = 64 : " + ordinary,
        "%old = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : "
        + physical,
    )
    text = text.replace(
        "ins(%c, %old : " + ordinary + ", " + ordinary + ")",
        "ins(%c, %old : " + ordinary + ", " + physical + ")",
    )
    text = re.sub(
        r"(memref.copy %out_full, %old[^\n]+ to )memref<32x64xi32>",
        lambda match: match[1] + physical,
        text,
    )
    return text.replace("      memref.dealloc %old : " + ordinary, "")


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(name, text, *arguments):
        path = work / f"{name}.mlir"
        path.write_text(text)
        return subprocess.run(
            [opt, str(path), *arguments], capture_output=True, text=True
        )

    def plan(name, text):
        result = invoke(
            name,
            text,
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
            "--bosc-ame-plan-verify",
        )
        assert result.returncode == 0, (name, result.stderr)
        (work / f"{name}.planned.mlir").write_text(result.stdout)
        return result.stdout

    def reject(name, text):
        for gate in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
            result = invoke(name, text, gate)
            assert result.returncode != 0, (name, gate, result.stdout)
            assert "error:" in result.stderr, (name, gate, result.stderr)

    def check_metadata(name, planned, expected, *, forge=False):
        exact, actual = hot_buffer(planned)
        assert actual == expected, (name, actual, expected)
        assert "plan.register_schedule" not in planned, name
        assert "plan.automatic_spm = true" in planned, name
        if not forge:
            return
        # All source facts, measured values and physical windows are separately
        # reconstructed. One edited field cannot broaden the measured recipe.
        for key, value in expected.items():
            corrupted = dict(expected)
            corrupted[key] = (
                value + "-forged" if isinstance(value, str) else value + 64
            )
            reject(
                f"{name}.bad_{key}",
                planned.replace(exact, buffer_attribute(corrupted), 1),
            )
        for key in (
            "role",
            "reason",
            "capacity",
            "alignment",
            "profile",
            "evidence",
            "mode",
            "saving_cycles_x2",
            "bank",
            "offset",
        ):
            if key in expected:
                missing = dict(expected)
                del missing[key]
                reject(
                    f"{name}.missing_{key}",
                    planned.replace(exact, buffer_attribute(missing), 1),
                )
        reject(
            name + ".extra_field",
            planned.replace(
                exact,
                buffer_attribute(dict(expected, bogus_profile_field=1)),
                1,
            ),
        )
        reject(
            name + ".bad_automatic",
            planned.replace(
                "plan.automatic_spm = true", "plan.automatic_spm = false", 1
            ),
        )
        reject(
            name + ".bad_automatic_type",
            planned.replace(
                "plan.automatic_spm = true", "plan.automatic_spm = 1 : i64", 1
            ),
        )
        # Missing optional debug state means enabled, for plan compatibility.
        compatible = planned.replace("plan.automatic_spm = true, ", "", 1)
        result = invoke(
            name + ".default_automatic", compatible, "--bosc-ame-plan-verify"
        )
        assert result.returncode == 0, result.stderr

    def emit(
        name,
        text,
        *,
        route,
        post_add=False,
        trips=6,
        rows=32,
        extra_observer=False,
        full_banks=False,
        extra_options="",
        initial=0,
        c_offset=0,
        k_trips=None,
    ):
        result = invoke(name, text, INTEGRATION + extra_options)
        assert result.returncode == 0, (name, result.stderr)
        emitted = result.stdout
        assert "bosc_ame.register_tasks" not in emitted, name
        assert "bosc_ame.tile.load_acc" in emitted, name
        assert "bosc_ame.tile.store_acc" in emitted, name
        if route == "spm":
            assert emitted.count("bosc_ame.spm.view") == 1, name
            assert "bosc_ame.ddr.frame" not in emitted, name
        else:
            assert (
                f"bosc_ame.ddr.frame : memref<{rows * 64 * 4}xi8>" in emitted
            ), name
        path = work / f"{name}.emitted.mlir"
        path.write_text(emitted)
        execute_outputs(
            path,
            trips,
            post_add=post_add,
            repeats=2,
            rows=rows,
            extra_observer=extra_observer,
            full_banks=full_banks,
            storage=route,
            initial=initial,
            c_offset=c_offset,
            k_trips=k_trips,
        )
        return emitted

    for post_add in (False, True):
        mode = "normal" if post_add else "overwrite"
        for trips in (6, 24):
            name = f"{mode}_hot_p_k{trips}"
            original = source(trips, post_add=post_add)
            planned = plan(name, original)
            check_metadata(
                name,
                planned,
                expected_hot(trips, post_add=post_add),
                forge=True,
            )
            emitted = emit(
                name + ".automatic_spm",
                original,
                route="spm",
                post_add=post_add,
                trips=trips,
            )
            if post_add:
                loads = [
                    line
                    for line in emitted.splitlines()
                    if "vector.load " in line
                ]
                assert len(loads) == 2, loads
                assert sum("#bosc_ame.spm<0>" in line for line in loads) == 1
                assert sum("bosc_ame.direct_ddr" in line for line in loads) == 1
                assert "bosc_ame.cpu_ddr_policy" in emitted
                lowered = invoke(
                    name + ".automatic_spm.emitted", emitted, "--lower-bosc-ame"
                )
                assert lowered.returncode == 0, lowered.stderr
                assert "llvm.inline_asm has_side_effects" in lowered.stdout
                assert "csrw 0xbc0" in lowered.stdout
                for policy in (0x0F0F0F0F0F0F0F0F, 0x0F0F000F000F000F):
                    assert (
                        f"llvm.mlir.constant({policy} : i64)" in lowered.stdout
                    )
                assert "llvm.call @bosc_ame_ddr_" not in lowered.stdout
                (work / f"{name}.automatic_spm.llvm.mlir").write_text(
                    lowered.stdout
                )
                ordinary = re.search(
                    r'plan.buffer = \{[^{}]*reason = "reuse-benefit-unproven"[^{}]*\}',
                    planned,
                )
                assert ordinary, planned
                forged = ordinary[0].replace(
                    'level = "ddr"',
                    'bank = 1 : i64, level = "spm", offset = 0 : i64',
                )
                reject(
                    name + ".bad_old_c_spm",
                    planned.replace(ordinary[0], forged, 1),
                )
            # A measured placement cannot be combined with a CPU recipe or
            # an alternate AME tile even if both are otherwise legal routes.
            route = next(
                line for line in planned.splitlines() if "plan.route =" in line
            )
            result = re.search(r"plan.route = (\{.*\}), cast =", route)
            assert result, route
            route_attribute = result[0].removesuffix(", cast =")
            bindings = []
            for endpoint in ("a", "b", "accumulator"):
                binding = re.search(endpoint + r" = (\{[^{}]+\})", route)
                assert binding, route
                bindings.append(
                    re.sub(
                        r'reason = "[^"]+"',
                        'reason = "ordered-matmul-operand"',
                        binding[1],
                    )
                )
            scalar_recipe = (
                'plan.compute = {engine = "scalar", kind = "matmul", '
                'order = "mnk", fp_policy = "preserve", '
                "shape = array<i64: 32, 64, 128>, operands = ["
                + ", ".join(bindings)
                + "]}"
            )
            scalar = route.replace(route_attribute, scalar_recipe)
            reject(
                name + ".bad_compute_scalar", planned.replace(route, scalar, 1)
            )
            for column in (16, 32):
                forged = route.replace(
                    "subtiles = {k = 2 : i64, n = 1 : i64}",
                    f"subtiles = {{k = 2 : i64, n = {64 // column} : i64}}",
                )
                forged = forged.replace(
                    "tile = {k = 64 : i64, m = 32 : i64, n = 64 : i64}",
                    f"tile = {{k = 64 : i64, m = 32 : i64, n = {column} : i64}}",
                )
                assert forged != route, route
                reject(
                    name + f".bad_compute_n{column}",
                    planned.replace(route, forged, 1),
                )

        original = source(6, post_add=post_add, capacity=True)
        fallback = emit(
            mode + ".full_banks",
            original,
            route="ddr",
            post_add=post_add,
            full_banks=True,
        )
        assert fallback.count("bosc_ame.spm.view") == 2
        report = invoke(
            mode + ".full_banks.report",
            original,
            INTEGRATION + " plan-report=true",
        )
        assert report.returncode == 0, report.stderr
        assert 'reason = "hot-accumulator-no-supported-window"' in report.stderr
        if post_add:
            old_fixed = fixed_old_spm(source(post_add=True))
            result = emit(
                mode + ".old_c_bank1", old_fixed, route="ddr", post_add=True
            )
            assert result.count("bosc_ame.spm.view") == 1
            report = invoke(
                mode + ".old_c_bank1.report",
                old_fixed,
                INTEGRATION + " plan-report=true",
            )
            assert report.returncode == 0, report.stderr
            assert (
                'reason = "hot-accumulator-no-supported-window"'
                in report.stderr
            )

        for label, kwargs in (
            ("unmeasured_trips", {"trips": 7}),
            ("unmeasured_shape", {"rows": 16}),
            ("unmeasured_observer", {"extra_observer": True}),
        ):
            trips, rows = kwargs.get("trips", 6), kwargs.get("rows", 32)
            extra = kwargs.get("extra_observer", False)
            original = source(post_add=post_add, **kwargs)
            name = mode + "." + label
            planned = plan(name, original)
            check_metadata(
                name,
                planned,
                expected_hot(
                    trips,
                    post_add=post_add,
                    rows=rows,
                    extra_observer=extra,
                    reason="hot-accumulator-no-measured-profile",
                ),
            )
            emit(
                name,
                original,
                route="ddr",
                post_add=post_add,
                trips=trips,
                rows=rows,
                extra_observer=extra,
            )

        # Whole-function closure is part of the timing key. Initializer,
        # external-offset or pure-work changes must not borrow another recipe.
        nonzero = (
            source(post_add=post_add)
            .replace(
                "    %c = memref.alloc()",
                "    %initial = arith.constant 37 : i32\n    %c = memref.alloc()",
                1,
            )
            .replace(
                "linalg.fill ins(%zero32 : i32) outs(%c",
                "linalg.fill ins(%initial : i32) outs(%c",
                1,
            )
        )
        shifted = source(post_add=post_add).replace(
            "%out_full = memref.reinterpret_cast %arg2 to offset: [0]",
            "%out_full = memref.reinterpret_cast %arg2 to offset: [16]",
            1,
        )
        shifted = shifted.replace(
            "strided<[768, 1]>>", "strided<[768, 1], offset: 16>>"
        )
        extra_work = source(post_add=post_add).replace(
            "    return",
            "    %extra_work = arith.addi %one32, %block32 : i32\n    return",
            1,
        )
        padding = source(post_add=post_add).replace(
            "linalg.fill ins(%zero32 : i32) outs(%c",
            "linalg.fill {contract.fill = {padding = true}} ins(%zero32 : i32) outs(%c",
            1,
        )
        for label, text, initial, c_offset in (
            ("nonzero_initial", nonzero, 37, 0),
            ("external_offset", shifted, 0, 16),
            ("extra_pure_work", extra_work, 0, 0),
            ("padding_contract", padding, 0, 0),
        ):
            name = mode + "." + label
            planned = plan(name, text)
            check_metadata(
                name,
                planned,
                expected_hot(
                    6,
                    post_add=post_add,
                    reason="hot-accumulator-no-measured-profile",
                ),
            )
            emit(
                name,
                text,
                route="ddr",
                post_add=post_add,
                initial=initial,
                c_offset=c_offset,
            )

        # Same mathematical K768, but twelve K64 source steps have not been
        # measured as the six K128 adapter and cannot inherit its cost record.
        fine_k = (
            source(post_add=post_add)
            .replace("128", "64")
            .replace(
                "%trips = arith.constant 6 : i32",
                "%trips = arith.constant 12 : i32",
            )
        )
        fine_k = fine_k.replace("memref<8192xi8>", "memref<4096xi8>")
        planned = plan(mode + ".unmeasured_k_tile", fine_k)
        check_metadata(
            mode + ".unmeasured_k_tile",
            planned,
            expected_hot(
                12,
                post_add=post_add,
                reason="hot-accumulator-no-measured-profile",
            ),
        )
        emit(
            mode + ".unmeasured_k_tile",
            fine_k,
            route="ddr",
            post_add=post_add,
            k_trips=12,
        )

        no_abi = source(post_add=post_add).replace(
            "contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}",
            "",
        )
        planned = plan(mode + ".no_abi", no_abi)
        check_metadata(
            mode + ".no_abi",
            planned,
            expected_hot(
                6,
                post_add=post_add,
                reason="hot-accumulator-no-measured-profile",
            ),
        )
        disabled = invoke(
            mode + ".debug_disabled.report",
            source(post_add=post_add),
            INTEGRATION + " buffer-placement=false plan-report=true",
        )
        assert disabled.returncode == 0, disabled.stderr
        assert 'reason = "hot-accumulator-debug-disabled"' in disabled.stderr
        emit(
            mode + ".debug_disabled",
            source(post_add=post_add),
            route="ddr",
            post_add=post_add,
            extra_options=" buffer-placement=false",
        )

        # The unique terminal output restores ACC fusion before SPM selection.
        unique = source(post_add=post_add, observe=False)
        planned = plan(mode + ".acc_priority", unique)
        assert 'kind = "integer-ame-register"' in planned
        assert 'role = "hot-accumulator"' not in planned
        result = invoke(mode + ".acc_priority", unique, INTEGRATION)
        assert result.returncode == 0, result.stderr
        assert "bosc_ame.register_tasks" in result.stdout
        assert "bosc_ame.spm.view" not in result.stdout
        assert "bosc_ame.ddr.frame" not in result.stdout

        ordered = source(post_add=post_add).replace(
            "linalg.matmul {cast = #linalg.type_fn<cast_signed>}",
            "linalg.matmul {cast = #linalg.type_fn<cast_signed>, "
            'contract.compute = {kind = "matmul", order = "mnk"}}',
        )
        result = invoke(mode + ".ordered_reference", ordered, INTEGRATION)
        assert result.returncode == 0, result.stderr
        assert "bosc_ame.spm.view" not in result.stdout
        assert "bosc_ame.tile." not in result.stdout
        assert "bosc_ame.ddr.frame" not in result.stdout

    pure = invoke(
        "pure_cpu_copy",
        """
module attributes {contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}} {
  func.func @copy(%input: memref<16xi32>, %output: memref<16xi32>) {
    memref.copy %input, %output : memref<16xi32> to memref<16xi32>
    return
  }
}
""",
        INTEGRATION,
        "--lower-bosc-ame",
    )
    assert pure.returncode == 0, pure.stderr
    assert "bosc_ame.cpu_ddr_policy" not in pure.stdout
    assert "csrw 0xbc0" not in pure.stdout


if __name__ == "__main__":
    main()
