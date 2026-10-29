# RUN: %PYTHON %s %t
"""Check virtual content versions against independently materialized execution.

The oracle lowers the original buffers with ordinary linalg-to-loops. It does
not use the stream planner's expressions, padding selection or task mappings.
"""

import math
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402

MODULE = """
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
module attributes {contract.abi = {args_aligned_bytes = 64 : i64,
                                  args_disjoint = true}} {
  func.func @row(%x: memref<24xf32>, %out: memref<24xf32>) {
    %zero = arith.constant 0.0 : f32
    %width = arith.constant 24.0 : f32
    %eps = arith.constant 1.0e-12 : f32
    %c0 = arith.constant 0 : index
    %tile = memref.alloc() : memref<32xf32>
    linalg.fill ins(%zero : f32) outs(%tile : memref<32xf32>)
    %valid = memref.subview %tile[0] [24] [1]
      : memref<32xf32> to memref<24xf32, strided<[1]>>
    memref.copy %x, %valid : memref<24xf32> to memref<24xf32, strided<[1]>>
    %sum = memref.alloc() : memref<f32>
    linalg.fill ins(%zero : f32) outs(%sum : memref<f32>)
    linalg.reduce ins(%tile : memref<32xf32>) outs(%sum : memref<f32>)
      dimensions = [0] (%a: f32, %b: f32) {
        %v = arith.addf %a, %b : f32
        linalg.yield %v : f32
      }
    %s = memref.load %sum[] : memref<f32>
    %mean = arith.divf %s, %width : f32
    linalg.generic {indexing_maps = [#id, #scalar, #id],
                    iterator_types = ["parallel"]}
      ins(%tile, %mean : memref<32xf32>, f32) outs(%tile : memref<32xf32>) {
      ^bb0(%a: f32, %m: f32, %old: f32):
        %v = arith.subf %a, %m : f32
        linalg.yield %v : f32
    }
    %tail = memref.subview %tile[24] [8] [1]
      : memref<32xf32> to memref<8xf32, strided<[1], offset: 24>>
    linalg.fill ins(%zero : f32) outs(%tail : memref<8xf32, strided<[1], offset: 24>>)
    %squares = memref.alloc() : memref<32xf32>
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%tile : memref<32xf32>) outs(%squares : memref<32xf32>) {
      ^bb0(%a: f32, %old: f32):
        %v = arith.mulf %a, %a : f32
        linalg.yield %v : f32
    }
    %var = memref.alloc() : memref<f32>
    linalg.fill ins(%zero : f32) outs(%var : memref<f32>)
    linalg.reduce ins(%squares : memref<32xf32>) outs(%var : memref<f32>)
      dimensions = [0] (%a: f32, %b: f32) {
        %v = arith.addf %a, %b : f32
        linalg.yield %v : f32
      }
    %variance_sum = memref.load %var[] : memref<f32>
    %variance = arith.divf %variance_sum, %width : f32
    %biased = arith.addf %variance, %eps : f32
    %invstd = math.rsqrt %biased : f32
    %result = memref.alloc() : memref<32xf32>
    linalg.generic {indexing_maps = [#id, #scalar, #id],
                    iterator_types = ["parallel"]}
      ins(%tile, %invstd : memref<32xf32>, f32) outs(%result : memref<32xf32>) {
      ^bb0(%a: f32, %inv: f32, %old: f32):
        %v = arith.mulf %a, %inv : f32
        linalg.yield %v : f32
    }
    %outvalid = memref.subview %result[0] [24] [1]
      : memref<32xf32> to memref<24xf32, strided<[1]>>
    memref.copy %outvalid, %out
      : memref<24xf32, strided<[1]>> to memref<24xf32>
    return
  }

  func.func @quant(%x: memref<24xf32>, %q: memref<24xi8>, %scaleout: memref<1xf32>) {
    %zero = arith.constant 0.0 : f32
    %neg_inf = arith.constant 0xFF800000 : f32
    %one = arith.constant 1.0 : f32
    %limit = arith.constant 127.0 : f32
    %nlimit = arith.constant -127.0 : f32
    %half = arith.constant 0.5 : f32
    %p1 = arith.constant 1 : i32
    %n1 = arith.constant -1 : i32
    %i0 = arith.constant 0 : i32
    %tile = memref.alloc() : memref<32xf32>
    linalg.fill ins(%zero : f32) outs(%tile : memref<32xf32>)
    %valid = memref.subview %tile[0] [24] [1]
      : memref<32xf32> to memref<24xf32, strided<[1]>>
    memref.copy %x, %valid : memref<24xf32> to memref<24xf32, strided<[1]>>
    %absolute = memref.alloc() : memref<32xf32>
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%tile : memref<32xf32>) outs(%absolute : memref<32xf32>) {
      ^bb0(%a: f32, %old: f32):
        %v = math.absf %a : f32
        linalg.yield %v : f32
    }
    %maximum = memref.alloc() : memref<f32>
    linalg.fill ins(%neg_inf : f32) outs(%maximum : memref<f32>)
    linalg.reduce ins(%absolute : memref<32xf32>) outs(%maximum : memref<f32>)
      dimensions = [0] (%a: f32, %b: f32) {
        %v = arith.maxnumf %a, %b : f32
        linalg.yield %v : f32
      }
    %max = memref.load %maximum[] : memref<f32>
    %iszero = arith.cmpf oeq, %max, %zero : f32
    %quotient = arith.divf %max, %limit : f32
    %scale = arith.select %iszero, %one, %quotient : f32
    %scale_tile = memref.alloc() : memref<1xf32>
    linalg.fill ins(%scale : f32) outs(%scale_tile : memref<1xf32>)
    memref.copy %scale_tile, %scaleout : memref<1xf32> to memref<1xf32>
    %result = memref.alloc() : memref<32xi8>
    linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel"]}
      ins(%tile, %scale : memref<32xf32>, f32) outs(%result : memref<32xi8>) {
      ^bb0(%a: f32, %s: f32, %old: i8):
        %div = arith.divf %a, %s : f32
        %lo = arith.maxnumf %div, %nlimit : f32
        %clip = arith.minnumf %lo, %limit : f32
        %pos = arith.cmpf oge, %clip, %zero : f32
        %sign = arith.select %pos, %p1, %n1 : i32
        %base = arith.fptosi %clip : f32 to i32
        %basef = arith.sitofp %base : i32 to f32
        %frac = arith.subf %clip, %basef : f32
        %absfrac = math.absf %frac : f32
        %away = arith.cmpf oge, %absfrac, %half : f32
        %adjust = arith.select %away, %sign, %i0 : i32
        %rounded = arith.addi %base, %adjust : i32
        %v = arith.trunci %rounded : i32 to i8
        linalg.yield %v : i8
    }
    %outvalid = memref.subview %result[0] [24] [1]
      : memref<32xi8> to memref<24xi8, strided<[1]>>
    memref.copy %outvalid, %q : memref<24xi8, strided<[1]>> to memref<24xi8>
    return
  }

  func.func @singleton(%x: memref<16xf32>, %out: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %source = memref.subview %x[%i] [1] [1]
        : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
      %load_tile = memref.alloc() : memref<1xf32>
      memref.copy %source, %load_tile
        : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %v = memref.load %load_tile[%c0] : memref<1xf32>
      %store_tile = memref.alloc() : memref<1xf32>
      linalg.fill ins(%v : f32) outs(%store_tile : memref<1xf32>)
      %target = memref.subview %out[%i] [1] [1]
        : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
      memref.copy %store_tile, %target
        : memref<1xf32> to memref<1xf32, strided<[1], offset: ?>>
    }
    return
  }
}
"""

# Keep the 24-of-32 case as an independent DDR fallback regression. The
# stream-positive fixture uses a 32-of-48 boundary, aligned to logical chunks.
UNALIGNED = MODULE
MODULE = (
    MODULE.replace("memref<32x", "memref<48x")
    .replace("24", "32")
    .replace("memref<8xf32", "memref<16xf32")
    .replace("[8]", "[16]")
)

SNAPSHOT = """
module attributes {contract.abi = {args_aligned_bytes = 64 : i64,
                                  args_disjoint = true}} {
  func.func @snapshot(%x: memref<16xf32>, %out: memref<16xf32>) {
    %zero = arith.constant 0.0 : f32
    %saved = memref.alloc() : memref<16xf32>
    memref.copy %x, %saved : memref<16xf32> to memref<16xf32>
    linalg.fill ins(%zero : f32) outs(%x : memref<16xf32>)
    memref.copy %saved, %out : memref<16xf32> to memref<16xf32>
    return
  }
}
"""

MIXED = """
module attributes {contract.abi = {args_aligned_bytes = 64 : i64,
                                  args_disjoint = true}} {
  func.func @mixed(%a: memref<4x64xi8>,
                   %b: memref<64x64xi8, strided<[1, 64]>>,
                   %c: memref<4x64xi32>, %x: memref<16xf32>,
                   %result: memref<1xf32>) {
    %zero = arith.constant 0.0 : f32
    %zero32 = arith.constant 0 : i32
    %saved = memref.alloc() : memref<16xf32>
    memref.copy %x, %saved : memref<16xf32> to memref<16xf32>
    %sum = memref.alloc() : memref<f32>
    linalg.fill ins(%zero : f32) outs(%sum : memref<f32>)
    %acc = memref.alloc() alignment = 64 : memref<4x64xi32>
    linalg.fill ins(%zero32 : i32) outs(%acc : memref<4x64xi32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<4x64xi8>, memref<64x64xi8, strided<[1, 64]>>)
      outs(%acc : memref<4x64xi32>)
    memref.copy %acc, %c : memref<4x64xi32> to memref<4x64xi32>
    linalg.reduce ins(%saved : memref<16xf32>) outs(%sum : memref<f32>)
      dimensions = [0] (%lhs: f32, %rhs: f32) {
        %v = arith.addf %lhs, %rhs : f32
        linalg.yield %v : f32
      }
    %s = memref.load %sum[] : memref<f32>
    %single = memref.alloc() : memref<1xf32>
    linalg.fill ins(%s : f32) outs(%single : memref<1xf32>)
    memref.copy %single, %result : memref<1xf32> to memref<1xf32>
    return
  }
}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def invoke(name, source, *passes):
        path = work / f"{name}.mlir"
        path.write_text(source)
        return subprocess.run(
            [str(buddy_opt_path()), str(path), *passes],
            capture_output=True,
            text=True,
        )

    def compile_source(name, source):
        planned = invoke(
            name, source, "--kernel-contract-annotate", "--bosc-ame-plan"
        )
        assert planned.returncode == 0, planned.stderr
        emitted = invoke(name + "_planned", planned.stdout, "--bosc-ame-emit")
        assert emitted.returncode == 0, emitted.stderr
        emitted_path = work / f"{name}_emitted.mlir"
        emitted_path.write_text(emitted.stdout)
        baseline = invoke(
            name + "_original", source, "--convert-linalg-to-loops"
        )
        assert baseline.returncode == 0, baseline.stderr
        baseline_path = work / f"{name}_baseline.mlir"
        baseline_path.write_text(baseline.stdout)
        return planned.stdout, emitted.stdout, emitted_path, baseline_path

    planned, emitted, new_path, old_path = compile_source("streams", MODULE)
    assert "plan.version = 2" in planned
    assert "plan.stream_virtual" in planned
    assert "plan.stream =" in planned
    assert "memref.alloc" not in emitted
    assert (
        "bosc_ame.spm.view" not in emitted
        and "bosc_ame.dma.start" not in emitted
    )
    assert "vector.reduction <add>" in emitted
    assert "vector.reduction <maxnumf>" in emitted
    assert "bosc_ame.stream_task_map" in emitted
    assert "bosc_ame.execution_id" in emitted
    assert "contract.task =" not in emitted
    assert (
        "vector.insert" not in emitted and "vector.from_elements" not in emitted
    )
    assert "vector_bounds = array<i64: 0, 32, 48>" in planned

    # A forged recipe or missing component member must fail before mutation.
    bad = re.sub(r"lanes = 16 : i64", "lanes = 8 : i64", planned, count=1)
    assert invoke("bad_lanes", bad, "--bosc-ame-emit").returncode != 0
    bad = planned.replace(", plan.stream_virtual", "", 1)
    assert bad != planned
    assert invoke("missing_virtual", bad, "--bosc-ame-emit").returncode != 0
    bad = re.sub(
        r"logical_inputs = array<i64[^>]*>",
        "logical_inputs = array<i64: 99999>",
        planned,
        count=1,
    )
    assert invoke("bad_dependencies", bad, "--bosc-ame-emit").returncode != 0
    bad = planned.replace(
        "vector_bounds = array<i64: 0, 32, 48>",
        "vector_bounds = array<i64: 0, 16, 48>",
        1,
    )
    assert bad != planned
    assert invoke("bad_vector_bounds", bad, "--bosc-ame-emit").returncode != 0

    def compare(
        function, values, outputs, new_module=new_path, old_module=old_path
    ):
        source = bytearray(struct.pack(f"<{len(values)}f", *values))
        source += bytearray([0xB7]) * 64
        original = bytes(source)
        arguments = {"%arg0": source}
        for i, size in enumerate(outputs, 1):
            arguments[f"%arg{i}"] = bytearray([0xA3]) * (size + 64)
        reference = {key: bytearray(value) for key, value in arguments.items()}
        new = Machine(new_module, function)
        old = Machine(old_module, function)
        new.launch(arguments)
        old.launch(reference)
        assert bytes(arguments["%arg0"]) == original
        for i, size in enumerate(outputs, 1):
            key = f"%arg{i}"
            assert arguments[key] == reference[key], (function, values, key)
            assert arguments[key][size:] == bytearray([0xA3]) * 64
        assert not new.dma_log
        return new, arguments

    # Cancellation and nonzero mean expose a wrong content-version or tail
    # policy. A zero-variance row also checks exact scalar rsqrt sequencing.
    for values in (
        [float(i % 7 - 1) for i in range(32)],
        [1.0] * 32,
        [16777216.0, 1.0, -16777216.0, -0.0] * 8,
    ):
        machine, _ = compare("row", values, [32 * 4])
        reads = sum(
            end - begin
            for name, begin, end, kind in machine.accesses
            if name == "%arg0" and kind == "read"
        )
        assert reads == 3 * 32 * 4, reads

    # Scale=1 fixes half-integer cases exactly; the adjacent f32 numbers must
    # not use the non-equivalent "add 0.5 then truncate" shortcut.
    below = struct.unpack("<f", struct.pack("<I", 0x3EFFFFFF))[0]
    above = struct.unpack("<f", struct.pack("<I", 0x3F000001))[0]
    for values in (
        [127.0, -127.0, 0.5, -0.5, below, -below, above, -above] * 4,
        [0.0, -0.0] * 16,
        [math.nan] * 32,
        [math.inf, -math.inf, 1.0, -1.0] * 8,
    ):
        machine, _ = compare("quant", values, [32, 4])
        reads = sum(
            end - begin
            for name, begin, end, kind in machine.accesses
            if name == "%arg0" and kind == "read"
        )
        assert reads == 2 * 32 * 4, reads

    compare("singleton", [float(i - 5) for i in range(16)], [16 * 4])

    # A group spanning a 24-of-32 virtual boundary would need unsupported
    # lane inserts or an extra packing buffer. Retain the ordinary DDR recipe
    # and its ordered 16-lane reduction, including exact nonfinite behavior.
    tail_plan, _, tail_new, tail_old = compile_source("unaligned", UNALIGNED)
    for function in ("row", "quant"):
        body = tail_plan.split(f"func.func @{function}(", 1)[1]
        body = body.split("\n  func.func @", 1)[0]
        assert "plan.stream_virtual" not in body
        assert 'level = "ddr"' in body
    for values in (
        [float(i % 7 - 1) for i in range(24)],
        [1.0] * 24,
        [16777216.0, 1.0, -16777216.0, -0.0] * 6,
    ):
        compare("row", values, [24 * 4], tail_new, tail_old)
    for values in (
        [127.0, -127.0, 0.5, -0.5, below, -below, above, -above] * 3,
        [0.0, -0.0] * 12,
        [math.nan] * 24,
        [math.inf, -math.inf, 1.0, -1.0] * 6,
    ):
        compare("quant", values, [24, 4], tail_new, tail_old)

    # The same root is overwritten after tt.load's logical snapshot. Delaying
    # the read would observe zeros, so the temporary must stay materialized.
    snap_plan, _, snap_new, snap_old = compile_source("snapshot", SNAPSHOT)
    assert "plan.stream_virtual" not in snap_plan
    for path in (snap_new, snap_old):
        source = bytearray(struct.pack("<16f", *range(16)))
        output = bytearray(64)
        Machine(path, "snapshot").launch({"%arg0": source, "%arg1": output})
        assert output == struct.pack("<16f", *range(16))
        assert source == bytearray(64)

    aliasing = SNAPSHOT.replace("args_disjoint = true", "args_disjoint = false")
    aliasing = aliasing.replace(
        "    linalg.fill ins(%zero : f32) outs(%x : memref<16xf32>)\n", ""
    )
    alias_plan, _, _, _ = compile_source("possibly_aliasing", aliasing)
    assert "plan.stream_virtual" not in alias_plan
    assert 'level = "ddr"' in alias_plan

    unknown_call = SNAPSHOT.replace(
        "  func.func @snapshot(",
        "  func.func private @clobber(memref<16xf32>)\n  func.func @snapshot(",
    ).replace(
        "    linalg.fill ins(%zero : f32) outs(%x : memref<16xf32>)",
        "    func.call @clobber(%x) : (memref<16xf32>) -> ()",
    )
    unknown = invoke(
        "unknown_call",
        unknown_call,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
    )
    assert unknown.returncode == 0, unknown.stderr
    assert "plan.stream_virtual" not in unknown.stdout

    # The snapshot source is disjoint from a register-resident matmul between
    # its copy and reduction. Its stability proof belongs to the logical
    # module: expanding the matmul first must not turn its physical execution
    # region into an unknown-effect barrier, or select newly eligible roots.
    mixed_plan, mixed_emit, mixed_path, _ = compile_source("mixed", MIXED)
    assert "plan.register_schedule" in mixed_plan
    assert "plan.stream_virtual" in mixed_plan
    assert "bosc_ame.register_task_map" in mixed_emit
    assert "bosc_ame.stream_task_map" in mixed_emit
    assert "memref.alloc" not in mixed_emit
    assert "contract.task =" not in mixed_emit
    a = bytearray((i * 37 + 11) % 256 for i in range(4 * 64))
    b = bytearray((i * 19 + 7) % 256 for i in range(64 * 64))
    c = bytearray([0xC3]) * (4 * 64 * 4 + 64)
    x = bytearray(struct.pack("<16f", *range(16)))
    result = bytearray([0xD4]) * (4 + 64)
    original_inputs = bytes(a), bytes(b), bytes(x)
    expected = bytearray(c)

    def signed(value):
        return value if value < 128 else value - 256

    for row in range(4):
        for col in range(64):
            value = sum(
                signed(a[row * 64 + k]) * signed(b[col * 64 + k])
                for k in range(64)
            )
            struct.pack_into("<i", expected, 4 * (row * 64 + col), value)
    mixed = Machine(mixed_path, "mixed")
    mixed.launch(
        {"%arg0": a, "%arg1": b, "%arg2": c, "%arg3": x, "%arg4": result}
    )
    assert c == expected
    assert result[:4] == struct.pack("<f", 120.0)
    assert result[4:] == bytearray([0xD4]) * 64
    assert (bytes(a), bytes(b), bytes(x)) == original_inputs
    assert not mixed.dma_log

    print(
        "PASS stream plans: content versions, ordered reductions, virtual padding, "
        "quantization boundaries, singleton DDR, snapshot/alias/call fallback, "
        "mixed execution groups, task maps"
    )


if __name__ == "__main__":
    main()
