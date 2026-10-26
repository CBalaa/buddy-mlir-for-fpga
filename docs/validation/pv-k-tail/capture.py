"""Capture current actual PV build hashes and statically audited IR facts.

Run after tools/build.py --case attention_pv_12x197x64x197 --stage asm.
This script does not run the simulator or supply successful execution results.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
sys.path.insert(
    0, str(repo / "examples/FPGA-BOSCAME/vit-base-patch16-224/tools")
)
from oracles import make_inputs, reference  # noqa: E402

from build import ROOT, argument_shapes, describe, digest, tool  # noqa: E402

case = describe("attention_pv_12x197x64x197")
out = ROOT / "build/kernels" / case["name"]
doc = repo / "docs/validation/pv-k-tail"
build = json.loads((out / "build-result.json").read_text())
assert build["status"] == "PASS" and build["stage"] == "asm"
build["artifact_dir"] = str(out.relative_to(repo))
(doc / "build-report.json").write_text(json.dumps(build, indent=2) + "\n")
sources = [
    case["kernel_source"],
    ROOT / "cases.py",
    ROOT / "tools/build.py",
    ROOT / "tools/simcheck.py",
    ROOT / "tools/oracles.py",
    repo
    / "midend/lib/Conversion/TritonToLinalgExperimental/ContractTileSpecialization.cpp",
    repo
    / "midend/lib/Conversion/TritonToLinalgExperimental/ContractTileSpecialization.h",
    repo
    / "midend/lib/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimentalPass.cpp",
    repo / "midend/lib/Analysis/KernelContract.cpp",
    repo / "midend/lib/Conversion/StructuredToMemref/StructuredToMemref.cpp",
    repo / "tests/Conversion/TritonToLinalgExperimental/vit-pv-oracle.py",
    repo
    / "tests/Conversion/TritonToLinalgExperimental/boscame-k-tail-specialization.py",
    repo / "tests/KernelContract/reaching-zero-loop-copy.py",
    repo / "examples/FPGA-BOSCAME/tools/boscame_buffer_sim.py",
    Path(__file__).resolve(),
]
artifacts = {
    str(p.relative_to(repo)): digest(p)
    for p in sorted(out.iterdir())
    if p.name.startswith("kernel.") or p.name in ("adapter.c", "frontend.json")
}
(doc / "artifact-hashes.json").write_text(
    json.dumps(
        {
            "scope": "actual PV frontend through restricted assembly and object; source and compiler-tool hashes",
            "artifacts": artifacts,
            "sources": {
                str(Path(p).relative_to(repo)): digest(p) for p in sources
            },
            "tools": {
                name: digest(tool(name))
                for name in ("buddy-opt", "buddy-translate", "llc", "clang")
            },
        },
        indent=2,
    )
    + "\n"
)
frontend = (out / "kernel.frontend.mlir").read_text()
planned = (out / "kernel.planned.mlir").read_text()
emitted = (out / "kernel.boscame.mlir").read_text()
assembly = (out / "kernel.nr.s").read_text()
llvm = (out / "kernel.ll").read_text()
assert len(re.findall(r"scf.for .* to %c3_i32 step %c1_i32", frontend)) == 2
assert frontend.count("linalg.matmul ") == 4
assert planned.count("shape = array<i64: 16, 16, 64>") == 4
assert "contract.compute_graph = {complete = true" in planned
assert "scope_count = 5 : i64" in planned
assert "plan.source_task_count = 29 : i64" in planned
matrix_lines = [
    line for line in planned.splitlines() if 'engine = "matrix"' in line
]
assert len(matrix_lines) == 4
for rows in (16, 5):
    for valid_k in (64, 5):
        candidates = [
            line
            for line in matrix_lines
            if f"row_bytes = {valid_k * 4} : i64" in line
            and f"rows = {rows} : i64" in line
        ]
        assert len(candidates) == 1
        assert 'reason = "dma-illegal-exact-acc32-rectangle"' in candidates[0]
        assert "source_pitch = 788 : i64" in candidates[0]
        assert "target_pitch = 256 : i64" in candidates[0]
        assert 'register_class = "acc"' in candidates[0]
assert "bosc_ame.tile.mma" not in emitted
assert not re.search(
    r"\b(?:v?f(?:madd|msub|nmadd|nmsub|macc|msac|nmacc|nmsac))", assembly
)
assert not re.search(r"\b(?:fast|contract|reassoc)\b", llvm)
assert re.search(r"high_water = array<i64: 10240, 9216>", planned)
route = {
    "scope": "static audit of actual planned/frontend IR; byte expectations are checked independently by strict simulation",
    "status": "PASS",
    "compute_graph": {
        "complete": True,
        "scope_count": 5,
        "source_task_count": 29,
        "task_map": "29 retained tasks; loop dependencies regenerated",
    },
    "grid": case["grid"],
    "programs": 624,
    "outputs_per_round": 151296,
    "branches": {"full_M_programs": 576, "tail_M_programs": 48},
    "reduction": {
        "prefix_loop_iterations": 3,
        "static_tail_block": 3,
        "tail_valid_K": 5,
        "arithmetic_BK": 64,
        "A_shape": [16, 64],
        "B_shape": [64, 16],
        "order": "each zero-initialized product reduced sequentially over64 K; separate block add; no new FMA",
    },
    "a_routes": [
        {
            "valid_shape": [rows, k],
            "engine": "matrix",
            "register_class": "acc",
            "instruction_bits": 32,
            "source_pitch": 788,
            "target_pitch": 256,
            "row_bytes": k * 4,
            "reason": "dma-illegal-exact-acc32-rectangle",
        }
        for rows in (16, 5)
        for k in (64, 5)
    ],
    "b_route": {
        "engine": "dma",
        "packets_per_program": 197,
        "packet_bytes": 64,
        "source_pitch": 256,
        "tail_rows": 5,
        "reason": "bounded-external-boundary",
    },
    "c_route": {
        "engine": "dma",
        "old_C_read": False,
        "full_M_rows": 16,
        "tail_M_rows": 5,
        "packet_bytes": 64,
        "reason": "bounded-external-boundary",
    },
    "spm_high_water_bytes": [10240, 9216],
    "expected_ddr_bytes_per_program": {"full_M": 26240, "tail_M": 16868},
    "expected_ddr_bytes_full_grid": 15923904,
    "ame_f32_mma_count": 0,
    "new_fma_count": 0,
}
(doc / "route-summary.json").write_text(json.dumps(route, indent=2) + "\n")
refs = {}
for stress in (False, True):
    result = reference(
        case, make_inputs(case, argument_shapes(case), stress=stress)
    )["C"]
    refs["stress" if stress else "normal"] = {
        "outputs": result.size,
        "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
    }
(doc / "reference-hashes.json").write_text(
    json.dumps(
        {
            "numerical_contract": "ordered f32 multiply and add, zero product per full BK64 including positive-zero padded tail, separate block add, SCALE after reduction",
            "references": refs,
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {"build": "PASS", "route_audit": "PASS", "reference_hashes": refs},
        indent=2,
    )
)
