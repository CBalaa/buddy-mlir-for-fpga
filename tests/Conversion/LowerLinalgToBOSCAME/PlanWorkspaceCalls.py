# RUN: %PYTHON %s %t
"""Reject calls that can introduce physical aliases across SPM workspaces."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_check import buddy_opt_path  # noqa: E402

DIAGNOSTIC = (
    "call may clobber live SPM workspace; v1 has no "
    "interprocedural workspace proof"
)
OWNERSHIP = (
    "SPM function has module-local symbol uses; v1 has no "
    "interprocedural workspace ownership proof"
)

PRIVATE_WORKSPACE = """
module {
  func.func @workspace() {
    %seven = arith.constant 7 : i32
    %a = memref.alloc() alignment = 64 : memref<64xi32>
    linalg.fill ins(%seven : i32) outs(%a : memref<64xi32>)
    memref.dealloc %a : memref<64xi32>
    return
  }
}
"""

NESTED_WORKSPACE = """
#id = affine_map<(d0) -> (d0)>
module attributes {
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
} {
  func.func private @helper() {
    %v = arith.constant 99 : i32
    %b = memref.alloc() alignment = 64 : memref<64xi32>
    linalg.fill ins(%v : i32) outs(%b : memref<64xi32>)
    memref.dealloc %b : memref<64xi32>
    return
  }
  func.func @caller(%input: memref<64xi32>, %out: memref<64xi32>) {
    %v = arith.constant 7 : i32
    %a = memref.alloc() alignment = 64 : memref<64xi32>
    %c = memref.alloc() alignment = 64 : memref<64xi32>
    linalg.fill ins(%v : i32) outs(%a : memref<64xi32>)
    func.call @helper() : () -> ()
    linalg.generic {indexing_maps = [#id, #id, #id],
                    iterator_types = ["parallel"]}
        ins(%a, %input : memref<64xi32>, memref<64xi32>)
        outs(%c : memref<64xi32>) {
      ^bb0(%left: i32, %right: i32, %old: i32):
        %sum = arith.addi %left, %right : i32
        linalg.yield %sum : i32
    }
    memref.copy %c, %out : memref<64xi32> to memref<64xi32>
    memref.dealloc %a : memref<64xi32>
    memref.dealloc %c : memref<64xi32>
    return
  }
}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def invoke(name, text, *passes):
        path = work / f"{name}.mlir"
        path.write_text(text)
        return subprocess.run(
            [str(buddy_opt_path()), str(path), *passes],
            text=True,
            capture_output=True,
        )

    def rejected(name, text, *passes):
        result = invoke(name, text, *passes)
        assert result.returncode != 0, (name, result.stdout)
        assert DIAGNOSTIC in result.stderr, (name, result.stderr)
        assert "LLVM ERROR" not in result.stderr, (name, result.stderr)

    def referenced(name, text, *passes):
        result = invoke(name, text, *passes)
        assert result.returncode != 0, (name, result.stdout)
        assert OWNERSHIP in result.stderr, (name, result.stderr)
        assert "LLVM ERROR" not in result.stderr, (name, result.stderr)

    # The helper's fresh logical allocation must not overwrite live caller A.
    rejected(
        "callee_private_workspace",
        NESTED_WORKSPACE,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
    )
    opaque = PRIVATE_WORKSPACE.replace(
        "  func.func @workspace",
        "  func.func private @opaque()\n  func.func @workspace",
    ).replace(
        "    memref.dealloc",
        "    func.call @opaque() : () -> ()\n    memref.dealloc",
    )
    rejected(
        "opaque_callee",
        opaque,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
    )
    indirect = PRIVATE_WORKSPACE.replace(
        "  func.func @workspace",
        "  func.func private @opaque()\n  func.func @workspace",
    ).replace(
        "    memref.dealloc",
        "    %callee = func.constant @opaque : () -> ()\n"
        "    func.call_indirect %callee() : () -> ()\n    memref.dealloc",
    )
    rejected(
        "indirect_callee",
        indirect,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
    )

    # Physical input adapters must obey the same rule for existing windows.
    fixed = opaque.replace(
        "%a = memref.alloc() alignment = 64 : memref<64xi32>",
        "%a = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} "
        ": memref<64xi32, #bosc_ame.spm<0>>",
    ).replace(
        "outs(%a : memref<64xi32>)",
        "outs(%a : memref<64xi32, #bosc_ame.spm<0>>)",
    )
    fixed = fixed.replace("memref.dealloc %a : memref<64xi32>", "")
    for entry in ("--linalg-to-boscame", "--lower-linalg-to-boscame"):
        rejected("fixed_" + entry[2:], fixed, entry)

    # An explicitly supplied plan cannot bypass the check at verify or emit.
    baseline = invoke(
        "private_workspace",
        PRIVATE_WORKSPACE,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
    )
    assert baseline.returncode == 0, baseline.stderr
    # A DDR-only caller can invoke the SPM callee concurrently even though it
    # has no local workspace for the old caller-side check to notice. V1 also
    # rejects serial and indirect references without a callee ownership proof.
    callers = {
        "parallel": """
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      func.call @workspace() : () -> ()
      scf.reduce
    }
    return
  }
""",
        "serial": """
  func.func @entry() {
    func.call @workspace() : () -> ()
    return
  }
""",
        "address": """
  func.func @entry() {
    %callee = func.constant @workspace : () -> ()
    return
  }
""",
        "indirect": """
  func.func @entry() {
    %callee = func.constant @workspace : () -> ()
    func.call_indirect %callee() : () -> ()
    return
  }
""",
    }
    for name, caller in callers.items():
        text = PRIVATE_WORKSPACE.rsplit("}", 1)[0] + caller + "}\n"
        referenced(name, text, "--kernel-contract-annotate", "--bosc-ame-plan")
        for entry in ("--linalg-to-boscame", "--lower-linalg-to-boscame"):
            referenced(name + entry[2:], text, entry)
    addressed_plan = baseline.stdout.replace(
        "    memref.dealloc",
        "    %callee = func.constant @workspace : () -> ()\n    memref.dealloc",
    )
    for entry in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        referenced("addressed_plan" + entry[2:], addressed_plan, entry)

    planned = baseline.stdout.replace(
        "  func.func @workspace",
        "  func.func private @opaque()\n  func.func @workspace",
    ).replace(
        "    memref.dealloc",
        "    func.call @opaque() : () -> ()\n    memref.dealloc",
    )
    planned = planned.replace(
        "complete = true, root_scope",
        'complete = false, limitation = "opaque call", root_scope',
    ).replace(
        "source_lineage_complete = true", "source_lineage_complete = false"
    )
    for entry in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        rejected("planned_" + entry[2:], planned, entry)

    # A function without private SPM cannot lose its contents to a callee.
    ddr_only = """
module {
  func.func private @opaque(memref<64xi32>)
  func.func @ddr_only(%out: memref<64xi32>) {
    func.call @opaque(%out) : (memref<64xi32>) -> ()
    return
  }
}
"""
    result = invoke(
        "ddr_only",
        ddr_only,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
        "--bosc-ame-plan-verify",
        "--bosc-ame-emit",
    )
    assert result.returncode == 0, result.stderr
    assert "call @opaque" in result.stdout
    print(
        "PASS private/fixed SPM calls reject; explicit plans and DDR-only calls checked"
    )


if __name__ == "__main__":
    main()
