# RUN: %PYTHON %s %t
"""Fixed SPM windows require serial definitions and users, including aliases."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_check import buddy_opt_path  # noqa: E402

DIAGNOSTIC = "fixed SPM workspace requires serial execution inside func.func"
SOURCE = """
module {
  func.func @workspace() {
    %zero = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %a = memref.alloc() alignment = 64 : memref<64xi32>
    linalg.fill ins(%zero : i32) outs(%a : memref<64xi32>)
    memref.dealloc %a : memref<64xi32>
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

    loop = """
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      linalg.fill ins(%zero : i32) outs(%a : memref<64xi32>)
      scf.reduce
    }
"""
    captured = SOURCE.replace("    memref.dealloc", loop + "    memref.dealloc")
    allocated = SOURCE.replace(
        "    %a = memref.alloc()",
        "    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {\n"
        "    %a = memref.alloc()",
    ).replace("    return", "      scf.reduce\n    }\n    return")
    alias = captured.replace(
        "    scf.parallel",
        "    %alias = memref.subview %a[0] [64] [1] : memref<64xi32> "
        "to memref<64xi32, strided<[1]>>\n    scf.parallel",
    ).replace(
        "      linalg.fill ins(%zero : i32) outs(%a : memref<64xi32>)",
        "      linalg.fill ins(%zero : i32) "
        "outs(%alias : memref<64xi32, strided<[1]>>)",
    )
    for name, text in (
        ("captured", captured),
        ("allocated", allocated),
        ("alias", alias),
    ):
        for entry in ("--linalg-to-boscame", "--lower-linalg-to-boscame"):
            rejected(name + entry[2:], text, entry)
        rejected(name, text, "--kernel-contract-annotate", "--bosc-ame-plan")

    # Pure scalar loads/stores inside an unmodeled concurrent region do not
    # introduce declared tasks. A parsed incomplete plan must still prove
    # physical workspace ownership before verification or emission succeeds.
    baseline = invoke(
        "baseline", SOURCE, "--kernel-contract-annotate", "--bosc-ame-plan"
    )
    assert baseline.returncode == 0, baseline.stderr
    planned = baseline.stdout.replace(
        "complete = true, root_scope",
        'complete = false, limitation = "unmodeled operation or memory effect", '
        "root_scope",
    )
    planned = planned.replace(
        "source_lineage_complete = true", "source_lineage_complete = false"
    )
    alloc = re.search(r"(%\w+) = memref.alloc", planned).group(1)
    planned = planned.replace(
        "    memref.dealloc",
        "    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {\n"
        f"      %value = memref.load {alloc}[%c0] : memref<64xi32>\n"
        f"      memref.store %value, {alloc}[%c0] : memref<64xi32>\n"
        "      scf.reduce\n    }\n    memref.dealloc",
    )
    for entry in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        rejected("parsed" + entry[2:], planned, entry)

    # Supported serial scopes retain legal placement and synchronous accesses.
    serial = captured.replace(
        "scf.parallel (%i) = (%c0) to (%c2) step (%c1)",
        "scf.for %i = %c0 to %c2 step %c1",
    ).replace("      scf.reduce\n", "")
    result = invoke(
        "serial",
        serial,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
        "--bosc-ame-emit",
    )
    assert result.returncode == 0, result.stderr
    print("PASS SPM definitions and aliases reject concurrent scopes")


if __name__ == "__main__":
    main()
