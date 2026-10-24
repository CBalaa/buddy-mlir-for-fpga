# RUN: %PYTHON %s %t
"""Check that fixed SPM addresses cannot escape their logical private roots."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_check import buddy_opt_path  # noqa: E402

DIAGNOSTIC = (
    "SPM root requires non-escaping aliases and completed local memory accesses"
)
SOURCE = """
module {
  func.func @private_root() {
    %zero = arith.constant 0 : i32
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

    returned = SOURCE.replace(
        "@private_root()", "@private_root() -> memref<64xi32>"
    ).replace(
        "    memref.dealloc %a : memref<64xi32>\n    return",
        "    return %a : memref<64xi32>",
    )
    address = SOURCE.replace(
        "@private_root()", "@private_root() -> index"
    ).replace(
        "    memref.dealloc %a : memref<64xi32>\n    return",
        "    %alias = memref.subview %a[0] [64] [1] : memref<64xi32> "
        "to memref<64xi32, strided<[1]>>\n"
        "    %address = memref.extract_aligned_pointer_as_index %alias "
        ": memref<64xi32, strided<[1]>> -> index\n"
        "    return %address : index",
    )
    dynamic_alias = SOURCE.replace(
        "    memref.dealloc %a : memref<64xi32>",
        "    %alias = memref.cast %a : memref<64xi32> to memref<?xi32>\n"
        "    memref.dealloc %alias : memref<?xi32>",
    )
    yielded = """
module {
  func.func @yielded() -> memref<64xi32> {
    %true = arith.constant true
    %zero = arith.constant 0 : i32
    %result = scf.if %true -> (memref<64xi32>) {
      %a = memref.alloc() alignment = 64 : memref<64xi32>
      linalg.fill ins(%zero : i32) outs(%a : memref<64xi32>)
      scf.yield %a : memref<64xi32>
    } else {
      %b = memref.alloc() alignment = 64 : memref<64xi32>
      linalg.fill ins(%zero : i32) outs(%b : memref<64xi32>)
      scf.yield %b : memref<64xi32>
    }
    return %result : memref<64xi32>
  }
}
"""
    unmodeled = SOURCE.replace(
        "    memref.dealloc %a : memref<64xi32>",
        "    %c0 = arith.constant 0 : index\n"
        "    memref.prefetch %a[%c0], read, locality<3>, data : memref<64xi32>\n"
        "    memref.dealloc %a : memref<64xi32>",
    )
    for name, text in (
        ("returned_root", returned),
        ("extracted_alias_address", address),
        ("nonstatic_alias", dynamic_alias),
        ("yielded_root", yielded),
        ("unmodeled_completion", unmodeled),
    ):
        rejected(name, text, "--kernel-contract-annotate", "--bosc-ame-plan")

    # The proof also covers existing physical windows normalized by adapters.
    fixed = (
        address.replace(
            "%a = memref.alloc() alignment = 64 : memref<64xi32>",
            "%a = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} "
            ": memref<64xi32, #bosc_ame.spm<0>>",
        )
        .replace(
            "outs(%a : memref<64xi32>)",
            "outs(%a : memref<64xi32, #bosc_ame.spm<0>>)",
        )
        .replace(
            ": memref<64xi32> to memref<64xi32, strided<[1]>>",
            ": memref<64xi32, #bosc_ame.spm<0>> "
            "to memref<64xi32, strided<[1]>, #bosc_ame.spm<0>>",
        )
        .replace(
            ": memref<64xi32, strided<[1]>> -> index",
            ": memref<64xi32, strided<[1]>, #bosc_ame.spm<0>> -> index",
        )
    )
    for entry in ("--linalg-to-boscame", "--lower-linalg-to-boscame"):
        rejected("fixed_" + entry[2:], fixed, entry)

    # No new graph event is needed for a pure address extraction. Verify and
    # emit must reject it even when the supplied plan has otherwise valid IDs.
    baseline = invoke(
        "baseline", SOURCE, "--kernel-contract-annotate", "--bosc-ame-plan"
    )
    assert baseline.returncode == 0, baseline.stderr
    planned = (
        baseline.stdout.replace("@private_root()", "@private_root() -> index")
        .replace(
            "    return",
            "    %address = memref.extract_aligned_pointer_as_index %alloc "
            ": memref<64xi32> -> index\n"
            "    return %address : index",
        )
        .replace("    memref.dealloc %alloc : memref<64xi32>\n", "")
    )
    for entry in ("--bosc-ame-plan-verify", "--bosc-ame-emit"):
        rejected("planned_" + entry[2:], planned, entry)

    # Static aliases with synchronous local users keep a legal SPM root.
    local = SOURCE.replace(
        "    memref.dealloc %a : memref<64xi32>",
        "    %alias = memref.subview %a[0] [64] [1] : memref<64xi32> "
        "to memref<64xi32, strided<[1]>>\n"
        "    linalg.fill ins(%zero : i32) "
        "outs(%alias : memref<64xi32, strided<[1]>>)\n"
        "    memref.dealloc %a : memref<64xi32>",
    )
    result = invoke(
        "local_completed_alias",
        local,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
        "--bosc-ame-emit",
    )
    assert result.returncode == 0, result.stderr

    # Deallocation through a verified static descriptor alias must disappear
    # with the original heap root. Keeping it would leave an illegal SPM free
    # after emission even though the logical ownership proof succeeded.
    cast = (
        "    %cast = memref.cast %a : memref<64xi32> "
        "to memref<64xi32, strided<[1]>>\n"
    )
    reinterpret = (
        "    %view = memref.reinterpret_cast %a to offset: [0], sizes: [64], "
        "strides: [1] : memref<64xi32> to memref<64xi32, strided<[1]>>\n"
    )
    chain = cast + (
        "    %view = memref.reinterpret_cast %cast to offset: [0], sizes: [64], "
        "strides: [1] : memref<64xi32, strided<[1]>> "
        "to memref<64xi32, strided<[1]>>\n"
    )
    for name, aliases, value in (
        ("cast_dealloc", cast, "%cast"),
        ("reinterpret_dealloc", reinterpret, "%view"),
        ("chained_dealloc", chain, "%view"),
    ):
        text = SOURCE.replace(
            "    memref.dealloc %a : memref<64xi32>",
            aliases
            + f"    memref.dealloc {value} : memref<64xi32, strided<[1]>>",
        )
        result = invoke(
            name,
            text,
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
            "--bosc-ame-emit",
            "--lower-bosc-ame",
        )
        assert result.returncode == 0, (name, result.stderr)
        assert "memref.dealloc" not in result.stdout, (name, result.stdout)
        assert "@free" not in result.stdout, (name, result.stdout)
    print(
        "PASS SPM ownership rejects escaped/unknown aliases and accepts local static views"
    )


if __name__ == "__main__":
    main()
