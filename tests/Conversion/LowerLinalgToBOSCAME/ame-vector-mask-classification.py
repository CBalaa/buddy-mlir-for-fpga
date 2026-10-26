# RUN: %PYTHON %s
"""RVV mask suffixes reach RVV validation without bypassing its allowlist."""

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from ame_to_word import convert_line  # noqa: E402
from restrict_fpga_assembly import transform  # noqa: E402

for mnemonic in ("vmor.mm", "vmand.mm", "vunknown.mm"):
    source = f"\t{mnemonic}\tv0, v8, v8\n"
    converted, changed = convert_line(source)
    assert converted == source and not changed
    try:
        transform(converted)
    except ValueError as error:
        assert "RVV instruction not allowed" in str(error)
    else:
        raise AssertionError("unverified RVV mask instruction accepted")

for source in (
    "munknown.mm acc0, acc1, acc2",
    "mqma.unknown.mm acc0, tr0, tr1",
):
    try:
        convert_line(source)
    except ValueError as error:
        assert "unsupported NR AME instruction" in str(error)
    else:
        raise AssertionError("unencoded AME instruction accepted")

print("PASS RVV mask classification preserves strict instruction validation")
