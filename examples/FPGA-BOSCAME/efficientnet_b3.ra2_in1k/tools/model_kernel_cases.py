#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0. SPDX-License-Identifier: Apache-2.0
"""Vision model cases live next to this tools/ directory."""
import sys
from pathlib import Path
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    example = here.parent
    target = here / "emit_model_cases.py"
    raise SystemExit(
        f"Committed cases: {example}. "
        f"Refresh with: python -B {target} --ops <raw-ops.json> "
        f"--output {example} (or REEMIT=1 run_full.sh)."
    )
