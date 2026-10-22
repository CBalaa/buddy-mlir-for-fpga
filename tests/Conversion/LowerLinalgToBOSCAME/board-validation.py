# RUN: %PYTHON %s 2>&1 | FileCheck %s
"""Keep board evidence tied to artifact hashes and reject corrupt DDR/logs."""

import subprocess
import sys
from pathlib import Path

checker = (
    Path(__file__).resolve().parents[3]
    / "examples/FPGA-BOSCAME/linalg-boscame/check_board_test.py"
)
subprocess.run([sys.executable, str(checker)], check=True)

# CHECK: Ran 6 tests
# CHECK: OK
