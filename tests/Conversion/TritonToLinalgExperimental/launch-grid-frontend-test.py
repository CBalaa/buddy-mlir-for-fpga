# RUN: %PYTHON %s
"""Check that the frontend passes its saved launch grid to buddy-opt."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
FRONTEND_DIR = REPO / "examples/FPGA-BOSCAME/linalg-boscame"
sys.path.insert(0, str(FRONTEND_DIR))
import frontend  # noqa: E402


class LaunchGridFrontendTest(unittest.TestCase):
    def test_build_frontend_uses_contract_grid(self):
        expected = {
            "gemm": "launch-grid=triton_gemm:2,1,1",
            "layernorm": "launch-grid=triton_layernorm:197,1,1",
        }
        for kernel, option in expected.items():
            with (
                self.subTest(kernel=kernel),
                tempfile.TemporaryDirectory() as tmp,
            ):
                commands = []

                def capture_run(command, commands=commands):
                    command = [str(part) for part in command]
                    commands.append(command)
                    output = Path(command[command.index("-o") + 1])
                    output.write_text("module {}\n")

                args = SimpleNamespace(
                    out_dir=Path(tmp), stop_after="linalg", buddy_opt=None
                )
                with (
                    patch.object(frontend, "export_ttir"),
                    patch.object(
                        frontend, "find_tool", return_value="buddy-opt"
                    ),
                    patch.object(frontend, "run", side_effect=capture_run),
                ):
                    frontend.build_frontend(args, kernel)

                self.assertEqual(len(commands), 1)
                pass_options = next(
                    part
                    for part in commands[0]
                    if part.startswith("--triton-to-linalg-experimental=")
                )
                self.assertIn(option, pass_options)


if __name__ == "__main__":
    unittest.main()
