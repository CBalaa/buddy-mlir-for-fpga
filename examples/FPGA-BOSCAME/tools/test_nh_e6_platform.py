"""Host-side check for the common NH/E6 platform module.

common/nh_e6/test_nh_e6_platform.c redirects the SYSCTRL/mailbox MMIO and the
common/uart register block to in-memory fakes, captures the UART byte stream
and includes the production uart.c and nh_e6_platform.c, so the E6 PMP/satp
setup, the CSR 0xbc0 cache policies, the NH release/wait flow, the progress and
verdict reporting, the E6 completion protocol and the DMA-failure publication
are executed with the exact board addresses, register offsets, CSR values,
barrier order and message text.

The test is compiled at two optimization levels (and once with the sanitizers)
to catch undefined behaviour in the hand-translated asm-to-C flow.

Run with:  python3 -m unittest discover examples/FPGA-BOSCAME/tools
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


FPGA = Path(__file__).resolve().parents[1]
RUNTIME = FPGA / "common" / "runtime"
UART = FPGA / "common" / "uart"
PLATFORM = FPGA / "common" / "nh_e6"
SOURCE = PLATFORM / "test_nh_e6_platform.c"


@unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
class NhE6PlatformTest(unittest.TestCase):
    def build_and_run(self, directory, optimization="-O1", extra_flags=()):
        binary = Path(directory) / "nh_e6_platform_test"
        subprocess.run(
            ["cc", "-std=c11", optimization, "-g", "-Wall", "-Wextra",
             "-Werror", f"-I{RUNTIME}", f"-I{UART}", f"-I{PLATFORM}",
             *extra_flags, str(SOURCE), "-o", str(binary)],
            check=True, capture_output=True, text=True)
        return subprocess.run([str(binary)], check=True, capture_output=True,
                              text=True).stdout

    def test_unoptimized_build(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_and_run(directory, "-O0")
        self.assertIn("nh_e6_platform: all checks passed", output)

    def test_optimized_build(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_and_run(directory, "-O2")
        self.assertIn("nh_e6_platform: all checks passed", output)

    def test_sanitizer_build(self):
        # The fake register files are indexed from the platform's fixed board
        # addresses, so a wrong offset shows up as an out-of-bounds access.
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_and_run(
                directory, "-O1",
                ("-fsanitize=address,undefined", "-fno-omit-frame-pointer"))
        self.assertIn("nh_e6_platform: all checks passed", output)
