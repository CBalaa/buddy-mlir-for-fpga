# RUN: %PYTHON %s %t
"""Fixed windows retain alias identity, exact DMA ranges and completion order."""

import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402

SOURCE = """module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @copy(%source: memref<32xi32>, %dest: memref<32xi32>) {
    %bytes = arith.constant 128 : index
    %zero = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %window = bosc_ame.spm.view {bank = 1 : i32, offset = 4096 : i64}
      : memref<64xi32, #bosc_ame.spm<1>>
    %middle = memref.subview %window[16] [32] [1]
      : memref<64xi32, #bosc_ame.spm<1>> to memref<32xi32, strided<[1], offset: 16>, #bosc_ame.spm<1>>
    %input = bosc_ame.dma.start %source, %middle, %bytes {channel = 0 : i32}
      : (memref<32xi32>, memref<32xi32, strided<[1], offset: 16>, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %input : !bosc_ame.dma_token<0>
    %output = bosc_ame.dma.start %middle, %dest, %bytes {channel = 0 : i32}
      : (memref<32xi32, strided<[1], offset: 16>, #bosc_ame.spm<1>>, memref<32xi32>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %output : !bosc_ame.dma_token<0>
    return
  }
}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = str(buddy_opt_path())

    def invoke(source, entry="linalg-to-boscame"):
        return subprocess.run(
            [opt, f"--{entry}=buffer-placement=false plan-report=true"],
            input=source,
            text=True,
            capture_output=True,
        )

    original = struct.pack("<32i", *(i - 16 for i in range(32)))
    guard = b"\xa7" * 64
    results = []
    for entry in ("linalg-to-boscame", "lower-linalg-to-boscame"):
        result = invoke(SOURCE, entry)
        assert result.returncode == 0, result.stderr
        results.append(result.stdout)
        assert "offset = 4160 : i64" in result.stdout
        assert "capacity = 128 : i64" in result.stderr
        module = work / f"{entry}.mlir"
        module.write_text(result.stdout)
        machine = Machine(module, "copy")
        machine.spm[1][:] = b"\x5a" * len(machine.spm[1])
        incoming = bytearray(original + guard)
        outgoing = bytearray(b"\xcc" * 128 + guard)
        machine.launch({"%arg0": incoming, "%arg1": outgoing})
        assert incoming == original + guard
        assert outgoing == original + guard
        assert machine.spm[1][4096:4160] == b"\x5a" * 64
        assert machine.spm[1][4288:4352] == b"\x5a" * 64
        checked = subprocess.run(
            [opt, str(module), "--bosc-ame-spm-dma-check"],
            text=True,
            capture_output=True,
        )
        assert checked.returncode == 0, checked.stderr
    assert results[0] == results[1]

    rejected = [
        (
            SOURCE.replace(
                "    bosc_ame.dma.wait %input",
                "    memref.store %zero, %dest[%c0] : memref<32xi32>\n"
                "    bosc_ame.dma.wait %input",
                1,
            ),
            "cannot normalize asynchronous DMA across an intervening effect",
        ),
        (
            SOURCE.replace(
                "    bosc_ame.dma.wait %input : !bosc_ame.dma_token<0>",
                "    bosc_ame.dma.wait %input : !bosc_ame.dma_token<0>\n"
                "    bosc_ame.dma.wait %input : !bosc_ame.dma_token<0>",
                1,
            ),
            "requires one local completion",
        ),
        (
            SOURCE.replace(
                "    return",
                "    %overlap = bosc_ame.spm.view "
                "{bank = 1 : i32, offset = 4160 : i64} "
                ": memref<64xi32, #bosc_ame.spm<1>>\n    return",
                1,
            ),
            "overlapping SPM input windows require a shared root",
        ),
    ]
    for source, diagnostic in rejected:
        result = invoke(source)
        assert result.returncode != 0
        assert diagnostic in result.stderr, result.stderr


if __name__ == "__main__":
    main()
