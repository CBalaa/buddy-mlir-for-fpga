# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Check direct AME stores use a byte pitch and never become a DMA."""

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_sim import simulate  # noqa: E402

SOURCE = """module {
  func.func @direct(%raw: memref<?xi8>) {
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<2x3xi32, #bosc_ame.spm<0>>
    %acc = bosc_ame.tile.load_acc %spm : memref<2x3xi32, #bosc_ame.spm<0>> -> vector<2x3xi32>
    %out = memref.reinterpret_cast %raw to offset: [0], sizes: [2, 5], strides: [5, 1] : memref<?xi8> to memref<2x5xi32, strided<[5, 1], offset: 0>>
    func.call @bosc_ame_ddr_write_begin() : () -> ()
    bosc_ame.tile.store_acc %acc, %out {bosc_ame.direct_ddr} : vector<2x3xi32>, memref<2x5xi32>
    return
  }
}
"""


def main():
    path = Path(sys.argv[1])
    path.write_text(SOURCE)
    data = bytearray(b"\xcc" * (2 * 5 * 4))
    machine, _ = simulate(path, "direct", {"%raw": data})
    # The 20-byte stride writes three i32 values on each row and preserves the
    # two-element padding at the end of each five-element physical row.
    assert data[12:20] == b"\xcc" * 8
    assert data[32:] == b"\xcc" * 8
    assert machine.dma_log == []
    assert len(machine.writes()) == 6
    # CHECK: PASS direct AME store simulator: pitched DDR rows and no DMA
    print("PASS direct AME store simulator: pitched DDR rows and no DMA")


if __name__ == "__main__":
    main()
