# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
# Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.
"""Check compiled NR DMA copies and exact guarded copy/vector-fill results."""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Buffer, Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402

MODULE = """module {
  func.func @copy(%input: memref<256xi8>, %out: memref<256xi8>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                    affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
        ins(%input : memref<256xi8>) outs(%out : memref<256xi8>) {
    ^bb0(%value: i8, %unused: i8):
      linalg.yield %value : i8
    }
    return
  }
  func.func @fill(%out: memref<256xi8>) {
    %value = arith.constant -7 : i8
    linalg.fill ins(%value : i8) outs(%out : memref<256xi8>)
    return
  }
  func.func @spm_copy() {
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<256xi8, #bosc_ame.spm<0>>
    %out = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
      : memref<256xi8, #bosc_ame.spm<1>>
    memref.copy %input, %out : memref<256xi8, #bosc_ame.spm<0>>
      to memref<256xi8, #bosc_ame.spm<1>>
    return
  }
  func.func @typed_spm_copy() {
    %c64 = arith.constant 64 : index
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
      : memref<128xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 256 : i64}
      : memref<128xi8, #bosc_ame.spm<1>>
    %source_typed = memref.view %source[%c64][]
      : memref<128xi8, #bosc_ame.spm<0>> to memref<16xi32, #bosc_ame.spm<0>>
    %target_typed = memref.view %target[%c64][]
      : memref<128xi8, #bosc_ame.spm<1>> to memref<16xi32, #bosc_ame.spm<1>>
    memref.copy %source_typed, %target_typed
      : memref<16xi32, #bosc_ame.spm<0>> to memref<16xi32, #bosc_ame.spm<1>>
    return
  }
}
"""


# Keep the historical logical source above as a precise compatibility-negative
# case. Independent DDR and SPM copies are outside the V1 planner recipe, but
# the target DMA lowering still supports and must verify these real endpoints.
TARGET_MODULE = """module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @copy(%input: memref<256xi8>, %out: memref<256xi8>) {
    %n = arith.constant 256 : index
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<256xi8, #bosc_ame.spm<0>>
    %b = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<256xi8, #bosc_ame.spm<1>>
    %t0 = bosc_ame.dma.start %input, %a, %n {channel = 0 : i32} : (memref<256xi8>, memref<256xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t0 : !bosc_ame.dma_token<0>
    %t1 = bosc_ame.dma.start %a, %b, %n {channel = 1 : i32} : (memref<256xi8, #bosc_ame.spm<0>>, memref<256xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t1 : !bosc_ame.dma_token<1>
    %t2 = bosc_ame.dma.start %b, %out, %n {channel = 1 : i32} : (memref<256xi8, #bosc_ame.spm<1>>, memref<256xi8>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t2 : !bosc_ame.dma_token<1>
    return
  }
  func.func @fill(%out: memref<256xi8>) {
    %zero = arith.constant 0 : index
    %end = arith.constant 256 : index
    %step = arith.constant 64 : index
    %value = arith.constant -7 : i8
    %vector = vector.broadcast %value : i8 to vector<64xi8>
    scf.for %i = %zero to %end step %step {
      vector.store %vector, %out[%i] : memref<256xi8>, vector<64xi8>
    }
    return
  }
  func.func @spm_copy() {
    %n = arith.constant 256 : index
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<256xi8, #bosc_ame.spm<0>>
    %out = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<256xi8, #bosc_ame.spm<1>>
    %t = bosc_ame.dma.start %input, %out, %n {channel = 1 : i32} : (memref<256xi8, #bosc_ame.spm<0>>, memref<256xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<1>
    return
  }
  func.func @typed_spm_copy() {
    %c64 = arith.constant 64 : index
    %source = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64} : memref<128xi8, #bosc_ame.spm<0>>
    %target = bosc_ame.spm.view {bank = 1 : i32, offset = 256 : i64} : memref<128xi8, #bosc_ame.spm<1>>
    %source_typed = memref.view %source[%c64][] : memref<128xi8, #bosc_ame.spm<0>> to memref<16xi32, #bosc_ame.spm<0>>
    %target_typed = memref.view %target[%c64][] : memref<128xi8, #bosc_ame.spm<1>> to memref<16xi32, #bosc_ame.spm<1>>
    %t = bosc_ame.dma.start %source_typed, %target_typed, %c64 {channel = 1 : i32} : (memref<16xi32, #bosc_ame.spm<0>>, memref<16xi32, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<1>
    return
  }
}
"""


def run(command):
    result = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def argument_names(machine):
    signature = machine.function.text.split("(", 1)[1].split(")", 1)[0]
    return re.findall(r"(%[\w.\-]+)\s*:", signature)


def function_assembly(source, name):
    function = re.search(
        rf"(?ms)^{re.escape(name)}:.*?^\s*\.size\s+{re.escape(name)},",
        source,
    )
    assert function, name
    return function[0]


def dma_calls(source):
    """Read actual constant arguments at calls, respecting caller clobbers."""
    registers = {"zero": 0}
    starts, pending = [], False
    for line in source.splitlines():
        fields = line.split("#", 1)[0].strip().split(None, 1)
        if len(fields) != 2:
            continue
        name, operands = fields
        args = [value.strip() for value in operands.split(",")]
        if name == "li":
            registers[args[0]] = int(args[1], 0)
        elif name == "lui":
            registers[args[0]] = int(args[1], 0) << 12
        elif name in ("addi", "addiw", "slli"):
            if args[1] in registers:
                value, immediate = registers[args[1]], int(args[2], 0)
                registers[args[0]] = (
                    value << immediate if name == "slli" else value + immediate
                )
            else:
                registers.pop(args[0], None)
        elif name == "mv":
            if args[1] in registers:
                registers[args[0]] = registers[args[1]]
            else:
                registers.pop(args[0], None)
        elif name in ("ld", "lw", "lwu"):
            registers.pop(args[0], None)
        elif name in ("call", "tail"):
            if operands == "bosc_ame_dma_start":
                assert not pending, "DMA start without the preceding wait"
                starts.append(
                    tuple(registers.get("a" + str(i)) for i in range(4))
                )
                pending = True
            elif operands == "bosc_ame_dma_wait":
                assert pending, "DMA wait without a start"
                pending = False
            else:
                raise AssertionError("unexpected runtime call: " + operands)
            for register in tuple(registers):
                if register.startswith(("a", "t")):
                    del registers[register]
    assert not pending
    return starts


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    module, staged, lowered, llvm, assembly = (
        work / ("copy" + suffix)
        for suffix in (".mlir", ".boscame.mlir", ".llvm.mlir", ".ll", ".s")
    )
    module.write_text(MODULE)
    opt = Path(buddy_opt_path())
    for section in MODULE.split("  func.func ")[1:]:
        body_start = section.index("{")
        depth = 0
        for position in range(body_start, len(section)):
            depth += (section[position] == "{") - (section[position] == "}")
            if not depth:
                function = section[: position + 1]
                break
        legacy = subprocess.run(
            [
                str(opt),
                "-",
                "--linalg-to-boscame=plan-version=1 enable-spm-dma=true dma-abi=args_aligned64_nonalias",
            ],
            input="module {func.func " + function + "}",
            text=True,
            capture_output=True,
        )
        assert legacy.returncode != 0, function
        diagnostic = (
            "v1 has no independent DDR-to-DDR copy recipe"
            if section.startswith("@copy(")
            else "unplanned external fill"
            if section.startswith("@fill(")
            else "read or partial inheritance has no initialized content version"
        )
        assert diagnostic in legacy.stderr, legacy.stderr
    staged.write_text(TARGET_MODULE)
    run([opt, staged, "--bosc-ame-spm-dma-check", "-o", "/dev/null"])
    text = staged.read_text()
    assert "vector<64xi8>" in text
    run([opt, staged, "--lower-bosc-ame", "-o", lowered])
    run(
        [
            opt.with_name("buddy-translate"),
            lowered,
            "--buddy-to-llvmir",
            "-o",
            llvm,
        ]
    )
    run(
        [
            REPO / "llvm/build/bin/llc",
            llvm,
            "-O2",
            "-mtriple=riscv64",
            "-target-abi=lp64d",
            "-mattr=+m,+a,+f,+d,+c,+v",
            "-o",
            assembly,
        ]
    )
    code = assembly.read_text()
    copy_code = function_assembly(code, "copy")
    direct_code = function_assembly(code, "spm_copy")
    typed_code = function_assembly(code, "typed_spm_copy")
    for copied in (copy_code, direct_code, typed_code):
        assert not re.search(r"\b(?:vset\w*|vle\w*|vse\w*)\b", copied), copied
    staged_calls = dma_calls(copy_code)
    assert len(staged_calls) == 3, staged_calls
    channel, source_address, target_address, length = staged_calls[1]
    assert channel in (0, 1) and length == 256, staged_calls
    assert 0x20000000 <= source_address < 0x20020000, staged_calls
    assert 0x20400000 <= target_address < 0x20420000, staged_calls
    direct_calls = dma_calls(direct_code)
    assert len(direct_calls) == 1, direct_calls
    channel, source_address, target_address, length = direct_calls[0]
    assert channel in (0, 1)
    assert (source_address, target_address, length) == (
        0x20000000,
        0x20400000,
        256,
    ), direct_calls
    typed_calls = dma_calls(typed_code)
    assert len(typed_calls) == 1, typed_calls
    channel, source_address, target_address, length = typed_calls[0]
    assert channel in (0, 1)
    # Include the spm.view's 256-byte bank offset and the memref.view's
    # additional 64-byte pointer shift exactly once, with a typed 64-byte DMA.
    assert (source_address, target_address, length) == (
        0x20000140,
        0x20400140,
        64,
    ), typed_calls
    fill_code = function_assembly(code, "fill")
    assert re.search(r"vsetvli[^\n]*e8,\s*m4", fill_code), fill_code
    assert re.search(r"\bvse8.v\b", fill_code), fill_code

    guard = bytes([0xA7]) * 64
    machine = Machine(staged, "copy")
    names = argument_names(machine)
    for round in (0, 1):
        payload = bytes((index * 13 + round * 97) % 256 for index in range(256))
        source = bytearray(guard + payload + guard)
        target = bytearray(guard + bytes([0xCD]) * 256 + guard)
        machine.values = {
            names[0]: Buffer(source, "i8", [256], base=64, name="input"),
            names[1]: Buffer(target, "i8", [256], base=64, name="out"),
        }
        machine.block(machine.function.blocks[0])
        assert source == target == guard + payload + guard
        assert not machine.pending
    assert (
        machine.accesses
        == [("input", 64, 320, "read"), ("out", 64, 320, "write")] * 2
    )

    fill = Machine(staged, "fill")
    output = bytearray(guard + bytes(256) + guard)
    fill.values = {
        argument_names(fill)[0]: Buffer(
            output, "i8", [256], base=64, name="out"
        )
    }
    fill.block(fill.function.blocks[0])
    assert output == guard + bytes([249]) * 256 + guard
    direct = Machine(staged, "spm_copy")
    direct.spm[0][:256] = payload
    direct.block(direct.function.blocks[0])
    assert direct.spm[1][:256] == payload and not direct.accesses
    typed = Machine(staged, "typed_spm_copy")
    typed.spm[0][320:384] = payload[:64]
    source_before, target_before = bytes(typed.spm[0]), bytes(typed.spm[1])
    typed.block(typed.function.blocks[0])
    assert bytes(typed.spm[0]) == source_before
    assert (
        typed.spm[1] == target_before[:320] + payload[:64] + target_before[384:]
    )
    assert not typed.pending and not typed.accesses
    # CHECK: PASS NR DMA copies: exact compiled endpoints and guarded copy/fill
    print("PASS NR DMA copies: exact compiled endpoints and guarded copy/fill")


if __name__ == "__main__":
    main()
