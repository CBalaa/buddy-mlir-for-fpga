# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Check SPM-only arithmetic, ordered reduction, and strict vector boundaries."""

import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path, experimental_pipeline  # noqa: E402

SOURCE = """
#id = affine_map<(i) -> (i)>
#id2 = affine_map<(i, j) -> (i, j)>
module {
  func.func @ordered_sum() -> f32 {
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %x = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1024xf32, #bosc_ame.spm<0>>
    %y = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<1024xf32, #bosc_ame.spm<1>>
    %sum = memref.alloca() : memref<f32>
    linalg.fill ins(%one : f32) outs(%x : memref<1024xf32, #bosc_ame.spm<0>>)
    linalg.fill ins(%zero : f32) outs(%sum : memref<f32>)
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<1024xf32, #bosc_ame.spm<0>>)
      outs(%y : memref<1024xf32, #bosc_ame.spm<1>>) {
      ^bb0(%in: f32, %out: f32):
        %square = arith.mulf %in, %in : f32
        linalg.yield %square : f32
    }
    linalg.reduce ins(%y : memref<1024xf32, #bosc_ame.spm<1>>)
      outs(%sum : memref<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %next = arith.addf %in, %init : f32
        linalg.yield %next : f32
    }
    %value = memref.load %sum[] : memref<f32>
    return %value : f32
  }
  func.func @rank_two_add(%out: memref<2x16xi32>) {
    %one = arith.constant 1 : i32
    %max = arith.constant 2147483647 : i32
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 4096 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    %b = bosc_ame.spm.view {bank = 1 : i32, offset = 4096 : i64} : memref<2x16xi32, #bosc_ame.spm<1>>
    %c = bosc_ame.spm.view {bank = 0 : i32, offset = 4224 : i64} : memref<2x16xi32, #bosc_ame.spm<0>>
    linalg.fill ins(%max : i32) outs(%a : memref<2x16xi32, #bosc_ame.spm<0>>)
    linalg.fill ins(%one : i32) outs(%b : memref<2x16xi32, #bosc_ame.spm<1>>)
    linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<2x16xi32, #bosc_ame.spm<0>>, memref<2x16xi32, #bosc_ame.spm<1>>)
      outs(%c : memref<2x16xi32, #bosc_ame.spm<0>>) {
      ^bb0(%lhs: i32, %rhs: i32, %old: i32):
        %value = arith.addi %lhs, %rhs : i32
        linalg.yield %value : i32
    }
    memref.copy %c, %out : memref<2x16xi32, #bosc_ame.spm<0>> to memref<2x16xi32>
    return
  }
  func.func @cancellation(%out: memref<f32>) {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %pattern = arith.constant dense<[1.000000e+20, 1.0, -1.000000e+20, 1.0,
      1.000000e+20, 1.0, -1.000000e+20, 1.0,
      1.000000e+20, 1.0, -1.000000e+20, 1.0,
      1.000000e+20, 1.0, -1.000000e+20, 1.0]> : vector<16xf32>
    %x = bosc_ame.spm.view {bank = 0 : i32, offset = 8192 : i64} : memref<1024xf32, #bosc_ame.spm<0>>
    scf.for %i = %c0 to %c1024 step %c16 {
      vector.store %pattern, %x[%i] : memref<1024xf32, #bosc_ame.spm<0>>, vector<16xf32>
    }
    linalg.fill ins(%zero : f32) outs(%out : memref<f32>)
    linalg.reduce ins(%x : memref<1024xf32, #bosc_ame.spm<0>>)
      outs(%out : memref<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %next = arith.addf %in, %init : f32
        linalg.yield %next : f32
    }
    return
  }
}
"""


def run(command, source=None):
    return subprocess.run(
        [str(arg) for arg in command],
        input=source,
        capture_output=True,
        text=True,
    )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = Path(buddy_opt_path())
    command = [
        opt,
        *experimental_pipeline(),
    ]
    result = run(command, SOURCE)
    assert result.returncode == 0, result.stdout + result.stderr
    text = result.stdout
    assert text.count("vector.reduction <add>") == 2, text
    assert "vector.extract " not in text, text
    assert "arith.addf " not in text, text
    assert "vector<1xf32>" not in text, text
    assert "iter_args" in text
    assert "vector<16xf32>" in text and "vector<16xi32>" in text
    assert not re.search(r"vector<\d+x\d+x", text)
    assert not re.search(r"\blinalg\.(generic|reduce)\b", text)
    assert "memref.copy" not in text
    for line in text.splitlines():
        if re.search(r"memref\.(load|store)", line):
            assert "#bosc_ame.spm" not in line, line
    assert "reassoc" not in text
    lowered = run([opt, "--lower-bosc-ame"], text)
    assert lowered.returncode == 0, lowered.stdout + lowered.stderr
    llvm = run(
        [opt.with_name("buddy-translate"), "--buddy-to-llvmir"], lowered.stdout
    )
    assert llvm.returncode == 0, llvm.stdout + llvm.stderr
    assert (
        len(
            re.findall(
                r"call float @llvm\.vector\.reduce\.fadd\.v16f32", llvm.stdout
            )
        )
        == 2
    ), llvm.stdout
    assert "extractelement <1 x float>" not in llvm.stdout, llvm.stdout
    assert not re.search(r"\bfadd float\b", llvm.stdout), llvm.stdout
    assert not re.search(r"\b(?:reassoc|fast)\b", llvm.stdout)
    llc = REPO / "llvm/build/bin/llc"
    if not llc.is_file():
        llc = shutil.which("llc")
    asm = run(
        [
            llc,
            "-O2",
            "-mtriple=riscv64",
            "-target-abi=lp64d",
            "-mattr=+m,+a,+f,+d,+c,+v",
            "-filetype=asm",
            "-o",
            "-",
        ],
        llvm.stdout,
    )
    assert asm.returncode == 0, asm.stdout + asm.stderr
    assert asm.stdout.count("vfredosum.vs") == 2, asm.stdout
    assert "vfredusum.vs" not in asm.stdout, asm.stdout
    assert not re.search(r"\bfadd\.s\b", asm.stdout), asm.stdout
    assert not re.search(r"\b(?:vslidedown|vslideup|vrgather)\.", asm.stdout)
    ordered_asm = asm.stdout.split("ordered_sum:", 1)[1].split(".Lfunc_end", 1)[
        0
    ]
    assert "vle32.v" in ordered_asm, ordered_asm
    # This incomplete-graph fixture has descriptors on the scalar allocation,
    # so conservative promotion retains its stack storage. Track stack-derived
    # registers to reject scalarized SPM reads while allowing that fallback.
    stack_registers = {"sp"}
    for instruction in ordered_asm.splitlines():
        fields = instruction.split("#", 1)[0].strip().split(None, 1)
        if len(fields) != 2 or fields[0].startswith("."):
            continue
        mnemonic, operands = fields
        if mnemonic == "flw":
            base = re.search(r"\((\w+)\)", operands).group(1)
            assert base in stack_registers, instruction
        operands = [operand.strip() for operand in operands.split(",")]
        dest = operands[0]
        if mnemonic in {"call", "sd", "sw", "sh", "sb", "fsd", "fsw"}:
            continue
        is_stack = (
            mnemonic in {"mv", "addi", "addiw", "andi"}
            and len(operands) >= 2
            and operands[1] in stack_registers
        ) or (
            mnemonic == "add"
            and any(operand in stack_registers for operand in operands[1:])
        )
        stack_registers.discard(dest)
        if is_stack:
            stack_registers.add(dest)
    assert not re.search(
        r"\b(?:vlenb|vs[1248]r\.v|vl[1248]re\d+\.v)\b", asm.stdout
    )
    (work / "spm-compute.mlir").write_text(text)
    (work / "spm-compute.s").write_text(asm.stdout)
    loops = run(
        [opt, "--convert-linalg-to-loops", "--lower-affine", "--canonicalize"],
        text,
    )
    assert loops.returncode == 0, loops.stdout + loops.stderr
    loop_path = work / "spm-compute.loops.mlir"
    loop_path.write_text(loops.stdout)
    # Lane-wise cancellation makes an ordered sum differ from independent
    # vector-lane accumulators: every fourth scalar addition ends at 1.0f.
    result_bytes = bytearray(4)
    Machine(loop_path, "cancellation").launch({"%arg0": result_bytes})
    assert struct.unpack("<f", result_bytes)[0] == 1.0
    matrix_bytes = bytearray(2 * 16 * 4)
    Machine(loop_path, "rank_two_add").launch({"%arg0": matrix_bytes})
    assert struct.unpack("<32I", matrix_bytes) == (0x80000000,) * 32
    for source, error in (
        (SOURCE.replace("1024xf32", "1007xf32"), "complete 64 B vectors"),
        (
            SOURCE.replace("arith.mulf %in, %in", "math.powf %in, %in"),
            "unsupported operation in SPM vector computation",
        ),
        (
            SOURCE.replace(
                "%next = arith.addf %in, %init : f32",
                "%next = arith.addf %in, %init fastmath<reassoc> : f32",
            ),
            "without fast-math flags",
        ),
    ):
        bad = run(command, source)
        assert bad.returncode != 0, "invalid SPM computation returned success"
        assert error in bad.stderr, bad.stderr
    # CHECK: PASS SPM compute: rank-1/rank-2 arithmetic, vector fill/copy and no scalar SPM accesses
    print(
        "PASS SPM compute: rank-1/rank-2 arithmetic, vector fill/copy and no scalar SPM accesses"
    )
    # CHECK: PASS ordered reduction: 16-lane SPM vector reads and vfredosum.vs without reassociation
    print(
        "PASS ordered reduction: 16-lane SPM vector reads and vfredosum.vs without reassociation"
    )
    # CHECK: PASS rejection: short vector tails, runtime math and reassociated sum fail nonzero
    print(
        "PASS rejection: short vector tails, runtime math and reassociated sum fail nonzero"
    )


if __name__ == "__main__":
    main()
