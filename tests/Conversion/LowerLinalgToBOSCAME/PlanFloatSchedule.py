# RUN: %PYTHON %s %t
"""Microtiles preserve independent partial sums, snapshots, and tail guards."""

import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def source(valid):
    return """
#id = affine_map<(d0,d1)->(d0,d1)>
module attributes {contract.abi={args_aligned_bytes=64:i64,args_disjoint=true}} {
func.func @micro(%arg0:memref<*xf32>,%arg1:memref<*xf32>,%arg2:memref<*xf32>) {
%c0=arith.constant 0:index
%c1=arith.constant 1:index
%c2=arith.constant 2:index
%c4=arith.constant 4:index
%zero=arith.constant 0.0:f32
%acc=memref.alloc():memref<1x16xf32>
linalg.fill ins(%zero:f32) outs(%acc:memref<1x16xf32>)
scf.for %ki=%c0 to %c2 step %c1 {
 %offset=arith.muli %ki,%c4:index
 %aext=memref.reinterpret_cast %arg0 to offset:[%offset],sizes:[1,4],strides:[4,1]
   :memref<*xf32> to memref<1x4xf32,strided<[4,1],offset:?>>
 %a=memref.alloc():memref<1x4xf32>
 memref.copy %aext,%a:memref<1x4xf32,strided<[4,1],offset:?>> to memref<1x4xf32>
 %bext=memref.reinterpret_cast %arg1 to offset:[%offset],sizes:[4,VALID],strides:[1,8]
   :memref<*xf32> to memref<4xVALIDxf32,strided<[1,8],offset:?>>
 %bs=memref.alloc():memref<64xf32>
 %b=memref.reinterpret_cast %bs to offset:[0],sizes:[4,16],strides:[1,4]
   :memref<64xf32> to memref<4x16xf32,strided<[1,4]>>
 linalg.fill ins(%zero:f32) outs(%b:memref<4x16xf32,strided<[1,4]>>)
 %bv=memref.subview %b[0,0][4,VALID][1,1]
   :memref<4x16xf32,strided<[1,4]>> to memref<4xVALIDxf32,strided<[1,4]>>
 memref.copy %bext,%bv:memref<4xVALIDxf32,strided<[1,8],offset:?>> to memref<4xVALIDxf32,strided<[1,4]>>
 %partial=memref.alloc():memref<1x16xf32>
 linalg.fill ins(%zero:f32) outs(%partial:memref<1x16xf32>)
 linalg.matmul ins(%a,%b:memref<1x4xf32>,memref<4x16xf32,strided<[1,4]>>) outs(%partial:memref<1x16xf32>)
 linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel"]}
  ins(%acc,%partial:memref<1x16xf32>,memref<1x16xf32>) outs(%acc:memref<1x16xf32>) {
  ^bb0(%a0:f32,%p:f32,%old:f32):
   %v=arith.addf %a0,%p:f32
   linalg.yield %v:f32
 }
}
%cext=memref.reinterpret_cast %arg2 to offset:[0],sizes:[1,VALID],strides:[16,1]
 :memref<*xf32> to memref<1xVALIDxf32,strided<[16,1]>>
%previous=memref.alloc():memref<1x16xf32>
linalg.fill ins(%zero:f32) outs(%previous:memref<1x16xf32>)
%pv=memref.subview %previous[0,0][1,VALID][1,1]
 :memref<1x16xf32> to memref<1xVALIDxf32,strided<[16,1]>>
memref.copy %cext,%pv:memref<1xVALIDxf32,strided<[16,1]>> to memref<1xVALIDxf32,strided<[16,1]>>
linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel"]}
 ins(%acc,%previous:memref<1x16xf32>,memref<1x16xf32>) outs(%acc:memref<1x16xf32>) {
 ^bb0(%a:f32,%p:f32,%old:f32):
  %v=arith.addf %a,%p:f32
  linalg.yield %v:f32
}
%av=memref.subview %acc[0,0][1,VALID][1,1]
 :memref<1x16xf32> to memref<1xVALIDxf32,strided<[16,1]>>
memref.copy %av,%cext:memref<1xVALIDxf32,strided<[16,1]>> to memref<1xVALIDxf32,strided<[16,1]>>
return
}
}
""".replace("VALID", str(valid))


def zero_stride_output_source():
    text = source(16)
    text = text.replace(
        "sizes:[1,16],strides:[16,1]\n :memref<*xf32> to memref<1x16xf32,strided<[16,1]>>",
        "sizes:[1,16],strides:[16,0]\n :memref<*xf32> to memref<1x16xf32,strided<[16,0]>>",
        1,
    )
    text = text.replace(
        "memref.copy %cext,%pv:memref<1x16xf32,strided<[16,1]>> to",
        "memref.copy %cext,%pv:memref<1x16xf32,strided<[16,0]>> to",
        1,
    )
    return text.replace(
        "memref.copy %av,%cext:memref<1x16xf32,strided<[16,1]>> to memref<1x16xf32,strided<[16,1]>>",
        "memref.copy %av,%cext:memref<1x16xf32,strided<[16,1]>> to memref<1x16xf32,strided<[16,0]>>",
        1,
    )


def shifted_previous_source():
    # Both views fit inside the initialized root. A descriptor offset of zero
    # does not make the byte-shifted upper half a prefix of the full snapshot.
    text = source(8).replace(
        "%previous=memref.alloc():memref<1x16xf32>",
        """%raw_previous=memref.alloc():memref<64xi8>
%c32=arith.constant 32:index
%previous=memref.view %raw_previous[%c0][]:memref<64xi8> to memref<1x16xf32>""",
    )
    return text.replace(
        """%pv=memref.subview %previous[0,0][1,8][1,1]
 :memref<1x16xf32> to memref<1x8xf32,strided<[16,1]>>""",
        """%shifted_previous=memref.view %raw_previous[%c32][]:memref<64xi8> to memref<1x8xf32>
%pv=memref.reinterpret_cast %shifted_previous to offset:[0],sizes:[1,8],strides:[16,1]
 :memref<1x8xf32> to memref<1x8xf32,strided<[16,1]>>""",
    )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def invoke(name, text, *passes):
        path = work / (name + ".mlir")
        path.write_text(text)
        return subprocess.run(
            [str(buddy_opt_path()), str(path), *passes],
            text=True,
            capture_output=True,
        )

    for valid in (16, 10, 8):
        original = source(valid)
        planned = invoke(
            "source", original, "--kernel-contract-annotate", "--bosc-ame-plan"
        )
        assert planned.returncode == 0, planned.stderr
        assert "plan.float_schedule" in planned.stdout
        assert f"valid_n = {valid} : i64" in planned.stdout
        emitted = invoke("planned", planned.stdout, "--bosc-ame-emit")
        assert emitted.returncode == 0, emitted.stderr
        assert not any(
            x in emitted.stdout
            for x in ("memref.alloc", "linalg.", "bosc_ame.spm.view")
        )
        oracle = invoke("original", original, "--convert-linalg-to-loops")
        assert oracle.returncode == 0, oracle.stderr
        new = work / "new.mlir"
        new.write_text(emitted.stdout)
        old = work / "old.mlir"
        old.write_text(oracle.stdout)
        for pattern in (
            [
                16777216.0,
                1.0,
                -16777216.0,
                1.0,
                -16777216.0,
                1.0,
                16777216.0,
                1.0,
            ],
            [0.0, -0.0, 1.0, -1.0, 3.25, -4.5, 1e-30, -1e-30],
        ):
            args = {
                "%arg0": bytearray(struct.pack("<8f", *pattern))
                + bytearray([0xA1]) * 64,
                "%arg1": bytearray(
                    struct.pack(
                        "<" + "f" * (valid * 8),
                        *[
                            (-1.0) ** (i % 3) if i % 5 else 0.5
                            for i in range(valid * 8)
                        ],
                    )
                )
                + bytearray([0xB2]) * 64,
                "%arg2": bytearray(
                    struct.pack(
                        "<" + "f" * valid, *[i + 0.25 for i in range(valid)]
                    )
                )
                + bytearray([0xC3]) * 64,
            }
            reference = {k: bytearray(v) for k, v in args.items()}
            machine = Machine(new, "micro")
            machine.launch(args)
            Machine(old, "micro").launch(reference)
            assert args == reference, (
                valid,
                pattern,
                args["%arg2"],
                reference["%arg2"],
            )
            traffic = {}
            for arg, begin, end, kind in machine.accesses:
                traffic[arg, kind] = traffic.get((arg, kind), 0) + end - begin
            assert traffic["%arg0", "read"] == ((valid + 7) // 8) * 8 * 4, (
                traffic
            )
            assert traffic["%arg1", "read"] == valid * 8 * 4, traffic
            assert (
                traffic["%arg2", "read"] == valid * 4
                and traffic["%arg2", "write"] == valid * 4
            ), traffic
        bad = planned.stdout.replace(
            "micro_n = 8 : i64", "micro_n = 16 : i64", 1
        )
        assert invoke("forged", bad, "--bosc-ame-emit").returncode != 0
        # Without a source-stability/noalias proof snapshots stay materialized.
        fallback = invoke(
            "alias",
            original.replace("args_disjoint=true", "args_disjoint=false"),
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
            "--bosc-ame-emit",
        )
        assert fallback.returncode == 0, fallback.stderr
        assert "bosc_ame.float_schedule" not in fallback.stdout
        assert "bosc_ame.spm.view" not in fallback.stdout
        fb = work / "fallback.mlir"
        fb.write_text(fallback.stdout)
        ref = {k: bytearray(v) for k, v in reference.items()}
        expected = {k: bytearray(v) for k, v in reference.items()}
        Machine(fb, "micro").launch(ref)
        Machine(old, "micro").launch(expected)
        assert ref == expected

    # Lane stores may only replace a previous-C snapshot after proving that
    # output lanes do not alias. With stride zero the snapshot must remain.
    for name, original, valid, expected_first in (
        ("zero_stride_output", zero_stride_output_source(), 16, 8.25),
        ("shifted_previous", shifted_previous_source(), 8, 8.0),
    ):
        planned = invoke(
            name + ".source",
            original,
            "--kernel-contract-annotate",
            "--bosc-ame-plan",
        )
        assert "plan.float_schedule" not in planned.stdout, name
        if planned.returncode != 0:
            # A zero-stride boundary has no supported physical route yet.
            # Rejecting it is safe; it must never take the register schedule.
            assert name == "zero_stride_output", planned.stderr
            assert "no legal boundary route" in planned.stderr, planned.stderr
            continue
        emitted = invoke(name + ".planned", planned.stdout, "--bosc-ame-emit")
        assert emitted.returncode == 0, emitted.stderr
        assert "bosc_ame.spm.view" not in emitted.stdout, name
        oracle = invoke(
            name + ".original", original, "--convert-linalg-to-loops"
        )
        assert oracle.returncode == 0, oracle.stderr
        new = work / (name + ".new.mlir")
        new.write_text(emitted.stdout)
        old = work / (name + ".old.mlir")
        old.write_text(oracle.stdout)
        args = {
            "%arg0": bytearray(struct.pack("<8f", *([1.0] * 8)))
            + bytearray([0xA1]) * 64,
            "%arg1": bytearray(
                struct.pack("<" + "f" * (valid * 8), *([1.0] * (valid * 8)))
            )
            + bytearray([0xB2]) * 64,
            "%arg2": bytearray(
                struct.pack("<" + "f" * valid, *([0.25] * valid))
            )
            + bytearray([0xC3]) * 64,
        }
        reference = {key: bytearray(value) for key, value in args.items()}
        Machine(new, "micro").launch(args)
        Machine(old, "micro").launch(reference)
        assert args == reference, name
        assert struct.unpack_from("<f", args["%arg2"])[0] == expected_first, (
            name
        )

    # The inner dot payload and both outer sums have separate rounding
    # properties. The replacement sums must clone explicit source properties.
    original = (
        source(16)
        .replace("%v=arith.addf %a0,%p:f32", "%v=arith.addf %a0,%p upward:f32")
        .replace("%v=arith.addf %a,%p:f32", "%v=arith.addf %a,%p upward:f32")
    )
    planned = invoke(
        "rounding.source",
        original,
        "--kernel-contract-annotate",
        "--bosc-ame-plan",
    )
    assert planned.returncode == 0, planned.stderr
    assert "plan.float_schedule" in planned.stdout
    emitted = invoke("rounding.planned", planned.stdout, "--bosc-ame-emit")
    assert emitted.returncode == 0, emitted.stderr
    assert (
        len(re.findall(r"arith\.addf[^\n]* upward : f32", emitted.stdout)) == 32
    )
    print(
        "PASS ordered float microtiles, partial rounding, tail guards, DDR fallback, shifted prefixes, rounding properties"
    )


if __name__ == "__main__":
    main()
