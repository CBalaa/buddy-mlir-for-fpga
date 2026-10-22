# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Only fuse a zero-initialized, adjacent, unmasked previous-C chain."""

import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/FPGA-BOSCAME/tools"),
)
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402

SOURCE = """module {
  func.func @gemm(%a: memref<64x64xi8>, %b: memref<64x64xi8, strided<[1, 64]>>, %c: @ARG_TYPE@) {
    %p = memref.alloc() alignment = 64 : memref<64x64xi32>
    %s = memref.alloc() alignment = 64 : memref<64x64xi32>
    %initial = arith.constant @INITIAL@ : i32
    %seven = arith.constant 7 : i32
    %idx0 = arith.constant 0 : index
    @VIEWS@
    linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"], bosc_ame.memory = {kind = "load", logical_shape = array<i64: 64, 64>, valid_shape = array<i64: 64, 64>, masked = @LOAD_MASK@}} ins(@LOAD@ : @LOAD_TYPE@) outs(%s : memref<64x64xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    @SIDE_EFFECT@
    linalg.fill ins(%initial : i32) outs(%p : memref<64x64xi32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame} ins(%a, %b : memref<64x64xi8>, memref<64x64xi8, strided<[1, 64]>>) outs(%p : memref<64x64xi32>)
    linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%p, %s : memref<64x64xi32>, memref<64x64xi32>) outs(%s : memref<64x64xi32>) {
    ^bb0(%left: i32, %right: i32, %out: i32):
      %sum = arith.addi %left, %right : i32
      linalg.yield %sum : i32
    }
    linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"], bosc_ame.memory = {kind = "store", destination = "external_ddr", logical_shape = array<i64: 64, 64>, valid_shape = array<i64: 64, 64>, masked = @STORE_MASK@}} ins(%s : memref<64x64xi32>) outs(@DESTINATION@ : @DESTINATION_TYPE@) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    memref.dealloc %s : memref<64x64xi32>
    memref.dealloc %p : memref<64x64xi32>
    return
  }
}
"""

# Match the frontend's actual bufferized spelling: a pitched snapshot view,
# product allocation after the load, and an unannotated handoff after the add.
BUFFER_SOURCE = """module attributes {bosc_ame.placement = "logical"} {
  func.func @gemm(%a_storage: memref<8192xi8>, %b_storage: memref<8192xi8>, %c: memref<65536xi32>, %program: i32) {
    %zero = arith.constant 0 : i32
    %seven = arith.constant 7 : i32
    %idx0 = arith.constant 0 : index
    %idx64 = arith.constant 64 : index
    %a = memref.reinterpret_cast %a_storage to offset: [0], sizes: [64, 128], strides: [128, 1] : memref<8192xi8> to memref<64x128xi8, strided<[128, 1]>>
    %b = memref.reinterpret_cast %b_storage to offset: [0], sizes: [128, 64], strides: [1, 128] : memref<8192xi8> to memref<128x64xi8, strided<[1, 128]>>
    %program_index = arith.index_cast %program : i32 to index
    %offset = arith.muli %program_index, %idx64 : index
    %destination = memref.reinterpret_cast %c to offset: [%offset], sizes: [64, 64], strides: [1024, 1] : memref<65536xi32> to memref<64x64xi32, strided<[1024, 1], offset: ?>>
    %snapshot_base = memref.alloc() alignment = 64 : memref<64576xi32>
    %snapshot = memref.reinterpret_cast %snapshot_base to offset: [0], sizes: [64, 64], strides: [1024, 1] : memref<64576xi32> to memref<64x64xi32, strided<[1024, 1]>>
    @ALIASES@
    @PRODUCT_BEFORE_LOAD@
    memref.copy %destination, %snapshot {bosc_ame.memory = {kind = "load", logical_shape = array<i64: 64, 64>, valid_shape = array<i64: 64, 64>, masked = false, origin_id = 11 : i64}} : memref<64x64xi32, strided<[1024, 1], offset: ?>> to memref<64x64xi32, strided<[1024, 1]>>
    @LOAD_GAP@
    @PRODUCT_AFTER_LOAD@
    linalg.fill ins(%zero : i32) outs(%product : memref<64x64xi32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame} ins(%a, %b : memref<64x128xi8, strided<[128, 1]>>, memref<128x64xi8, strided<[1, 128]>>) outs(%product : memref<64x64xi32>)
    linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} ins(%snapshot, %product : memref<64x64xi32, strided<[1024, 1]>>, memref<64x64xi32>) outs(%snapshot : memref<64x64xi32, strided<[1024, 1]>>) {
    ^bb0(%left: i32, %right: i32, %out: i32):
      %sum = arith.addi %left, %right : i32
      linalg.yield %sum : i32
    }
    @STORE_GAP@
    %bridge = memref.alloc() alignment = 64 : memref<64x64xi32>
    memref.copy %snapshot, %bridge : memref<64x64xi32, strided<[1024, 1]>> to memref<64x64xi32>
    @BRIDGES@
    memref.copy @STORE_SOURCE@, %destination {bosc_ame.memory = {kind = "store", destination = "external_ddr", logical_shape = array<i64: 64, 64>, valid_shape = array<i64: 64, 64>, masked = false, origin_id = 12 : i64}} : memref<64x64xi32> to memref<64x64xi32, strided<[1024, 1], offset: ?>>
    @CONSUMERS@
    return
  }
}
"""


def lower_source(work, name, source, plan_report=False):
    path = work / f"{name}.mlir"
    path.write_text(source)
    report = " plan-report=true" if plan_report else ""
    return subprocess.run(
        [
            buddy_opt_path(),
            str(path),
            "--linalg-to-boscame=enable-spm-dma=true enable-ame=true "
            "dma-abi=args_aligned64_nonalias target=nr-fpga" + report,
        ],
        capture_output=True,
        text=True,
    )


def lower(work, name, **changes):
    values = {
        "ARG_TYPE": "memref<64x64xi32>",
        "INITIAL": "0",
        "VIEWS": "",
        "LOAD_MASK": "false",
        "LOAD": "%c",
        "LOAD_TYPE": "memref<64x64xi32>",
        "SIDE_EFFECT": "",
        "STORE_MASK": "false",
        "DESTINATION": "%c",
        "DESTINATION_TYPE": "memref<64x64xi32>",
    }
    values.update(changes)
    source = SOURCE
    for key, value in values.items():
        source = source.replace(f"@{key}@", value)
    return lower_source(work, name, source)


def lower_bufferized(work, name, plan_report=False, **changes):
    values = {
        "ALIASES": "",
        "PRODUCT_BEFORE_LOAD": "",
        "PRODUCT_AFTER_LOAD": "%product = memref.alloc() alignment = 64 : memref<64x64xi32>",
        "LOAD_GAP": "",
        "STORE_GAP": "",
        "BRIDGES": "",
        "STORE_SOURCE": "%bridge",
        "CONSUMERS": "",
    }
    values.update(changes)
    source = BUFFER_SOURCE
    for key, value in values.items():
        source = source.replace(f"@{key}@", value)
    assert not re.search(r"@[A-Z_]+@", source)
    return lower_source(work, name, source, plan_report=plan_report)


def check_bufferized(work):
    result = lower_bufferized(work, "bufferized")
    check_success(result, fused=True)
    assert "bosc_ame.dma.start" not in result.stdout, result.stdout
    assert "memref.alloc" not in result.stdout, result.stdout
    assert "memref.copy" not in result.stdout, result.stdout
    assert "linalg.generic" not in result.stdout, result.stdout
    reported = lower_bufferized(
        work, "bufferized_plan_report", plan_report=True
    )
    check_success(reported, fused=True)
    assert "buffer-source-events: count=2 scope=pre-accumulator-fusion" in (
        reported.stderr
    ), reported.stderr
    assert re.search(
        r"buffer-source-event\[\d+\]: function=@gemm source-origin-id=11 "
        r"op=linalg.generic effect=read",
        reported.stderr,
    ), reported.stderr
    assert re.search(
        r"buffer-source-event\[\d+\]: function=@gemm source-origin-id=12 "
        r"op=linalg.generic effect=write",
        reported.stderr,
    ), reported.stderr
    post_planning_accesses = reported.stderr.split("buffer-accesses:", 1)[1]
    assert not re.search(
        r"source-origin-id=(?:11|12)\b", post_planning_accesses
    )
    # A second private handoff must allow each copy as both a producer and a
    # consumer of its intermediate buffer.
    chained = lower_bufferized(
        work,
        "bufferized_multilevel",
        BRIDGES="""%bridge2 = memref.alloc() alignment = 64 : memref<64x64xi32>
    memref.copy %bridge, %bridge2 : memref<64x64xi32> to memref<64x64xi32>""",
        STORE_SOURCE="%bridge2",
    )
    check_success(chained, fused=True)
    assert "bosc_ame.dma.start" not in chained.stdout, chained.stdout
    load_after_product = lower_bufferized(
        work,
        "bufferized_load_after_product_alloc",
        PRODUCT_BEFORE_LOAD="%product = memref.alloc() alignment = 64 : memref<64x64xi32>",
        PRODUCT_AFTER_LOAD="",
    )
    check_success(load_after_product, fused=True)
    assert "bosc_ame.dma.start" not in load_after_product.stdout

    cases = {
        "snapshot_consumer": {
            "CONSUMERS": "memref.store %seven, %snapshot[%idx0, %idx0] : memref<64x64xi32, strided<[1024, 1]>>"
        },
        "bridge_consumer": {
            "CONSUMERS": "memref.store %seven, %bridge[%idx0, %idx0] : memref<64x64xi32>"
        },
        "snapshot_alias": {
            "ALIASES": "%alias = memref.reinterpret_cast %snapshot_base to offset: [0], sizes: [64, 64], strides: [1024, 1] : memref<64576xi32> to memref<64x64xi32, strided<[1024, 1]>>",
            "CONSUMERS": "memref.store %seven, %alias[%idx0, %idx0] : memref<64x64xi32, strided<[1024, 1]>>",
        },
        "load_side_effect": {
            "LOAD_GAP": "memref.store %seven, %destination[%idx0, %idx0] : memref<64x64xi32, strided<[1024, 1], offset: ?>>"
        },
        "store_side_effect": {
            "STORE_GAP": "memref.store %seven, %destination[%idx0, %idx0] : memref<64x64xi32, strided<[1024, 1], offset: ?>>"
        },
    }
    for name, changes in cases.items():
        check_success(lower_bufferized(work, name, **changes), fused=False)

    # A and B use signed i8 values, C starts near both i32 limits, and two
    # 64-wide reductions cross those limits. The oracle wraps only at i32.
    lowered = work / "bufferized.lowered.mlir"
    lowered.write_text(result.stdout)
    a = [
        [((i * 13 + k * 7) % 256) - 128 for k in range(128)] for i in range(64)
    ]
    b = [
        [((j * 19 + k * 11) % 256) - 128 for k in range(128)] for j in range(64)
    ]
    guard = bytes([0xB7]) * 256
    output = bytearray(bytes([0xD3]) * (65536 * 4) + guard)
    expected = bytearray(output)
    for i in range(64):
        for j in range(64):
            previous = 0x7FFFFFF0 if (i + j) % 2 == 0 else -0x7FFFFFF0
            position = (i * 1024 + j) * 4
            struct.pack_into("<i", output, position, previous)
            total = previous + sum(a[i][k] * b[j][k] for k in range(128))
            wrapped = (total + (1 << 31)) % (1 << 32) - (1 << 31)
            struct.pack_into("<i", expected, position, wrapped)
    machine = Machine(lowered, "gemm")
    machine.launch(
        {
            "%arg0": bytearray(value % 256 for row in a for value in row),
            "%arg1": bytearray(value % 256 for row in b for value in row),
            "%arg2": output,
        },
        {"%arg3": 0},
    )
    assert output == expected
    assert machine.dma_log == []


def check_success(result, fused):
    assert result.returncode == 0, result.stderr
    direct = re.findall(
        r"bosc_ame.tile\.(?:load|store)_acc[^\n]*bosc_ame.direct_ddr",
        result.stdout,
    )
    assert len(direct) == (2 if fused else 0), result.stdout


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    check_bufferized(work)
    check_success(lower(work, "canonical"), fused=True)
    nonzero = lower(work, "nonzero", INITIAL="7")
    check_success(nonzero, fused=False)
    assert "arith.constant dense<7> : vector<16xi32>" in nonzero.stdout
    lowered = work / "nonzero.lowered.mlir"
    lowered.write_text(nonzero.stdout)
    previous = (0x7FFFFFFE).to_bytes(4, "little", signed=True)
    output = bytearray(previous * (64 * 64))
    Machine(lowered, "gemm").launch(
        {
            "%arg0": bytearray(64 * 64),
            "%arg1": bytearray(64 * 64),
            "%arg2": output,
        }
    )
    # Zero A/B leave the nonzero product initializer intact. Adding it to C
    # also checks that the unfused path retains exact signed i32 wrapping.
    expected = (-2147483643).to_bytes(4, "little", signed=True)
    assert output == expected * (64 * 64)
    check_success(lower(work, "masked_load", LOAD_MASK="true"), fused=False)
    check_success(
        lower(
            work,
            "side_effect",
            SIDE_EFFECT="memref.store %seven, %c[%idx0, %idx0] : memref<64x64xi32>",
        ),
        fused=False,
    )
    view_type = "memref<64x64xi32, strided<[64, 1]>>"
    check_success(
        lower(
            work,
            "different_views",
            ARG_TYPE="memref<128x64xi32>",
            VIEWS="""%before = memref.subview %c[0, 0] [64, 64] [1, 1] : memref<128x64xi32> to memref<64x64xi32, strided<[64, 1]>>
    %after = memref.subview %c[64, 0] [64, 64] [1, 1] : memref<128x64xi32> to memref<64x64xi32, strided<[64, 1], offset: 4096>>""",
            LOAD="%before",
            LOAD_TYPE=view_type,
            DESTINATION="%after",
            DESTINATION_TYPE="memref<64x64xi32, strided<[64, 1], offset: 4096>>",
        ),
        fused=False,
    )
    masked = lower(work, "masked_store", STORE_MASK="true")
    assert masked.returncode != 0
    assert "direct-ddr store does not support masked stores" in masked.stderr
    # CHECK: PASS direct AME fusion: zero initialization, identity, masks and order
    print(
        "PASS direct AME fusion: zero initialization, identity, masks and order"
    )


if __name__ == "__main__":
    main()
