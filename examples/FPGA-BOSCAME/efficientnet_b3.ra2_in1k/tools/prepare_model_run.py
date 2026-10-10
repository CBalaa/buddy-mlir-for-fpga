#!/usr/bin/env python3
# ===- prepare_model_run.py - image + weights DDR plan ----------*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Verify the linked ELF exports weight_arena_raw / _end, copy image.bin
# and weights.bin, write ddr-load.plan for fpga_run.sh. Same layout as
# qwen3-0.6b/model/tools/prepare_model_run.py.
#
# ===----------------------------------------------------------------------===//
"""Prepare checked DDR segments from the linked EfficientNet image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def aligned(size):
    return (size + 63) & ~63


def check_image_bytes(image, extracted):
    expected_size = extracted.stat().st_size
    actual_size = image.stat().st_size
    if not expected_size or actual_size not in (expected_size, aligned(expected_size)):
        raise ValueError("boot bin length does not match this ELF")
    with image.open("rb") as actual, extracted.open("rb") as expected:
        while block := expected.read(1024 * 1024):
            if actual.read(len(block)) != block:
                raise ValueError("boot bin bytes do not match this ELF")
        if any(actual.read()):
            raise ValueError("boot bin has nonzero alignment padding")
    return expected_size


def check_arena(symbols, stem, size):
    begin_name, end_name = stem + "_raw", stem + "_end"
    if begin_name not in symbols or end_name not in symbols:
        raise ValueError(f"missing {begin_name}/{end_name}")
    begin, end = symbols[begin_name], symbols[end_name]
    if (size <= 0 or begin % 64 or end % 64 or
            not symbols["__workspace_start"] <= begin < end <= symbols["__workspace_end"]):
        raise ValueError(f"{stem}: invalid linked arena")
    if aligned(size) != end - begin:
        raise ValueError(
            f"{stem}: padded size {aligned(size)} differs from arena {end - begin}")
    return begin, end


def main():
    tools = Path(__file__).resolve().parents[4] / "llvm/build-2d26/bin"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weight-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nm", type=Path, default=tools / "llvm-nm")
    parser.add_argument("--objcopy", type=Path, default=tools / "llvm-objcopy")
    args = parser.parse_args()
    manifest = json.loads(args.weight_manifest.read_text())
    if args.weights.stat().st_size != manifest["bytes"] or digest(args.weights) != manifest["sha256"]:
        raise ValueError("weight segment does not match its manifest")
    raw = subprocess.check_output([str(args.nm), "--defined-only", str(args.elf)], text=True)
    symbols = {fields[2]: int(fields[0], 16) for line in raw.splitlines()
               if len(fields := line.split()) == 3}
    if symbols.get("_start") != 0x80000000:
        raise ValueError("unexpected boot address")
    if not 0xb8000000 <= symbols["__workspace_start"] < symbols["__workspace_end"] <= 0x100000000:
        raise ValueError("workspace exceeds NR linker address range")
    with tempfile.TemporaryDirectory(prefix="effnet-elf-bin-") as temporary:
        extracted = Path(temporary) / "image.bin"
        subprocess.run([str(args.objcopy), "-O", "binary", str(args.elf), str(extracted)],
                       check=True)
        raw_image_bytes = check_image_bytes(args.image, extracted)
    weight_begin, weight_end = check_arena(symbols, "weight_arena", manifest["bytes"])
    inputs = [
        ("model", args.image, "image.bin", symbols["_start"], None),
        ("weights", args.weights, "weights.bin", weight_begin, weight_end),
    ]
    ranges = []
    for name, source, filename, address, end in inputs:
        size = aligned(source.stat().st_size)
        if address % 64 or not 0x80000000 <= address < address + size <= 0x100000000:
            raise ValueError(f"{name} is outside NR address range")
        if address < 0xb8000000 and address + size > 0xb0000000:
            raise ValueError(f"{name} overlaps RA inaccessible aperture")
        if end is not None and address + size != end:
            raise ValueError(f"{name} exceeds its linked arena")
        ranges.append((address, address + size))
    ranges.sort()
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("segments overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    segments = []
    for name, source, filename, address, end in inputs:
        dest = args.output / filename
        shutil.copyfile(source, dest)
        size = dest.stat().st_size
        with dest.open("ab") as handle:
            handle.write(b"\0" * (-size % 64))
        item = dict(name=name, file=filename, address=address,
                    size=dest.stat().st_size, sha256=digest(dest))
        if end is not None:
            item["arena_end"] = end
        segments.append(item)
    text = ["# Platform loader DDR window; segments also satisfy NR linker bounds.",
            "version = 1", "ddr_base = 0x80000000", "ddr_size = 0x400000000",
            "alignment = 64"]
    for item in segments:
        text += ["", "[[segments]]", f'name = "{item["name"]}"',
                 f'file = "{item["file"]}"', f'address = 0x{item["address"]:x}',
                 f'size = {item["size"]}', f'sha256 = "{item["sha256"]}"']
    (args.output / "ddr-load.plan").write_text("\n".join(text) + "\n")
    (args.output / "deployment.json").write_text(json.dumps({
        "elf": str(args.elf.resolve()), "elf_sha256": digest(args.elf),
        "segments": segments, "boot_bin_matches_elf": True,
        "elf_objcopy_bytes": raw_image_bytes,
        "per_resource_arena_bounds_verified": True,
        "workspace_end": hex(symbols["__workspace_end"]),
        "execution": "prepared only; no hardware result",
    }, indent=2) + "\n")
    print(args.output / "ddr-load.plan")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as error:
        raise SystemExit(str(error))
