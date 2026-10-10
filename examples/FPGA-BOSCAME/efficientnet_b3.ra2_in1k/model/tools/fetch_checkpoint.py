#!/usr/bin/env python3
# ===- fetch_checkpoint.py - download EfficientNet-B3 weights ---*- python -*-===//
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
# Download timm/efficientnet_b3.ra2_in1k into the caller-owned output
# directory. Weights are large and must stay out of git (see
# ../../.gitignore). Prefer HF_ENDPOINT=https://hf-mirror.com. When a
# 40-character commit is known, pass --revision; do not pin `main`.
#
# ===----------------------------------------------------------------------===//
"""Download the pinned timm EfficientNet-B3 checkpoint via Hugging Face / hf-mirror."""
import argparse
import os
from pathlib import Path

REPOSITORY = "timm/efficientnet_b3.ra2_in1k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--revision", default=None,
                        help="pin a 40-character commit when known; do not use main")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download
    snapshot_download(
        args.repository,
        local_dir=str(args.output),
        revision=args.revision,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.md"],
    )
    print("downloaded", args.repository, "to", args.output)


if __name__ == "__main__":
    main()
