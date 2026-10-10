#!/usr/bin/env python3
# ===- fetch_assets.py - vision checkpoint assets ----------------*- python -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
"""Fetch EfficientNet checkpoint weights (no tokenizer).

Qwen's ``fetch_assets.py`` downloads tokenizer JSON only. Vision models use
``fetch_checkpoint.py`` for ``model.safetensors``. This wrapper keeps the same
tool path for deployment parity.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="checkpoint directory (default layout for import_model.py)")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, "-B", str(here / "fetch_checkpoint.py"),
           "--output", str(args.output)]
    subprocess.run(cmd, check=True)
    manifest = {
        "model": "timm/efficientnet_b3.ra2_in1k",
        "weights_downloaded": True,
        "tokenizer_downloaded": False,
        "note": "Vision deployment has no tokenizer blob; see model/text/README.md",
    }
    (args.output / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
