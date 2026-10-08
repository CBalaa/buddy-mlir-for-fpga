#!/usr/bin/env python3
# ===- host_reference.py - FP32 logits for board verify ---------*- python -*-===//
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
# Official timm forward on the same fused Conv+BN model. --board-input
# uses import_model.board_input() so the NR launch can compare argmax
# without uploading an image tensor.
#
# ===----------------------------------------------------------------------===//
"""FP32 reference logits from the official timm EfficientNet-B3 checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--image", type=Path, default=None,
                        help="optional NCHW float32 .npy; otherwise a fixed seed")
    parser.add_argument("--board-input", action="store_true",
                        help="use the same deterministic NCHW fill as the NR launch")
    parser.add_argument("--stages", type=int, default=None)
    args = parser.parse_args()

    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import import_model as im

    model = im.build_model(args.checkpoint, args.stages)
    if args.image is not None:
        array = np.load(args.image)
        data = torch.from_numpy(array).to(torch.float32)
    elif args.board_input:
        data = torch.from_numpy(im.board_input(args.input_size))
    else:
        torch.manual_seed(0)
        data = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        logits = model(data)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "input.npy", data.cpu().numpy())
    np.save(args.output / "logits.npy", logits.cpu().numpy())
    max_logit = logits.max().detach().cpu().to(torch.float32)
    max_bits = int(np.frombuffer(np.array(max_logit.item(), dtype=np.float32).tobytes(),
                                 dtype=np.uint32)[0])
    report = {
        "input_shape": list(data.shape),
        "logits_shape": list(logits.shape),
        "logits_max": float(max_logit.item()),
        "logits_max_bits": max_bits,
        "predicted_class": int(logits.argmax(dim=-1).item()) if logits.ndim >= 2 else None,
        "stages": args.stages,
        "fused_conv_bn": True,
        "board_input": bool(args.board_input),
    }
    (args.output / "host-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
