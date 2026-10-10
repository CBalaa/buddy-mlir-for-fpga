#!/usr/bin/env bash
# ===- run_model.sh - upload EfficientNet image and stream UART ------*- sh -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# Uploads image.bin plus the weights.bin DDR segment via ../../../fpga_run.sh.
# Success on UART: "verify efficientnet_b3.ra2_in1k: PASS" and
# "[nr] RA returned: PASS". FPGA 0-3 are TB1, 4-7 are TB2; gclk6 is
# board-level and must be free.
#
# ===----------------------------------------------------------------------===//
set -euo pipefail
model_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 1 ]]; then
  echo 'usage: run_model.sh PREPARED_IMAGE_DIR [fpga_run options, e.g. --fpga=5]' >&2
  exit 2
fi
run_dir=$1
shift
exec "$model_dir/../../../fpga_run.sh" "$run_dir/image.bin" \
  --layout-plan "$run_dir/ddr-load.plan" \
  --segment "$run_dir/weights.bin" \
  --completion-marker='[nr] RA returned:' "$@"
