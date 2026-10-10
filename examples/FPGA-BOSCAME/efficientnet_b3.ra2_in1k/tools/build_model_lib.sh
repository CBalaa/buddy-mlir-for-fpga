#!/usr/bin/env bash
# ===- build_model_lib.sh - adapters to libeffnet_triton.a -----------*- sh -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# Discover _mlir_ciface_kernel_* names from the generated adapters and
# archive the matching Triton NR objects. Set TRITON_BUILD to the tree
# that actually compiled those cases (run_full.sh uses
# model/build/fp32-full/triton-build (or CONFIG=... path), not triton/build).
#
# ===----------------------------------------------------------------------===//
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODEL="$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k/model"
TRITON_BUILD="${TRITON_BUILD:-$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k/triton/build}"
LLVM_BIN="${LLVM_BIN:-$REPO/llvm/build-2d26/bin}"
OUT="${OUT:-$MODEL/build/model-lib}"
ADAPTERS="${ADAPTERS:-$MODEL/build/replacement/effnet_triton_adapters.c}"

if [[ ! -f "$ADAPTERS" ]]; then
  echo "missing generated adapters: $ADAPTERS" >&2
  exit 1
fi
CASES=()
while read -r case; do
  [[ -n "$case" ]] && CASES+=(--case "$case")
done < <(grep -o '_mlir_ciface_kernel_[a-z0-9_]*' "$ADAPTERS" \
           | sed 's/_mlir_ciface_kernel_//' | sort -u)
if [[ ${#CASES[@]} == 0 ]]; then
  echo "no Triton kernel references in $ADAPTERS" >&2
  exit 1
fi
echo "discovered $((${#CASES[@]} / 2)) kernels from $(basename "$ADAPTERS")"
rm -rf "$OUT"
python3 -B "$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k/tools/archive_kernels.py" \
  --triton-build "$TRITON_BUILD" "${CASES[@]}" \
  --output "$OUT" --ar "$LLVM_BIN/llvm-ar" --nm "$LLVM_BIN/llvm-nm"
echo "archive: $OUT/libeffnet_triton.a"
