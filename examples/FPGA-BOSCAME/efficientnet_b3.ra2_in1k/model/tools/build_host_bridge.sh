#!/usr/bin/env bash
# ===- build_host_bridge.sh - host SO for replaced-graph EE --------*- sh -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# Same role as qwen3-0.6b/model/tools/build_host_bridge.sh: host kernels
# from the Triton build tree + graph-facing adapters, linked as one
# shared library for Buddy ExecutionEngine.
#
# ===----------------------------------------------------------------------===//
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
MODEL="$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k/model"
EFF="$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k"
TRITON_BUILD="${TRITON_BUILD:-$EFF/triton/build}"
CC="${HOST_CC:-$REPO/llvm/build-2d26/bin/clang}"
OUT="${OUT:-$MODEL/build/host-bridge}"
ADAPTERS="${ADAPTERS:-$MODEL/build/replacement/effnet_triton_adapters.c}"

if [[ ! -f "$ADAPTERS" ]]; then
  echo "missing generated adapters: $ADAPTERS (run triton_call_replace.py)" >&2
  exit 1
fi

CASES=$(grep -o '_mlir_ciface_kernel_[a-z0-9_]*' "$ADAPTERS" \
        | sed 's/^_mlir_ciface_kernel_//' | sort -u)
if [[ -z "$CASES" ]]; then
  echo "no archive entries referenced by $ADAPTERS" >&2
  exit 1
fi

rm -rf "$OUT" && mkdir -p "$OUT"
FLAGS=(-O2 -fPIC -ffp-contract=off -DHOST_TEST -I"$EFF")

objects=()
count=0
for case in $CASES; do
  if [[ ! -f "$TRITON_BUILD/$case/host/kernel.ll" ]]; then
    echo "missing host kernel IR for $case under $TRITON_BUILD" >&2
    exit 1
  fi
  "$CC" "${FLAGS[@]}" -c "$TRITON_BUILD/$case/host/kernel.ll" \
    -o "$OUT/$case.kernel.o"
  "$CC" "${FLAGS[@]}" -c "$TRITON_BUILD/$case/adapter.c" \
    -o "$OUT/$case.ciface.o"
  objects+=("$OUT/$case.kernel.o" "$OUT/$case.ciface.o")
  count=$((count + 1))
done

"$CC" "${FLAGS[@]}" -c "$ADAPTERS" -o "$OUT/effnet_triton_adapters.o"
objects+=("$OUT/effnet_triton_adapters.o")

"$CC" -shared -o "$OUT/libeffnet_triton_host.so" "${objects[@]}" -lm

echo "$count kernels -> $OUT/libeffnet_triton_host.so"
"$REPO/llvm/build-2d26/bin/llvm-nm" --dynamic --defined-only \
  "$OUT/libeffnet_triton_host.so" | grep -c " T " || true
