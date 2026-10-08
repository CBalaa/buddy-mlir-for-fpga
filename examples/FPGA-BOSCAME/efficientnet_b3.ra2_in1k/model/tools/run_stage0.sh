#!/usr/bin/env bash
# ===- run_stage0.sh - truncated-stage smoke through prepared image -*- sh -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# Same stages as run_full.sh but --stages 1 for a cheap bring-up before
# the full network. Uses committed model-cases/ by default (REEMIT=1 to
# refresh). Output uses Qwen-style build/<config>:
#   CONFIG=stage0 (default) → model/build/stage0/
#   OUT overrides CONFIG when set.
#
# ===----------------------------------------------------------------------===//
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EFF="$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k"
MODEL="$EFF/model"
CONFIG="${CONFIG:-stage0}"
OUT="${OUT:-$MODEL/build/$CONFIG}"
STAGES="${STAGES:-1}"
FPGA="${FPGA:-5}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-900}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
RUN_FPGA="${RUN_FPGA:-0}"
export PYTHONSAFEPATH=1
if [[ -f "$EFF/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$EFF/env.sh"
fi
export TRITON_PYTHON="${TRITON_PYTHON:-/home/caihanyi/miniconda3/envs/boscame/bin/python}"
export BUDDY_PYTHON="${BUDDY_PYTHON:-$TRITON_PYTHON}"
export LLVM_SYSPATH="${LLVM_SYSPATH:-$REPO/llvm/build-2d26}"
export LLVM_BIN="${LLVM_BIN:-$LLVM_SYSPATH/bin}"
export BUDDY_MLIR_BINARY_DIR="${BUDDY_MLIR_BINARY_DIR:-$REPO/build-migrate/bin}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$OUT"
echo "OUT=$OUT (CONFIG=$CONFIG)"
CHECKPOINT="${CHECKPOINT:-$MODEL/assets/checkpoint}"
if [[ ! -f "$CHECKPOINT/model.safetensors" && ! -f "$CHECKPOINT/pytorch_model.bin" ]]; then
  "$BUDDY_PYTHON" -B "$MODEL/tools/fetch_checkpoint.py" --output "$CHECKPOINT"
fi

"$BUDDY_PYTHON" -B "$MODEL/tools/import_model.py" \
  --checkpoint "$CHECKPOINT" --output "$OUT/import" --input-size 288 --stages "$STAGES"

"$BUDDY_PYTHON" -B "$MODEL/tools/operator_inventory.py" \
  --ops "$OUT/import/raw-ops.json" --output "$OUT/operator-inventory.json"

MODEL_CASES="${MODEL_CASES:-$EFF/model-cases}"
REEMIT="${REEMIT:-0}"
if [[ "$REEMIT" == "1" || ! -f "$MODEL_CASES/case-index.json" ]]; then
  "$BUDDY_PYTHON" -B "$MODEL/tools/emit_model_cases.py" \
    --ops "$OUT/import/raw-ops.json" --output "$MODEL_CASES"
fi

export EFFNET_CASE_ROOTS="$MODEL_CASES"
export EFFNET_TRITON_BUILD_ROOT="$OUT/triton-build"
cd "$EFF/triton"
"$TRITON_PYTHON" -P -B build.py --jobs "${JOBS:-4}" --host
"$TRITON_PYTHON" -P -B build.py --jobs "${JOBS:-4}"

"$BUDDY_PYTHON" -B "$MODEL/tools/triton_call_replace.py" \
  --checkpoint "$CHECKPOINT" --triton-build "$OUT/triton-build" \
  --output "$OUT/replacement" --input-size 288 --stages "$STAGES"

python3 - "$OUT/replacement/triton-call-replacement.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("uncovered"):
    raise SystemExit("graph replacement still has uncovered ops; see " + sys.argv[1])
if report.get("status") != "PASS":
    raise SystemExit("graph replacement status is " + str(report.get("status")))
print("replacement PASS,", len(report.get("distinct_symbols", [])), "symbols")
PY

"$BUDDY_PYTHON" -B "$MODEL/tools/host_reference.py" \
  --checkpoint "$CHECKPOINT" --output "$OUT/fp32-reference" \
  --input-size 288 --board-input --stages "$STAGES"

"$BUDDY_PYTHON" -B "$MODEL/tools/lower_model_nr.py" \
  --checkpoint "$CHECKPOINT" --triton-build "$OUT/triton-build" \
  --output "$OUT/nr-forward" --input-size 288 --stages "$STAGES"

ADAPTERS="$OUT/replacement/effnet_triton_adapters.c" \
  TRITON_BUILD="$OUT/triton-build" \
  OUT="$OUT/model-lib" \
  bash "$MODEL/tools/build_model_lib.sh"

"$TRITON_PYTHON" -P -B "$MODEL/tools/build_model_image.py" \
  --repo-root "$REPO" \
  --layout "$OUT/nr-forward/weight-layout.json" \
  --graph-ir "$OUT/nr-forward/forward.ll" \
  --archive "$OUT/model-lib/libeffnet_triton.a" \
  --adapters "$OUT/replacement/effnet_triton_adapters.c" \
  --reference "$OUT/fp32-reference/host-reference.json" \
  --output "$OUT/image"
# Default: [model]/[op]/[kernel] UART progress (see build_model_image.py).

"$TRITON_PYTHON" -P -B "$MODEL/tools/prepare_model_run.py" \
  --image "$OUT/image/effnet_model.bin" \
  --elf "$OUT/image/effnet_model.elf" \
  --weights "$OUT/nr-forward/weights.bin" \
  --weight-manifest "$OUT/nr-forward/weight-layout.json" \
  --output "$OUT/prepared"

echo "prepared: $OUT/prepared"
if [[ "$RUN_FPGA" == "1" ]]; then
  "$MODEL/tools/run_model.sh" "$OUT/prepared" \
    --fpga="$FPGA" --capture-seconds="$CAPTURE_SECONDS" \
    --startup-timeout="$STARTUP_TIMEOUT"
fi
