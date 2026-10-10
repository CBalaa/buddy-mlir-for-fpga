#!/usr/bin/env bash
# ===- run_full.sh - import through prepared image -------------------*- sh -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# Full FPGA-BOSCAME model path (same stages as qwen3-0.6b/model/tools):
#   fetch → import → Triton host+NR → replace → reference → lower →
#   archive → image → prepare. Default RUN_FPGA=0 (build only).
# Committed cases live next to this script's example. Set REEMIT=1 to refresh
# them from this import's raw-ops.json. Sources $EFF/env.sh first so
# BUDDY_PYTHON matches this-repo bindings.
#
# Output layout matches Qwen's build/<config> mode:
#   CONFIG=fp32-full              → model/build/fp32-full/
#   CONFIG=ame-v05/fp32-full      → model/build/ame-v05/fp32-full/
#   OUT=/abs/path                 → overrides CONFIG
# Legacy tree model/build/full is still usable via CONFIG=full or OUT=...
#
# ===----------------------------------------------------------------------===//
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EFF="$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k"
MODEL="$EFF/model"
CONFIG="${CONFIG:-fp32-full}"
OUT="${OUT:-$MODEL/build/$CONFIG}"
FPGA="${FPGA:-5}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-1800}"
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
  "$BUDDY_PYTHON" -B "$EFF/tools/fetch_checkpoint.py" --output "$CHECKPOINT"
fi

"$BUDDY_PYTHON" -B "$EFF/tools/import_model.py" \
  --checkpoint "$CHECKPOINT" --output "$OUT/import" --input-size 288

"$BUDDY_PYTHON" -B "$EFF/tools/operator_inventory.py" \
  --ops "$OUT/import/raw-ops.json" --output "$OUT/operator-inventory.json"

# Static full-model cases are the directories next to tools/ (committed).
# Rebuilds skip emit by default; REEMIT=1 refreshes them from this import.
MODEL_CASES="${MODEL_CASES:-$EFF}"
REEMIT="${REEMIT:-0}"
if [[ "$REEMIT" == "1" || ! -f "$MODEL_CASES/case-index.json" ]]; then
  "$BUDDY_PYTHON" -B "$EFF/tools/emit_model_cases.py" \
    --ops "$OUT/import/raw-ops.json" --output "$MODEL_CASES"
fi

export EFFNET_CASE_ROOTS="$MODEL_CASES"
export EFFNET_TRITON_BUILD_ROOT="$OUT/triton-build"
cd "$EFF/triton"
# Host first so a kernel bug fails before the long NR encode.
"$TRITON_PYTHON" -P -B build.py --jobs "${JOBS:-4}" --host
"$TRITON_PYTHON" -P -B build.py --jobs "${JOBS:-4}"

"$BUDDY_PYTHON" -B "$EFF/tools/triton_call_replace.py" \
  --checkpoint "$CHECKPOINT" --triton-build "$OUT/triton-build" \
  --output "$OUT/replacement" --input-size 288

python3 - "$OUT/replacement/triton-call-replacement.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("uncovered"):
    raise SystemExit("graph replacement still has uncovered ops; see " + sys.argv[1])
if report.get("status") != "PASS":
    raise SystemExit("graph replacement status is " + str(report.get("status")))
print("replacement PASS,", len(report.get("distinct_symbols", [])), "symbols")
PY

"$BUDDY_PYTHON" -B "$EFF/tools/host_reference.py" \
  --checkpoint "$CHECKPOINT" --output "$OUT/fp32-reference" \
  --input-size 288 --board-input

"$BUDDY_PYTHON" -B "$EFF/tools/lower_model_nr.py" \
  --checkpoint "$CHECKPOINT" --triton-build "$OUT/triton-build" \
  --output "$OUT/nr-forward" --input-size 288

ADAPTERS="$OUT/replacement/effnet_triton_adapters.c" \
  TRITON_BUILD="$OUT/triton-build" \
  OUT="$OUT/model-lib" \
  bash "$EFF/tools/build_model_lib.sh"

"$TRITON_PYTHON" -P -B "$EFF/tools/build_model_image.py" \
  --repo-root "$REPO" \
  --layout "$OUT/nr-forward/weight-layout.json" \
  --graph-ir "$OUT/nr-forward/forward.ll" \
  --archive "$OUT/model-lib/libeffnet_triton.a" \
  --adapters "$OUT/replacement/effnet_triton_adapters.c" \
  --reference "$OUT/fp32-reference/host-reference.json" \
  --output "$OUT/image"
# Default image includes [model]/[op]/[kernel] UART progress; add
# --no-progress-uart above only for quiet throughput builds.

"$TRITON_PYTHON" -P -B "$EFF/tools/prepare_model_run.py" \
  --image "$OUT/image/effnet_model.bin" \
  --elf "$OUT/image/effnet_model.elf" \
  --weights "$OUT/nr-forward/weights.bin" \
  --weight-manifest "$OUT/nr-forward/weight-layout.json" \
  --output "$OUT/prepared"

echo "prepared: $OUT/prepared"
echo "run: $EFF/tools/run_model.sh $OUT/prepared --fpga=$FPGA --capture-seconds=$CAPTURE_SECONDS --startup-timeout=$STARTUP_TIMEOUT"

if [[ "$RUN_FPGA" == "1" ]]; then
  "$EFF/tools/run_model.sh" "$OUT/prepared" \
    --fpga="$FPGA" --capture-seconds="$CAPTURE_SECONDS" \
    --startup-timeout="$STARTUP_TIMEOUT"
fi
