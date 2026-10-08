# ===- env.sh - EfficientNet FPGA-BOSCAME toolchain ----------------*- sh -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repository root.
# SPDX-License-Identifier: Apache-2.0
#
# ===----------------------------------------------------------------------===//
#
# Source from a bash shell after activating the Python environment that
# built this-repo LLVM MLIR bindings and Triton (cpython-312).
#
# Points Triton / Buddy / RISC-V tools at THIS checkout once
# llvm/build-2d26 and build-migrate have been compiled with:
#   -DMLIR_ENABLE_BINDINGS_PYTHON=ON
#   -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON
#   -DPython3_EXECUTABLE pointing at the same interpreter.
#
# Do not put a sibling buddy-mlir tree on PYTHONPATH: buddy_mlir is a
# native extension and must match this LLVM.
#
# Override TRITON_PYTHON / BUDDY_PYTHON / LLVM_SYSPATH before sourcing
# if the lab layout differs. Values like "python -P" are stripped to
# the interpreter path; PYTHONSAFEPATH is set by triton-env.sh.
#
# ===----------------------------------------------------------------------===//

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export BUDDY_SOURCE_DIR="$REPO"

# Interpreter must be a filesystem path, not "python -P".
# The default is the lab conda env used to compile the bindings; if it
# is missing, fall back to python3 on PATH.
_EFF_PY_DEFAULT="/home/caihanyi/miniconda3/envs/boscame/bin/python"
if [[ ! -x "$_EFF_PY_DEFAULT" ]]; then
  _EFF_PY_DEFAULT="$(command -v python3 || true)"
fi
_eff_strip_py() {
  local raw="${1:-}" first
  first="${raw%% *}"
  if [[ -n "$first" && -x "$first" ]]; then
    printf '%s' "$first"
  elif [[ -n "$_EFF_PY_DEFAULT" && -x "$_EFF_PY_DEFAULT" ]]; then
    printf '%s' "$_EFF_PY_DEFAULT"
  else
    printf '%s' "python3"
  fi
}
export TRITON_PYTHON="$(_eff_strip_py "${TRITON_PYTHON:-$_EFF_PY_DEFAULT}")"
# Graph import uses the same ABI as the LLVM/Buddy Python extensions.
export BUDDY_PYTHON="$(_eff_strip_py "${BUDDY_PYTHON:-$TRITON_PYTHON}")"
export LLVM_SYSPATH="${LLVM_SYSPATH:-$REPO/llvm/build-2d26}"
export LLVM_BIN="${LLVM_BIN:-$LLVM_SYSPATH/bin}"
export BUDDY_MLIR_BINARY_DIR="${BUDDY_MLIR_BINARY_DIR:-$REPO/build-migrate/bin}"
export BUDDY_BIN="${BUDDY_BIN:-$BUDDY_MLIR_BINARY_DIR}"

# This-repo packages first. Drop sibling buddy-mlir python_packages.
_BUDDY_PY="$REPO/build-migrate/python_packages"
_eff_pythonpath="$_BUDDY_PY"
if [[ -n "${PYTHONPATH:-}" ]]; then
  IFS=':' read -ra _eff_parts <<< "$PYTHONPATH"
  for _eff_part in "${_eff_parts[@]}"; do
    [[ -z "$_eff_part" || "$_eff_part" == "$_BUDDY_PY" ]] && continue
    if [[ "$_eff_part" == *"/buddy-mlir/"* && "$_eff_part" != *"/buddy-mlir-for-fpga/"* ]]; then
      continue
    fi
    _eff_pythonpath="${_eff_pythonpath}:${_eff_part}"
  done
fi
export PYTHONPATH="$_eff_pythonpath"
unset _eff_part _eff_parts _eff_pythonpath _BUDDY_PY
unset -f _eff_strip_py
unset _EFF_PY_DEFAULT
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# shellcheck disable=SC1091
source "$REPO/examples/FPGA-BOSCAME/common/triton/triton-env.sh"
