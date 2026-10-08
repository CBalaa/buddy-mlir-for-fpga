#!/usr/bin/env bash
# Build the host shared library used to execute the replaced graph.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_BUILD="${TRITON_BUILD:-$ROOT/build/kernels}"
ADAPTERS="${ADAPTERS:-$ROOT/build/replacement/vit_triton_adapters.c}"
OUT="${OUT:-$ROOT/build/host-bridge}"
# The exported LLVM IR uses opaque pointers; the system clang may be older
# than the LLVM build used by Buddy and Triton.
CC="${HOST_CC:-$ROOT/../../../llvm/build-2d26/bin/clang}"
if [[ ! -f "$ADAPTERS" ]]; then echo "missing $ADAPTERS" >&2; exit 1; fi
mapfile -t ROOT_CASES < <(grep -o '_mlir_ciface_kernel_[A-Za-z0-9_]*' "$ADAPTERS" | sed 's/_mlir_ciface_kernel_//' | sort -u)
((${#ROOT_CASES[@]})) || { echo "adapter has no Triton calls" >&2; exit 1; }
# Keep helper kernels referenced by a composed adapter in the bridge as well
# as the graph's direct calls.  Patch embedding's adapter references the
# 196x768x768 linear ciface after materializing im2col.
declare -A seen_cases=()
CASES=()
add_case() {
  local name="$1"
  [[ ${seen_cases[$name]+yes} ]] && return
  seen_cases[$name]=1
  CASES+=("$name")
}
for name in "${ROOT_CASES[@]}"; do add_case "$name"; done
for ((index = 0; index < ${#CASES[@]}; ++index)); do
  name="${CASES[$index]}"
  adapter="$TRITON_BUILD/$name/adapter.c"
  [[ -f "$adapter" ]] || continue
  while IFS= read -r dependency; do
    [[ -n "$dependency" ]] && add_case "$dependency"
  done < <(grep -o '_mlir_ciface_kernel_[A-Za-z0-9_]*' "$adapter" |
           sed 's/_mlir_ciface_kernel_//' | sort -u)
done
rm -rf "$OUT"; mkdir -p "$OUT"
flags=(-O2 -fPIC -ffp-contract=off -DHOST_TEST -I"$ROOT")
objects=()
for name in "${CASES[@]}"; do
  test -f "$TRITON_BUILD/$name/host/kernel.ll" || { echo "missing host kernel $name" >&2; exit 1; }
  "$CC" "${flags[@]}" -c "$TRITON_BUILD/$name/host/kernel.ll" -o "$OUT/$name.kernel.o"
  "$CC" "${flags[@]}" -c "$TRITON_BUILD/$name/adapter.c" -o "$OUT/$name.ciface.o"
  objects+=("$OUT/$name.kernel.o" "$OUT/$name.ciface.o")
done
"$CC" "${flags[@]}" -c "$ADAPTERS" -o "$OUT/vit_triton_adapters.o"
objects+=("$OUT/vit_triton_adapters.o")
"$CC" -shared -o "$OUT/libvit_triton_host.so" "${objects[@]}" -lm
printf '%s kernels -> %s\n' "${#CASES[@]}" "$OUT/libvit_triton_host.so"
