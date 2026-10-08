#!/usr/bin/env bash
# Upload the separately linked ViT boot and weight segments through fpga_run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DIR="${1:-$ROOT/build/image}"
if (($#)); then shift; fi
for name in vit_model-load.bin ddr-load.plan; do
  test -f "$IMAGE_DIR/$name" || { echo "missing $IMAGE_DIR/$name" >&2; exit 1; }
done
if [[ -f "$IMAGE_DIR/model-image.json" ]]; then
  python3 - "$IMAGE_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
manifest_path = directory / "model-image.json"
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") != "PASS_STATIC":
    raise SystemExit(f"refusing non-static image: {manifest.get('status')}")
image = manifest.get("image", {})
boot = directory / "vit_model-load.bin"
actual = hashlib.sha256(boot.read_bytes()).hexdigest()
expected = image.get("segments", [{}])[0].get("sha256")
if expected and actual != expected:
    raise SystemExit(f"boot image sha256 mismatch: {actual} != {expected}")
trace = manifest.get("operator_trace", {})
print(f"[vit-run] image={directory}")
print(f"[vit-run] boot_sha256={actual}")
for segment in image.get("segments", [])[1:]:
    path = directory / segment["file"]
    if not path.is_file():
        raise SystemExit(f"missing segment file: {path}")
    actual_segment = hashlib.sha256(path.read_bytes()).hexdigest()
    if segment.get("sha256") and actual_segment != segment["sha256"]:
        raise SystemExit(f"segment sha256 mismatch for {path.name}: {actual_segment} != {segment['sha256']}")
    print(f"[vit-run] {segment['name']}_sha256={actual_segment}")
print(f"[vit-run] trace_ops={bool(trace.get('enabled'))} "
      f"hang_diagnostics={bool(trace.get('hang_diagnostics'))} "
      f"call_count={trace.get('call_count', 0)}")
PY
fi
SEGMENTS=()
while IFS= read -r segment; do
  [[ -n "$segment" ]] && SEGMENTS+=(--segment "$IMAGE_DIR/$segment")
done < <(python3 - "$IMAGE_DIR/ddr-load.plan" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as stream:
    plan = tomllib.load(stream)
for segment in plan.get("segments", [])[1:]:
    print(segment["file"])
PY
)
exec "$ROOT/../fpga_run.sh" "$IMAGE_DIR/vit_model-load.bin" \
  --layout-plan "$IMAGE_DIR/ddr-load.plan" \
  "${SEGMENTS[@]}" \
  --remote-dir "${FPGA_REMOTE_DIR:-Desktop/fpga-tester-chh}" \
  --completion-marker='[nr] RA returned:' "$@"
