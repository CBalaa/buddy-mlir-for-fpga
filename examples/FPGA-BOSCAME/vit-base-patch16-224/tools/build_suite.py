#!/usr/bin/env python3
"""Configuration helper and compatibility entry point for the ViT Triton suite.

Unlike Qwen, ViT cases do not contain handwritten ``kernel.mlir``.  Triton
``build.py`` is authoritative; this file only keeps the common.mk
``print-config`` protocol and delegates normal invocations.
"""
from pathlib import Path
import hashlib, json, os, runpy, sys

def variables():
    return {key[len("VIT_CFG_"):]: value for key, value in os.environ.items()
            if key.startswith("VIT_CFG_")}

def config():
    return {"directory": str(Path.cwd()), "variables": variables(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--print-config":
        print(json.dumps(config(), sort_keys=True)); return
    if len(sys.argv) > 1 and sys.argv[1] == "--config-stamp":
        path = Path(sys.argv[2]); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config(), sort_keys=True, indent=2) + "\n"); return
    if len(sys.argv) > 1 and sys.argv[1] == "--audit-lowering":
        text = Path(sys.argv[2]).read_text()
        if 'bosc_ame.target = "nr-fpga"' not in text:
            raise SystemExit("NR build rejected: missing target=nr-fpga")
        return
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "triton" / "build.py"),
                   run_name="__main__")

if __name__ == "__main__": main()
