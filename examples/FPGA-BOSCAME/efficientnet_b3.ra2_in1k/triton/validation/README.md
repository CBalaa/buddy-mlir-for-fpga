# Triton operator validation (EfficientNet)

Records host and FPGA evidence for Triton-fronted operator cases from
`tools/generate_cases.py` (smoke) and the committed `model-cases/` tree
(full graph shapes; refresh via `emit_model_cases.py`).

Layout matches `qwen3-0.6b/triton/validation/`:

- `host/` — host execution and manifest hashes
- `all/` — full suite FPGA run via `tools/record_run.py`
- `codegen.json`, `elf-audit.json` — ISA audit summaries

Record a successful suite run from `efficientnet_b3.ra2_in1k/triton/`:

```bash
source ../../common/triton/triton-env.sh
make check GROUP=smoke JOBS=4
make run GROUP=smoke FPGA=5
python3 ../tools/record_run.py build/suite-smoke/manifest.json \
  ../../build/fpga-runs/run-ACTUAL --output validation/smoke
```

The UART line must include `verify efficientnet operator suite: PASS` (see
`tools/build_suite.py` / `tools/record_run.py`).
