# Triton operator validation (EfficientNet)

Host and FPGA evidence for Triton-fronted cases. Nothing under this directory
is an archived board run yet.

From `triton/`:

```bash
source ../../common/triton/triton-env.sh
make check GROUP=smoke JOBS=4
make run GROUP=smoke FPGA=5
python3 ../tools/record_run.py build/suite-smoke/manifest.json \
  ../../build/fpga-runs/run-ACTUAL --output validation/smoke
```

The UART line must include `verify efficientnet operator suite: PASS`.
