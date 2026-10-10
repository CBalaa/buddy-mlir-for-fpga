# FPGA operator validation (EfficientNet static kernels)

Compact UART and manifest records for static operator cases belong here.
None are checked in with this tree. Weights and build trees stay out of git.

From this example directory, after a suite run:

```bash
python3 tools/record_run.py build/suite-smoke/manifest.json \
  ../build/fpga-runs/run-ACTUAL --output validation/smoke
```

Requirements: image SHA256 match, DDR readback OK, one PASS line per case,
`verify efficientnet operator suite: PASS`, and `[nr] RA returned: PASS`.

Triton-fronted notes are in [triton/validation](../triton/validation/README.md).
