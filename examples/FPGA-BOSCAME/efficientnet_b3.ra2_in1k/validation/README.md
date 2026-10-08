# FPGA operator validation (EfficientNet static kernels)

Same role as `qwen3-0.6b/validation/`: auditable UART + manifest records for
**static** `kernel.mlir` operator cases under this example directory.

Use the shared runner and record tool:

```bash
python3 tools/record_run.py build/suite-smoke/manifest.json \
  ../build/fpga-runs/run-ACTUAL --output validation/smoke
```

Requirements: image SHA256 match, DDR readback OK, one PASS line per case,
`verify efficientnet operator suite: PASS`, and `[nr] RA returned: PASS`.

Triton-fronted validation lives in [triton/validation](../triton/validation/README.md).
Full-model classifier runs are archived under `model/validation/board/`.
