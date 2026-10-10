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

整模归档说明用同目录的 `operator-inventory-note.json` 和 `w8a8-rewrite.json`。成功的板上记录写到 `model/validation/board/`，该目录不入库。

```bash
python -B tools/archive_model_run.py \
  --run ../build/fpga-runs/run-XXXX \
  --prepared model/build/fp32-full/prepared \
  --reference model/build/fp32-full/fp32-reference/host-reference.json \
  --output model/validation/board/fp32-full-YYYYMMDD
```
