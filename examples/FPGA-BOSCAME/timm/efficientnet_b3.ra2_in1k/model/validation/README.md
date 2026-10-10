# Model validation evidence (EfficientNet)

Same role as `qwen3-0.6b/model/validation/`: compact JSON and UART summaries for
board runs. Large binaries and build trees stay gitignored under `model/build/`.

## Placeholder schemas

- `operator-inventory-note.json` — reminder that full operator mapping runs via
  `tools/final_operator_mapping.py` after import.
- `w8a8-rewrite.json` — empty template for W8A8 rewrite reports once
  `triton_call_replace.py` enables linear W8A8 on the full graph.

Archive successful FP32 runs with:

```bash
python -B model/tools/archive_model_run.py \
  --run ../../build/fpga-runs/run-XXXX \
  --prepared model/build/fp32-full/prepared \
  --reference model/build/fp32-full/fp32-reference/host-reference.json \
  --output model/validation/board/fp32-full-YYYYMMDD
```

W8A8 board evidence uses the same archivers once a W8A8 prepared image exists.
