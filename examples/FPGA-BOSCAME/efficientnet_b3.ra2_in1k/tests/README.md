# EfficientNet model tests

Host-side regressions mirror the Qwen `model/tests/` layout. Vision deployment
does not use tokenizer, attention-position, or fixed-text checks.

## Portable (run from the example directory)

```bash
python -m unittest tests.test_quant_reference
```

`test_quant_reference.py` validates the shared W8A8 `quant_reference.py` contract
used by Triton int8 linear kernels and `quant_model_reference.py`.

## LLM-only (stubbed)

These Qwen tests are not applicable; equivalent files are absent or skipped:

| Qwen test | EfficientNet |
| --- | --- |
| `test_tokenizer_probe.py` | N/A — no tokenizer |
| `test_fixed_text_check.py` | stub tool only |
| `test_fixed_prompt.py` | N/A |
| `test_image_abi.py` | W8A8 image ABI targets LLM KV/descriptors; use FP32 `build_model_image` tests when added |
| `test_graph_contracts.py` | LLM graph matching; vision uses conv/dwconv/linear contracts in `triton_call_replace.py` |

`test_graph_contracts_stub.py` is retained as a reference copy only; do not expect it to pass without Qwen graph fixtures.
