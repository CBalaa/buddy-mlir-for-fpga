# Path parity: qwen3-0.6b → efficientnet_b3.ra2_in1k

Deployment pipeline parity (W8A8 linear, weight image, graph tools, validation
scaffolding). LLM runtime code is stubbed or documented N/A.

Legend: **ported** = adapted copy; **existing** = already in DST before this port;
**stub** = exits with N/A; **N/A** = no vision equivalent.

## Example root

| Qwen path | EffNet path | Status |
| --- | --- | --- |
| `tools/build_suite.py` | `tools/build_suite.py` | **existing** (`efficientnet operator suite`) |
| `tools/record_run.py` | `tools/record_run.py` | **ported** (suite regex) |
| `tools/vectorize_nr.py` | `tools/vectorize_nr.py` | **existing** |
| `tools/host_main.c` | `tools/host_main.c` | **existing** |
| `tools/generate_cases.py` | `tools/generate_cases.py` | **existing** (do not delete) |
| `validation/README.md` | `validation/README.md` | **ported** |
| `PATH_PARITY.md` | `PATH_PARITY.md` | this file |

## `tools/` (operator NR)

| Qwen | EffNet | Status |
| --- | --- | --- |
| `generate_matmul.py` | (same name) | **ported** (EffNet classifier/SE + tile-tail) |
| `generate_quant.py` | (same name) | **ported** (paired quant/dequant shapes) |
| `generate_elementwise.py` | (same name) | **stub** → use `generate_cases.py` |

## `model/tools/`

| Qwen | EffNet | Status |
| --- | --- | --- |
| `import_model.py` | same | **existing** |
| `triton_call_replace.py` | same | **existing** + **`--w8a8`** / `rewrite_linear_to_w8a8` |
| `lower_model_nr.py` | same | **existing** |
| `build_model_image.py` | same | **existing** (FP32 board) |
| `prepare_model_run.py` | same | **existing** |
| `run_model.sh` / `run_full.sh` | same | **existing** |
| `archive_model_run.py` | same | **existing** (classifier UART) |
| `emit_model_cases.py` | same | **existing** (replaces `model_kernel_cases.py`) |
| `quant_reference.py` | same | **ported** (numpy, as-is) |
| `quant_model_reference.py` | same | **ported** (classifier linear check) |
| `weight_layout.py` | same | **ported** |
| `build_weight_image.py` | same | **ported** (no KV/tokenizer/RoPE) |
| `build_nr_weight_segment.py` | same | **ported** |
| `build_nr_w8a8_segment.py` | same | **ported** |
| `build_nr_w8a8_image.py` | same | **ported** (no chat CLI; `effnet_*` runtime symbols; FP32 uses `build_model_image`) |
| `graph_contract.py` | same | **ported** |
| `graph_report.py` | same | **ported** |
| `inspect_frontend.py` | same | **ported** |
| `final_operator_mapping.py` | same | **ported** |
| `kernel_build_overlay.py` | same | **ported** |
| `kernel_profile.py` | same | **ported** (`effnet_profile_*`) |
| `check_kernel_profile.py` | same | **ported** |
| `memory_plan.py` | same | **ported** |
| `report_model_memory.py` | same | **ported** |
| `intermediate_probe.py` | same | **ported** (`effnet_intermediate_*`) |
| `check_intermediates.py` | same | **ported** |
| `score_graph_host.py` | same | **ported** |
| `checkpoint_inventory.py` | same | **ported** |
| `checkpoint_verify.py` | same | **ported** |
| `reference_fp32.py` / `.c` | same | **ported** (`effnet_ref_nr_*`) |
| `run_full.sh` / `run_stage0.sh` | same | **existing** (`CONFIG` → `build/<config>`, default `fp32-full` / `stage0`) |
| `fetch_assets.py` | same | **ported** (wraps `fetch_checkpoint.py`) |
| `fetch_checkpoint.py` | same | **existing** |
| `archive_attention_position_run.py` | same | **stub** |
| `attention_position_cases.py` | same | **stub** |
| `audit_attention_native_ir.py` | same | **stub** |
| `pack_tokenizer.py` | same | **stub** |
| `gen_unicode_tables.py` | same | **stub** |
| `build_tokenizer_probe.py` | same | **stub** |
| `check_tokenizer_*` | same | **stub** |
| `check_fixed_text.py` / `check_text.py` | same | **stub** |
| `model_kernel_cases.py` | same | **stub** → `emit_model_cases.py` |

## `model/` subtrees

| Qwen | EffNet | Status |
| --- | --- | --- |
| `model/text/` | `model/text/README.md` | **N/A** doc |
| `model/optimization/` | `model/optimization/README.md` | **ported** (vision shapes) |
| `model/tests/` | `model/tests/` | **ported** (+ LLM tests stubbed) |
| `model/validation/` | `model/validation/` | **ported** (templates) |
| `triton/validation/` | `triton/validation/README.md` | **ported** |

## Docs

| Qwen | EffNet | Status |
| --- | --- | --- |
| `MODEL.md` | `MODEL.md` | **updated** (W8A8 path) |
| `model/README.md` | `model/README.md` | **updated** |
