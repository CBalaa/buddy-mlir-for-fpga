##===- makefile - EfficientNet full-model pipeline ---------*- Makefile -*-===//
##
## Licensed under the Apache License, Version 2.0 (the "License").
## See the LICENSE file at the repository root.
## SPDX-License-Identifier: Apache-2.0
##
## Mirrors ../qwen3-0.6b/model/makefile: import → replace → lower → image,
## plus host-bridge EE check and board-run archival. Source ../env.sh so
## BUDDY_PYTHON matches this-repo LLVM bindings.
##
##===----------------------------------------------------------------------===//

.DEFAULT_GOAL := help
PYTHON ?= python3
BUDDY_PYTHON ?= python3
CHECKPOINT ?= assets/checkpoint
TRITON_BUILD ?= ../triton/build
INPUT_SIZE ?= 288
STAGES ?=
# Qwen-style build/<config>. Prefer CONFIG=...; OUT=... overrides the path.
# Defaults: full → fp32-full, stage0 → stage0 (set per-target below).
REF_DIR ?= build/fp32-reference
REPLACE_DIR ?= build/replacement

.PHONY: help checkpoint import inventory cases replace reference lower \
        lib host-bridge host-run image prepare run full stage0 archive

help:
	@echo "Preparation:  checkpoint"
	@echo "Graph import: import inventory"
	@echo "Kernels:      cases          (refresh committed ../model-cases from raw-ops.json)"
	@echo "Replacement:  replace"
	@echo "Reference:    reference"
	@echo "NR lowering:  lower"
	@echo "Host EE:      host-bridge host-run   (optional pre-board check)"
	@echo "Board image:  lib image prepare"
	@echo "Board run:    run FPGA=5"
	@echo "Archive:      archive RUN=... PREPARED=... REFERENCE=... OUT=..."
	@echo "All-in-one:   full | stage0   (or RUN_FPGA=1 make full)"
	@echo "Config dir:   CONFIG=fp32-full|stage0|ame-v05/fp32-full  (OUT=... overrides)"
	@echo ""
	@echo "Buddy Python bindings must be on PYTHONPATH. Triton kernels are built"
	@echo "from ../triton (or EFFNET_TRITON_BUILD_ROOT), not from this makefile."

checkpoint:
	$(PYTHON) -B tools/fetch_checkpoint.py --output $(CHECKPOINT)

import:
	$(BUDDY_PYTHON) -B tools/import_model.py \
	  --checkpoint $(CHECKPOINT) --output $(or $(IMPORT_DIR),build/import) \
	  --input-size $(INPUT_SIZE) $(if $(STAGES),--stages $(STAGES),)

inventory:
	$(PYTHON) -B tools/operator_inventory.py \
	  --ops $(or $(IMPORT_DIR),build/import)/raw-ops.json \
	  --output $(or $(INVENTORY),build/operator-inventory.json)

cases:
	$(PYTHON) -B tools/emit_model_cases.py \
	  --ops $(or $(IMPORT_DIR),build/import)/raw-ops.json \
	  --output $(or $(MODEL_CASES),../model-cases)

replace:
	$(BUDDY_PYTHON) -B tools/triton_call_replace.py \
	  --checkpoint $(CHECKPOINT) --triton-build $(TRITON_BUILD) \
	  --output $(or $(REPLACE_DIR),build/replacement) \
	  --input-size $(INPUT_SIZE) $(if $(STAGES),--stages $(STAGES),) \
	  $(if $(W8A8),--w8a8,) $(foreach n,$(W8A8_ONLY),--w8a8-only $(n))

reference:
	$(BUDDY_PYTHON) -B tools/host_reference.py \
	  --checkpoint $(CHECKPOINT) --output $(or $(REF_DIR),build/fp32-reference) \
	  --input-size $(INPUT_SIZE) --board-input $(if $(STAGES),--stages $(STAGES),)

lower:
	$(BUDDY_PYTHON) -B tools/lower_model_nr.py \
	  --checkpoint $(CHECKPOINT) --triton-build $(TRITON_BUILD) \
	  --output $(or $(LOWER_DIR),build/nr-forward) \
	  --input-size $(INPUT_SIZE) $(if $(STAGES),--stages $(STAGES),) \
	  $(if $(W8A8),--w8a8,)

lib:
	OUT=$(or $(LIB_DIR),build/model-lib) \
	ADAPTERS=$(or $(ADAPTERS),$(REPLACE_DIR)/effnet_triton_adapters.c) \
	TRITON_BUILD=$(TRITON_BUILD) \
	bash tools/build_model_lib.sh

host-bridge:
	OUT=$(or $(HOST_BRIDGE),build/host-bridge) \
	ADAPTERS=$(or $(ADAPTERS),$(REPLACE_DIR)/effnet_triton_adapters.c) \
	TRITON_BUILD=$(TRITON_BUILD) \
	bash tools/build_host_bridge.sh

host-run: host-bridge
	$(BUDDY_PYTHON) -B tools/run_graph_host.py \
	  --checkpoint $(CHECKPOINT) \
	  --output $(or $(HOST_RUN),build/host-run) \
	  --reference-dir $(REF_DIR) \
	  --input-size $(INPUT_SIZE) \
	  --replace --triton-build $(TRITON_BUILD) \
	  --external-lib $(or $(HOST_BRIDGE),build/host-bridge)/libeffnet_triton_host.so \
	  $(if $(STAGES),--stages $(STAGES),)

image:
	$(PYTHON) -B tools/build_model_image.py \
	  --repo-root $(abspath ../../../../..) \
	  --layout $(or $(LOWER_DIR),build/nr-forward)/weight-layout.json \
	  --graph-ir $(or $(LOWER_DIR),build/nr-forward)/forward.ll \
	  --archive $(or $(LIB_DIR),build/model-lib)/libeffnet_triton.a \
	  --adapters $(or $(ADAPTERS),$(REPLACE_DIR)/effnet_triton_adapters.c) \
	  --reference $(REF_DIR)/host-reference.json \
	  --output $(or $(IMAGE_DIR),build/image) \
	  --input-size $(INPUT_SIZE)

prepare:
	$(PYTHON) -B tools/prepare_model_run.py \
	  --image $(or $(IMAGE_DIR),build/image)/effnet_model.bin \
	  --elf $(or $(IMAGE_DIR),build/image)/effnet_model.elf \
	  --weights $(or $(LOWER_DIR),build/nr-forward)/weights.bin \
	  --weight-manifest $(or $(LOWER_DIR),build/nr-forward)/weight-layout.json \
	  --output $(or $(PREPARED),build/prepared)

run:
	bash tools/run_model.sh $(or $(PREPARED),build/prepared) \
	  --fpga=$(or $(FPGA),5) --capture-seconds=$(or $(CAPTURE_SECONDS),1800) \
	  --startup-timeout=$(or $(STARTUP_TIMEOUT),900)

archive:
	$(PYTHON) -B tools/archive_model_run.py \
	  --run $(RUN) --prepared $(PREPARED) \
	  --reference $(REFERENCE) --output $(OUT)

full:
	CONFIG=$(or $(CONFIG),fp32-full) \
	$(if $(OUT),OUT=$(OUT)) \
	RUN_FPGA=$(or $(RUN_FPGA),0) FPGA=$(or $(FPGA),5) \
	CAPTURE_SECONDS=$(or $(CAPTURE_SECONDS),1800) \
	bash tools/run_full.sh

stage0:
	CONFIG=$(or $(CONFIG),stage0) \
	$(if $(OUT),OUT=$(OUT)) \
	RUN_FPGA=$(or $(RUN_FPGA),0) FPGA=$(or $(FPGA),5) \
	CAPTURE_SECONDS=$(or $(CAPTURE_SECONDS),900) \
	bash tools/run_stage0.sh
