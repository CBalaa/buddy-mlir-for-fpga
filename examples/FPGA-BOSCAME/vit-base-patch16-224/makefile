.DEFAULT_GOAL := help
PYTHON ?= python3
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ASSETS ?= $(ROOT)/assets
CHECKPOINT ?= $(ASSETS)/model.safetensors
IMPORT ?= $(ROOT)/build/import
KERNELS ?= $(ROOT)/build/kernels
REPLACEMENT ?= $(ROOT)/build/replacement
LOWERED ?= $(ROOT)/build/lowered
IMAGE ?= $(ROOT)/build/image
ARCHIVE ?= $(ROOT)/build/nr-archive/libvit_triton_nr.a
INPUT_IMAGE ?=
INPUT_SEED ?=
EXPECTED_TOP1 ?=
.PHONY: help assets import kernels replace lower host-lib nr-archive image clean
help:
	@echo "assets import kernels replace lower host-lib nr-archive image"
assets:
	@test -f $(ASSETS)/config.json -a -f $(ASSETS)/preprocessor_config.json -a -f $(CHECKPOINT) || \
	  $(PYTHON) -B tools/fetch_assets.py --output $(ASSETS)
import:
	@test -f $(ASSETS)/config.json -a -f $(CHECKPOINT)
	$(PYTHON) -B tools/import_model.py --assets $(ASSETS) --checkpoint $(CHECKPOINT) --output $(IMPORT) --save-mlir
kernels:
	$(PYTHON) -B triton/build.py --output $(KERNELS) --suite $(if $(HOST),--host,)
replace:
	@test -f $(IMPORT)/params.json -a -f $(IMPORT)/raw-ops.json
	$(PYTHON) -B tools/replace_triton.py --import-dir $(IMPORT) --triton-build $(KERNELS) --output $(REPLACEMENT) --require-complete
host-lib:
	@test -f $(REPLACEMENT)/vit_triton_adapters.c
	TRITON_BUILD=$(KERNELS) ADAPTERS=$(REPLACEMENT)/vit_triton_adapters.c tools/build_host_bridge.sh
nr-archive:
	@test -f $(REPLACEMENT)/vit_triton_adapters.c
	RISCV_CC='$(ROOT)/../../../llvm/build-2d26/bin/clang --target=riscv64-unknown-elf' \
	  LLVM_AR=$(ROOT)/../../../llvm/build-2d26/bin/llvm-ar \
	  TRITON_BUILD=$(KERNELS) ADAPTERS=$(REPLACEMENT)/vit_triton_adapters.c \
	  tools/build_nr_archive.sh
lower:
	@test -f $(REPLACEMENT)/subgraph0.triton.mlir -a -f $(REPLACEMENT)/forward.triton.mlir
	$(PYTHON) -B tools/lower_model.py --replacement $(REPLACEMENT) --output $(LOWERED) \
	  --buddy-opt $(ROOT)/../../../build-python/bin/buddy-opt \
	  --buddy-translate $(ROOT)/../../../build-python/bin/buddy-translate
image:
	@test -f $(IMPORT)/params.json -a -f $(CHECKPOINT) -a -f $(LOWERED)/forward.ll -a -f $(LOWERED)/main.ll
	@test -f $(REPLACEMENT)/subgraph0.triton.mlir -a -f $(REPLACEMENT)/vit_triton_adapters.c -a -f $(ARCHIVE)
	@$(PYTHON) -c 'import hashlib,json; from pathlib import Path; p=Path("$(ROOT)/build/host-graph-check.json"); g=Path("$(REPLACEMENT)/subgraph0.triton.mlir"); d=json.loads(p.read_text()); assert d["status"] == "PASS" and d["parameter_order_verified"] and d["mismatched_elements"] == 0 and d["subgraph_sha256"] == hashlib.sha256(g.read_bytes()).hexdigest(), "current host graph check must pass before image build"'
	$(PYTHON) -B tools/build_model_image.py \
	  --repo-root $(ROOT)/../../.. \
	  --params $(IMPORT)/params.json \
	  --checkpoint $(CHECKPOINT) \
	  --graph-ir $(LOWERED)/forward.ll \
	  --main-graph-ir $(LOWERED)/main.ll \
	  --subgraph-mlir $(REPLACEMENT)/subgraph0.triton.mlir \
	  --archive $(ARCHIVE) \
	  --adapters $(REPLACEMENT)/vit_triton_adapters.c \
	  --output $(IMAGE) $(if $(INPUT_IMAGE),--input-image $(INPUT_IMAGE),) $(if $(INPUT_SEED),--input-seed $(INPUT_SEED),) $(if $(EXPECTED_TOP1),--expected-top1 $(EXPECTED_TOP1),)
clean:
	rm -rf build
