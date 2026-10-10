##===- makefile - EfficientNet FPGA-BOSCAME entry ----------*- Makefile -*-===//
##
## Licensed under the Apache License, Version 2.0 (the "License").
## See the LICENSE file at the repository root.
## SPDX-License-Identifier: Apache-2.0
##
## Enumerate */kernel.mlir. Graph cases sit next to this file. Triton lives
## in triton/. Full-model scripts live in tools/; weights stay under model/.
##
##===----------------------------------------------------------------------===//

.DEFAULT_GOAL := all
PYTHON ?= python3
CASES := $(sort $(patsubst %/kernel.mlir,%,$(wildcard */kernel.mlir)))
.PHONY: all check generate cases list clean $(CASES)
all: $(CASES)
$(CASES):
	$(MAKE) -C $@ all
check:
	@set -e; for c in $(CASES); do $(MAKE) -C $$c check; done
generate cases:
	$(PYTHON) -B tools/generate_cases.py
	$(PYTHON) -B tools/generate_matmul.py
	$(PYTHON) -B tools/generate_quant.py
list:
	@printf '%s\n' $(CASES)
clean:
	@set -e; for c in $(CASES); do $(MAKE) -C $$c clean; done

.PHONY: suite suite-host run model-help model-full model-stage0
suite:
	$(PYTHON) -B tools/build_suite.py --group=$(or $(GROUP),smoke)
suite-host:
	$(PYTHON) -B tools/build_suite.py --group=$(or $(GROUP),smoke) --host
	./build/suite-$(or $(GROUP),smoke)-host/suite
run: suite
	../fpga_run.sh build/suite-$(or $(GROUP),smoke)/suite.bin --fpga=$(or $(FPGA),5) --capture-seconds=$(or $(CAPTURE_SECONDS),1800) --completion-marker='[nr] RA returned:'
model-help:
	@echo "Full model:  bash tools/run_full.sh"
	@echo "One stage:   bash tools/run_stage0.sh"
	@echo "Board only:  bash tools/run_model.sh model/build/fp32-full/prepared --fpga=5"
	@echo "See model/README.md"
model-full:
	bash tools/run_full.sh
model-stage0:
	bash tools/run_stage0.sh
