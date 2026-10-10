##===- makefile - EfficientNet FPGA-BOSCAME entry ----------*- Makefile -*-===//
##
## Licensed under the Apache License, Version 2.0 (the "License").
## See the LICENSE file at the repository root.
## SPDX-License-Identifier: Apache-2.0
##
## Enumerate */kernel.mlir. Triton lives in triton/. Real graph shapes are
## the static cases under model-cases/. The full graph lives in model/.
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

.PHONY: suite suite-host run model-help
suite:
	$(PYTHON) -B tools/build_suite.py --group=$(or $(GROUP),all)
suite-host:
	$(PYTHON) -B tools/build_suite.py --group=$(or $(GROUP),all) --host
	./build/suite-$(or $(GROUP),all)-host/suite
run: suite
	../../fpga_run.sh build/suite-$(or $(GROUP),all)/suite.bin --fpga=$(or $(FPGA),5) --capture-seconds=$(or $(CAPTURE_SECONDS),1800) --completion-marker='[nr] RA returned:'
model-help:
	$(MAKE) -C model help
