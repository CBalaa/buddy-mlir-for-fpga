.DEFAULT_GOAL := all
TRITON_PYTHON ?= python3
GROUP ?= all
JOBS ?= 4
FPGA ?= 5
CAPTURE_SECONDS ?= 1800
SELECT := $(if $(CASE),--case=$(CASE),--group=$(GROUP))
.PHONY: all export check list host run
all:
	$(TRITON_PYTHON) build.py $(SELECT) $(if $(CASE),,--suite) --jobs=$(JOBS)
export:
	$(TRITON_PYTHON) export.py $(if $(CASE),--case=$(CASE),)
check host:
	$(TRITON_PYTHON) build.py $(SELECT) $(if $(CASE),,--suite) --host --jobs=$(JOBS)
list:
	$(TRITON_PYTHON) build.py --list
run: all
	../../fpga_run.sh build/$(if $(CASE),$(CASE)/nr/$(CASE).bin,suite-$(GROUP)/suite.bin) --fpga=$(FPGA) --capture-seconds=$(CAPTURE_SECONDS) --completion-marker='[nr] RA returned:'
