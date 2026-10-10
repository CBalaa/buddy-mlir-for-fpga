# Thin wrapper around build.py, which owns the whole pipeline, the toolchain
# configuration and the build artifacts (IR + final .bin).
#
# ``make``         Triton frontend + AME lowering + NR image + ELF audit
# ``make host``    host oracle built from the very same LLVM IR
# ``make run``     upload the flat image with the shared FPGA runner
.DEFAULT_GOAL := all
ROOT := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
NAME := $(notdir $(ROOT))
BUILD := build
PYTHON ?= python3
.PHONY: all host check nr run clean
all:
	$(PYTHON) -B $(ROOT)/build.py
nr: all
host check:
	$(PYTHON) -B $(ROOT)/build.py --host
run: all
	$(ROOT)/../fpga_run.sh $(ROOT)/$(BUILD)/$(NAME)/$(NAME).bin --fpga=$(or $(FPGA),5) \
	  --capture-seconds=$(or $(CAPTURE_SECONDS),900) --completion-marker='[nr] RA returned:'
clean:
	rm -rf $(BUILD)
