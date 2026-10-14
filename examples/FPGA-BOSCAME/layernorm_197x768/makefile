# The experimental BOSCAME path independently stages three loads and one store
# through DMA; the LayerNorm computation retains its 1024-lane padding semantics.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR :=

ROOT := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
BUILD_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
REPO := $(abspath $(ROOT)/../../..)
REPO_ROOT := $(REPO)
include $(REPO)/runtime/boscame/toolchain.mk
PYTHON ?= python3
BUDDY_BIN ?= $(firstword $(foreach dir,$(REPO)/build/bin,$(if $(wildcard $(dir)/buddy-opt),$(dir))))
BUDDY_OPT ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-opt,buddy-opt)
BUDDY_TRANSLATE ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-translate,buddy-translate)
LLC ?= $(call fpga_llvm_tool,llc)
TOOLS := $(REPO)/examples/FPGA-BOSCAME/tools
NAME := layernorm_197x768
OUT := $(ROOT)/build/$(NAME)
SCRATCH := $(ROOT)/build/.scratch
BOSCAME_PASS ?= --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias"

.PHONY: all ttir boscame llvm asm adapter simcheck clean
all: llvm
$(sort $(OUT) $(SCRATCH)):
	mkdir -p $@

$(OUT)/kernel.ttir $(OUT)/kernel.linalg.mlir &: $(ROOT)/kernel.py $(ROOT)/build.py $(BUILD_MAKEFILE) | $(OUT) $(SCRATCH)
	$(PYTHON) -B $(ROOT)/build.py --ttir-only
ttir: $(OUT)/kernel.ttir

$(OUT)/kernel.boscame.mlir: $(OUT)/kernel.ttir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT)) | $(OUT)
	$(BUDDY_OPT) $< $(BOSCAME_PASS) -o $@
boscame: $(OUT)/kernel.boscame.mlir

$(OUT)/kernel.llvm.mlir: $(OUT)/kernel.boscame.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --lower-bosc-ame -o $@
$(OUT)/kernel.ll: $(OUT)/kernel.llvm.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_TRANSLATE))
	$(BUDDY_TRANSLATE) $< --buddy-to-llvmir -o $@
llvm: $(OUT)/kernel.ll

$(OUT)/kernel.s: $(OUT)/kernel.ll $(BUILD_MAKEFILE)
	$(LLC) $< -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d -mattr=+m,+a,+f,+d,+c,+v -code-model=medium -o $@
$(OUT)/kernel.nr.s: $(OUT)/kernel.s $(TOOLS)/restrict_fpga_assembly.py
	$(PYTHON) $(TOOLS)/restrict_fpga_assembly.py < $< > $@
asm: $(OUT)/kernel.nr.s

$(OUT)/adapter.c: $(ROOT)/build.py $(BUILD_MAKEFILE) | $(OUT)
	$(PYTHON) -B $(ROOT)/build.py --adapter > $@
adapter: $(OUT)/adapter.c

simcheck: boscame
	$(PYTHON) -B $(ROOT)/simcheck.py --module $(OUT)/kernel.boscame.mlir
clean:
	rm -rf $(ROOT)/build
