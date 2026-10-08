.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO := $(abspath $(ROOT)/../../../..)
LLVM_BIN ?= $(REPO)/llvm/build-2d26/bin
CLANG ?= $(LLVM_BIN)/clang
HOST_CC ?= cc
OBJCOPY ?= $(LLVM_BIN)/llvm-objcopy
OBJDUMP ?= $(LLVM_BIN)/llvm-objdump
PYTHON ?= python3
LINKER ?= ld.lld
BUILD := $(ROOT)/build
TRACE ?= 0

RISCV_FLAGS := --target=riscv64-unknown-elf -march=rv64imafdc_zicsr_zifencei \
  -mabi=lp64d -mcmodel=medany -mno-relax -O2 -nostdlib -ffreestanding \
  -fno-builtin -fno-pie -ffunction-sections -fdata-sections -I$(ROOT) \
  -DVIT_LINEAR_TRACE=$(TRACE)
HOST_FLAGS := -O2 -std=c11 -DHOST_TEST -DVIT_LINEAR_TRACE=$(TRACE) -I$(ROOT)

.PHONY: all host-check audit clean
all: $(BUILD)/linear_196x768x768_ame_dma_spm.bin host-check audit

$(BUILD):
	mkdir -p $@

$(BUILD)/linear_ame_dma_spm.o: linear_ame_dma_spm.c linear_ame_dma_spm.h | $(BUILD)
	$(CLANG) $(RISCV_FLAGS) -c $< -o $@

$(BUILD)/launch.o: launch.c linear_ame_dma_spm.h | $(BUILD)
	$(CLANG) $(RISCV_FLAGS) -c $< -o $@

$(BUILD)/launcher.o: launcher.S | $(BUILD)
	$(CLANG) $(RISCV_FLAGS) -c $< -o $@

$(BUILD)/linear_196x768x768_ame_dma_spm.elf: $(BUILD)/launcher.o \
  $(BUILD)/launch.o $(BUILD)/linear_ame_dma_spm.o link.ld
	$(LINKER) -m elf64lriscv --gc-sections -T link.ld \
	  -Map=$(BUILD)/linear_196x768x768_ame_dma_spm.map -o $@ \
	  $(BUILD)/launcher.o $(BUILD)/launch.o $(BUILD)/linear_ame_dma_spm.o

$(BUILD)/linear_196x768x768_ame_dma_spm.bin: \
  $(BUILD)/linear_196x768x768_ame_dma_spm.elf
	$(OBJCOPY) -O binary $< $@

$(BUILD)/host-check: launch.c linear_ame_dma_spm.c linear_ame_dma_spm.h | $(BUILD)
	$(HOST_CC) $(HOST_FLAGS) -Wall -Wextra -Werror launch.c linear_ame_dma_spm.c \
	  -o $@

host-check: $(BUILD)/host-check
	$(BUILD)/host-check

audit: $(BUILD)/linear_196x768x768_ame_dma_spm.elf
	$(PYTHON) $(REPO)/examples/FPGA-BOSCAME/tools/check_nr_elf.py $< \
	  --objdump $(OBJDUMP) --output $(BUILD)/elf-audit.json

clean:
	rm -rf $(BUILD)
