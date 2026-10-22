# Complete NH/E6 images for saved buffer inputs and real TTIR frontend inputs.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR :=
ROOT := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
REPO_ROOT := $(abspath $(ROOT)/../../..)
include $(REPO_ROOT)/runtime/boscame/toolchain.mk
PYTHON ?= python3
KERNEL ?= gemm
FRONTEND ?= 0
BASE ?= $(REPO_ROOT)/build/examples/linalg-boscame
BUDDY_OPT ?= $(REPO_ROOT)/build/bin/buddy-opt
BUDDY_TRANSLATE ?= $(REPO_ROOT)/build/bin/buddy-translate
LLC ?= $(call fpga_llvm_tool,llc)
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
TOOLS := $(REPO_ROOT)/examples/FPGA-BOSCAME/tools
RUNTIME := $(REPO_ROOT)/runtime/boscame/runtime
PLATFORM := $(REPO_ROOT)/runtime/boscame/nh_e6
UART := $(REPO_ROOT)/runtime/boscame/uart
NR := $(REPO_ROOT)/runtime/boscame/nr
VARIANT := $(if $(filter 1,$(FRONTEND)),frontend,buffer)
OUT := $(BASE)/board/$(VARIANT)/$(KERNEL)
IR := $(if $(filter 1,$(FRONTEND)),$(BASE)/frontend/$(KERNEL)/kernel,$(BASE)/$(KERNEL))
ELF := $(OUT)/$(KERNEL).elf
BIN := $(OUT)/$(KERNEL).bin
RISCV_FLAGS := --target=riscv64-unknown-elf -march=rv64imafdc_zicsr_zifencei \
  -mabi=lp64d -mcmodel=medany -mno-relax -O2 -nostdlib -ffreestanding \
  -fno-builtin -fno-pie -ffunction-sections -fdata-sections \
  -I$(ROOT) -I$(RUNTIME) -I$(PLATFORM) -I$(UART) -I$(NR)
C_FLAGS := $(RISCV_FLAGS) -fno-vectorize -fno-slp-vectorize -ffp-contract=off \
  -DBOSCAME_BOARD_FRONTEND=$(FRONTEND) '-DBOSC_AME_DMA_ASSERT(x)='
CLANG := $(call fpga_llvm_tool,clang)
OBJCOPY := $(call fpga_llvm_tool,llvm-objcopy)
OBJDUMP := $(call fpga_llvm_tool,llvm-objdump)
MANIFEST := $(PYTHON) $(ROOT)/manifest.py
MANIFEST_ARGS := --out $(OUT) --buddy-opt $(BUDDY_OPT) --buddy-translate $(BUDDY_TRANSLATE) --llc $(LLC) --kernel $(KERNEL) --variant $(VARIANT)

.PHONY: all bin check artifacts clean
all:
	$(MAKE) -f $(ROOT)/makefile KERNEL=gemm bin
	$(MAKE) -f $(ROOT)/makefile KERNEL=layernorm bin
$(OUT):
	mkdir -p $@

ifeq ($(FRONTEND),1)
artifacts:
	$(MANIFEST) capture $(MANIFEST_ARGS)
	$(PYTHON) $(ROOT)/frontend.py --kernel $(KERNEL) --out-dir $(BASE)/frontend --buddy-opt $(BUDDY_OPT) --buddy-translate $(BUDDY_TRANSLATE)
ADAPTER := $(ROOT)/frontend_adapter.c
else
artifacts:
	$(MANIFEST) capture $(MANIFEST_ARGS)
	$(PYTHON) $(ROOT)/build.py --kernel $(KERNEL) --out-dir $(BASE) --buddy-opt $(BUDDY_OPT) --buddy-translate $(BUDDY_TRANSLATE) --check-abi
ADAPTER := $(ROOT)/$(if $(filter gemm,$(KERNEL)),adapter,layernorm_adapter).c
endif

$(OUT)/kernel.s: artifacts | $(OUT)
	$(LLC) $(IR).ll -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d -mattr=+m,+a,+f,+d,+c,+v,+xboscame -code-model=medium -o $@
$(OUT)/kernel.encoded.s: $(OUT)/kernel.s $(TOOLS)/ame_to_word.py
	$(PYTHON) $(TOOLS)/ame_to_word.py --gpr-mode=fixed --preserve-matrix-registers < $< > $@
$(OUT)/kernel.nr.s: $(OUT)/kernel.encoded.s $(TOOLS)/restrict_fpga_assembly.py
	$(PYTHON) $(TOOLS)/restrict_fpga_assembly.py --expand-f32-sub --order-vector-adds < $< > $@
$(OUT)/kernel.o: $(OUT)/kernel.nr.s
	$(CLANG) $(RISCV_FLAGS) -march=rv64imafdcv_zicsr_zifencei -c $< -o $@
$(OUT)/driver.o: $(ROOT)/$(KERNEL)-board.c $(ROOT)/board.h $(ROOT)/abi.h $(ROOT)/frontend_abi.h $(ROOT)/../layernorm_197x768/launch.c | $(OUT)
	$(CLANG) $(C_FLAGS) -c $< -o $@
$(OUT)/board.o: $(ROOT)/board.c $(ROOT)/board.h | $(OUT)
	$(CLANG) $(C_FLAGS) -c $< -o $@
$(OUT)/adapter.o: $(ADAPTER) $(ROOT)/abi.h $(ROOT)/frontend_abi.h $(ROOT)/layernorm_abi.h | $(OUT)
	$(CLANG) $(C_FLAGS) -c $< -o $@
$(OUT)/boot.o: $(PLATFORM)/nh_e6_boot.S | $(OUT)
	$(CLANG) $(RISCV_FLAGS) -c $< -o $@
$(OUT)/platform.o: $(PLATFORM)/nh_e6_platform.c | $(OUT)
	$(CLANG) $(C_FLAGS) -c $< -o $@
$(OUT)/uart.o: $(UART)/uart.c | $(OUT)
	$(CLANG) $(C_FLAGS) -c $< -o $@
$(OUT)/dma.o: $(RUNTIME)/bosc_ame_dma.c | $(OUT)
	$(CLANG) $(C_FLAGS) -c $< -o $@
$(OUT)/math.o: $(NR)/nr_math.c | $(OUT)
	$(CLANG) $(C_FLAGS) -c $< -o $@
OBJECTS := $(addprefix $(OUT)/,boot.o platform.o uart.o dma.o math.o board.o driver.o adapter.o kernel.o)
$(ELF): $(OBJECTS) $(ROOT)/link.ld
	$(LINKER) -m elf64lriscv --gc-sections --wrap=bosc_ame_dma_start --wrap=bosc_ame_dma_wait -T $(ROOT)/link.ld -Map=$(OUT)/$(KERNEL).map -o $@ $(OBJECTS)
$(BIN): $(ELF)
	$(PYTHON) $(TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) --output $(OUT)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
	$(MANIFEST) finish $(MANIFEST_ARGS) --llvm-ir $(IR).ll
bin check: $(BIN)
clean:
	rm -rf $(BASE)/board/$(VARIANT)/$(KERNEL)
