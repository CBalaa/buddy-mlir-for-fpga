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
OBJCOPY ?= $(call fpga_llvm_tool,llvm-objcopy)
OBJDUMP ?= $(call fpga_llvm_tool,llvm-objdump)
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
TOOLS := $(REPO)/examples/FPGA-BOSCAME/tools
NAME := layernorm_197x768
OUT := $(ROOT)/build/$(NAME)
SCRATCH := $(ROOT)/build/.scratch
BOSCAME_PASS ?= --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias"
RUNTIME := $(REPO)/runtime/boscame/runtime
UART := $(REPO)/runtime/boscame/uart
PLATFORM := $(REPO)/runtime/boscame/nh_e6
NR := $(REPO)/runtime/boscame/nr
OBJ := $(OUT)/obj
ELF := $(OUT)/$(NAME).elf
BIN := $(OUT)/$(NAME).bin
RISCV_FLAGS := -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -fno-pie \
  -ffunction-sections -fdata-sections -I$(ROOT) -I$(RUNTIME) -I$(UART) -I$(PLATFORM) -I$(NR)
C_FLAGS := $(RISCV_FLAGS) -fno-vectorize -fno-slp-vectorize -ffp-contract=off \
  '-DBOSC_AME_DMA_ASSERT(x)='

.PHONY: all ttir boscame llvm asm adapter bin check simcheck clean
all: llvm
$(sort $(OUT) $(SCRATCH) $(OBJ)):
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

$(OUT)/kernel.o: $(OUT)/kernel.nr.s $(BUILD_MAKEFILE)
	$(RISCV_CC) $(RISCV_FLAGS) -march=rv64imafdcv_zicsr_zifencei -c $< -o $@
$(OBJ)/launch.o: $(ROOT)/launch.c $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@
$(OBJ)/adapter.o: $(OUT)/adapter.c $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -DLAYERNORM_TRACE=1 -DLAYERNORM_REVERSE=1 -c $< -o $@
$(OBJ)/nh_e6_boot.o: $(PLATFORM)/nh_e6_boot.S $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(RISCV_FLAGS) -c $< -o $@
$(OBJ)/nh_e6_platform.o: $(PLATFORM)/nh_e6_platform.c $(PLATFORM)/nh_e6_platform.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@
$(OBJ)/uart.o: $(UART)/uart.c $(UART)/uart.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@
$(OBJ)/bosc_ame_dma.o: $(RUNTIME)/bosc_ame_dma.c $(RUNTIME)/bosc_ame_dma.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@
$(OBJ)/nr_math.o: $(NR)/nr_math.c $(NR)/nr_runtime.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@
IMAGE_OBJECTS := $(OBJ)/nh_e6_boot.o $(OBJ)/nh_e6_platform.o $(OBJ)/uart.o \
  $(OBJ)/bosc_ame_dma.o $(OBJ)/nr_math.o $(OBJ)/launch.o $(OBJ)/adapter.o $(OUT)/kernel.o
$(ELF): $(IMAGE_OBJECTS) $(ROOT)/link.ld $(BUILD_MAKEFILE)
	$(LINKER) -m elf64lriscv --gc-sections -T $(ROOT)/link.ld -Map=$(OUT)/$(NAME).map -o $@ $(IMAGE_OBJECTS)
$(BIN): $(ELF) $(TOOLS)/check_nr_elf.py
	$(PYTHON) $(TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) --output $(OUT)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
bin: $(BIN)
check: bin
	@echo "ok: LayerNorm NH/E6 image passed the retained NR ELF audit"

simcheck: boscame
	$(PYTHON) -B $(ROOT)/simcheck.py --module $(OUT)/kernel.boscame.mlir
clean:
	rm -rf $(ROOT)/build
