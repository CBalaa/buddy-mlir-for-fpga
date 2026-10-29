# makefile -- the layernorm_197x768 experimental BOSCAME pipeline.
#
#     kernel.frontend.mlir (external, standard MLIR Contract input)
#       -> kernel.bufferized.mlir --bosc-ame-plan
#       -> kernel.planned.mlir --bosc-ame-emit             [make boscame]
#       -> kernel.boscame.mlir
#     --lower-bosc-ame + --buddy-to-llvmir     [make llvm]
#       -> kernel.llvm.mlir + kernel.ll
#     llc -> RVV assembly; NR filter           [make asm]
#       -> kernel.nr.s
#     clang -c + ld.lld + audit + objcopy      [make bin]
#       -> build/layernorm_197x768/layernorm_197x768.bin
#
# 'make' runs all of it.  Every stage is a file rule, so re-running only
# rebuilds what changed.  'make boscame', 'make llvm' and 'make asm' stop at
# the named stage.
#
# The image runs on the NH/E6 platform (runtime/boscame/nh_e6), which owns the
# boot, the dual-core release/wait, the E6 address-space setup (seven TOR PMP
# regions, satp=0) and the CSR 0xbc0 cache policy.  link.ld places the DDR
# windows inside the PMP region the platform exports -- otherwise a DMA that
# leaves that region hangs with no diagnostic.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR :=

ROOT := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
BUILD_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
REPO := $(abspath $(ROOT)/../../..)

# Shared platform toolchain: prefers the in-tree LLVM build, then PATH.
REPO_ROOT := $(REPO)
include $(REPO)/runtime/boscame/toolchain.mk

BUDDY_BIN ?= $(BUDDY_MLIR_BINARY_DIR)
BUDDY_OPT ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-opt,buddy-opt)
BUDDY_TRANSLATE ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-translate,buddy-translate)
# Python simulator helpers use the same explicitly selected backend tools.
export BUDDY_OPT BUDDY_TRANSLATE
LLC ?= $(call fpga_llvm_tool,llc)
OBJCOPY ?= $(call fpga_llvm_tool,llvm-objcopy)
OBJDUMP ?= $(call fpga_llvm_tool,llvm-objdump)
# ld.lld ships with LLVM, but a BUDDY build does not require the lld project;
# fall back to the linker on PATH when the in-tree build has none.
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
PYTHON ?= python3

TOOLS := $(REPO)/examples/FPGA-BOSCAME/tools
include $(TOOLS)/ordered_reduction_policy.mk
NAME := layernorm_197x768
OUT := $(ROOT)/build/$(NAME)
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

.PHONY: all bufferized boscame llvm adapter asm bin check simcheck clean
all: bin

$(sort $(OUT) $(OBJ)):
	mkdir -p $@

# --- 1. Import the external Contract -----------------------------------
# Override FRONTEND_IR to consume a newly exported handoff from triton-riscv.
FRONTEND_IR ?= $(ROOT)/kernel.frontend.mlir
.PHONY: input-force
input-force:
$(OUT)/kernel.frontend.mlir: $(FRONTEND_IR) $(BUILD_MAKEFILE) input-force | $(OUT)
	@cmp -s $< $@ || cp $< $@

# --- 2. Verify, plan and emit ------------------------------------------
$(OUT)/kernel.bufferized.mlir: $(OUT)/kernel.frontend.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --kernel-contract-verify -o $@
bufferized: $(OUT)/kernel.bufferized.mlir

$(OUT)/kernel.planned.mlir: $(OUT)/kernel.bufferized.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --bosc-ame-plan --bosc-ame-plan-verify -o $@

$(OUT)/kernel.boscame.mlir: $(OUT)/kernel.planned.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --bosc-ame-emit -o $@
boscame: $(OUT)/kernel.boscame.mlir

# --- 3. BOSCAME -> LLVM dialect -> LLVM IR -----------------------------
$(OUT)/kernel.llvm.mlir: $(OUT)/kernel.boscame.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT)) $(OUT)/.ordered-reduction-policy $(FPGA_REDUCTION_POLICY_MAKEFILE)
	$(BUDDY_OPT) $< --lower-bosc-ame="$(FPGA_ORDERED_REDUCTION_LOWER_OPTION)" -o $@
$(OUT)/kernel.ll: $(OUT)/kernel.llvm.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_TRANSLATE))
	$(BUDDY_TRANSLATE) --buddy-to-llvmir $< -o $@
llvm: $(OUT)/kernel.ll

# --- 4. RISC-V assembly -------------------------------------------------
$(OUT)/kernel.s: $(OUT)/kernel.ll $(BUILD_MAKEFILE)
	$(LLC) $< -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d -mattr=+m,+a,+f,+d,+c,+v -code-model=medium -o $@

# --- 5. NR post-processing: restricted NR instruction stream ------------
$(OUT)/kernel.nr.s: $(OUT)/kernel.s $(TOOLS)/restrict_fpga_assembly.py $(OUT)/.ordered-reduction-policy $(FPGA_REDUCTION_POLICY_MAKEFILE)
	$(PYTHON) $(TOOLS)/restrict_fpga_assembly.py $(ORDERED_REDUCTION_FILTER_FLAGS) --expand-f32-sub < $< > $@
asm: $(OUT)/kernel.nr.s

$(OUT)/kernel.o: $(OUT)/kernel.nr.s $(BUILD_MAKEFILE)
	$(RISCV_CC) $(RISCV_FLAGS) -march=rv64imafdcv_zicsr_zifencei -c $< -o $@

# --- 6. application driver, ABI bridge, platform and DMA runtime ---------
# The MLIR wrapper takes four typed unranked f32 descriptor pointers plus six
# grid/program scalars.  The adapter validates the DMA ABI, checks that the
# four root buffers are aligned, non-alias, large enough and contiguous, then
# launches all 197 rows in reverse order.
$(OUT)/adapter.c: $(ROOT)/build.py $(BUILD_MAKEFILE) | $(OUT)
	$(PYTHON) -B $(ROOT)/build.py --adapter > $@
adapter: $(OUT)/adapter.c

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

# --- 7. NH/E6 image, its audit and the flat .bin ------------------------
$(ELF): $(IMAGE_OBJECTS) $(ROOT)/link.ld $(BUILD_MAKEFILE)
	$(LINKER) -m elf64lriscv --gc-sections -T $(ROOT)/link.ld -Map=$(OUT)/$(NAME).map -o $@ $(IMAGE_OBJECTS)
$(BIN): $(ELF) $(TOOLS)/check_nr_elf.py $(OUT)/.ordered-reduction-policy $(FPGA_REDUCTION_POLICY_MAKEFILE)
	$(PYTHON) $(TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) $(ORDERED_REDUCTION_FLAGS) --output $(OUT)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
	@echo "built $@"
bin: $(BIN)

# --- 8. functional check that needs no board ----------------------------
# Runs the pass output in tools/boscame_buffer_sim.py (functional DMA/SPM
# model) for all 197 programs and compares every f32 output with an independent
# sequential-f32 oracle at the board check's 2e-5 absolute/relative tolerance.
simcheck: boscame
	$(PYTHON) -B $(ROOT)/simcheck.py --module $(OUT)/kernel.boscame.mlir

# --- 9. image checks ----------------------------------------------------
check: bin
	$(PYTHON) -B $(ROOT)/build.py --check-abi $(OUT)/kernel.ll
# Check exact user-buffer ranges for the default DDR/register route or the
# frozen four-DMA compatibility route, including every launched row.
	$(PYTHON) -B $(TOOLS)/check_layernorm_boundary.py $(OUT)/kernel.boscame.mlir
	@echo "ok: $(NAME) passed the typed ABI, external boundary and NR ELF audits"

clean:
	rm -rf $(ROOT)/build
