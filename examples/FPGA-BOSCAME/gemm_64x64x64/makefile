# makefile -- build the BOSCAME-dialect 64x64x64 SPM/DMA GEMM.
#
# The kernel is written in the BOSCAME dialect and lowered by the unified
# `-lower-bosc-ame` pass; there is no hand-written C kernel in this directory.
#
#     make all      # lower + assemble + structural check
#     make board    # link the E6/NH image
#     make audit    # PMP/window guard + check_nr_elf.py on the image
#     make run FPGA=4
#
# The pipeline is the whole point of the example:
#
#     gemm_64x64x64_ame_dma_spm.mlir
#       --buddy-opt --lower-bosc-ame            # checks + tile/SPM/DMA + LLVM
#       -> LLVM-dialect MLIR
#       --buddy-translate --buddy-to-llvmir     -> LLVM IR
#       --llc -mattr=...,+xboscame              -> RISC-V asm with AME
#       --ame_to_word.py --gpr-mode=fixed       -> fixed-register AME words
#       --restrict_fpga_assembly.py             -> validated + fenced NR asm
#       --clang -c && ld.lld                    -> E6/NH ELF/BIN
#
# The BOSCAME extension is not passed on
# the llc command line: `-lower-bosc-ame` writes it onto the lowered functions
# as a target feature, so "this module uses the FPGA AME datapath" has a single
# source of truth.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR :=

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
REPO := $(abspath $(ROOT)/../../..)
TOOLS := $(REPO)/examples/FPGA-BOSCAME/tools

# Shared platform toolchain: prefers the in-tree LLVM build, then PATH.
REPO_ROOT := $(REPO)
include $(REPO)/runtime/boscame/toolchain.mk

BUDDY_BIN ?= $(firstword $(foreach dir,$(REPO)/build/bin,$(if $(wildcard $(dir)/buddy-opt),$(dir))))
BUDDY_OPT ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-opt,buddy-opt)
BUDDY_TRANSLATE ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-translate,buddy-translate)
LLC ?= $(call fpga_llvm_tool,llc)
OBJCOPY ?= $(call fpga_llvm_tool,llvm-objcopy)
OBJDUMP ?= $(call fpga_llvm_tool,llvm-objdump)
NM ?= $(call fpga_llvm_tool,llvm-nm)
# ld.lld ships with LLVM, but a BUDDY build does not require the lld project;
# fall back to the linker on PATH when the in-tree build has none.
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
PYTHON ?= python3

BUILD := $(ROOT)/build
NAME := gemm_64x64x64
KERNEL := $(ROOT)/gemm_64x64x64_ame_dma_spm.mlir
LAUNCH := $(ROOT)/gemm_64x64x64_launch.c
RUNTIME := $(REPO)/runtime/boscame/runtime
UART := $(REPO)/runtime/boscame/uart
PLATFORM := $(REPO)/runtime/boscame/nh_e6
FPGA ?= 4

RISCV_FLAGS := -march=rv64imafdc_zicsr_zifencei \
  -mabi=lp64d -mcmodel=medany -mno-relax -O2 -nostdlib -ffreestanding \
  -fno-builtin -fno-pie -ffunction-sections -fdata-sections \
  -I$(ROOT) -I$(RUNTIME) -I$(UART) -I$(PLATFORM)
# AME tile instructions do not require RVV.  Keep both the oracle and llc's
# memref-descriptor copies scalar to avoid unvalidated vector configurations.
# Freestanding builds omit programming assertions; wait's fatal checks remain.
C_FLAGS := $(RISCV_FLAGS) -fno-vectorize -fno-slp-vectorize '-DBOSC_AME_DMA_ASSERT(x)='
LLC_FLAGS := -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,-v,+xboscame

.PHONY: all check board audit pmpcheck run disasm clean

all: $(BUILD)/$(NAME).nr.o check board

$(BUILD):
	mkdir -p $@

# --- 1. the unified pass ------------------------------------------------
# `--lower-bosc-ame` runs the SPM/DMA contract checks, compiles every
# `bosc_ame.spm.view` to the fixed bank base address, lowers the DMA requests
# to the runtime ABI, lowers the tile operations to `bosc_ame.intr.*` (with the
# configuration and the byte strides the hardware needs) and finishes the
# memref/func/arith/scf/... lowering to LLVM.  Which tile and accumulator
# registers hold the values is left to LLVM's register allocator.
#
# This is the entry point the example ships with: it is the one every pipeline
# uses, and the tile operations must be handled by it.  The FPGA tile entry
# point `--lower-bosc-ame-tile` runs the same lowering body, so the rule below
# also lowers the same kernel through it and `check` asserts that the two
# results are identical on this kernel - the two entry points cannot drift.
$(BUILD)/$(NAME).llvm.mlir: $(KERNEL) $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT)) | $(BUILD)
	$(BUDDY_OPT) $< --lower-bosc-ame -o $@

# --- 1b. the FPGA tile entry point, for the equivalence check ------------
$(BUILD)/$(NAME).tile.llvm.mlir: $(KERNEL) $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT)) | $(BUILD)
	$(BUDDY_OPT) $< --lower-bosc-ame-tile -o $@

# --- 2. LLVM dialect -> LLVM IR ----------------------------------------
$(BUILD)/$(NAME).ll: $(BUILD)/$(NAME).llvm.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_TRANSLATE))
	$(BUDDY_TRANSLATE) --buddy-to-llvmir $< -o $@

# --- 3. RISC-V assembly with the AME instructions -----------------------
$(BUILD)/$(NAME).s: $(BUILD)/$(NAME).ll $(BUILD_MAKEFILE) $(wildcard $(LLC))
	$(LLC) $< -filetype=asm $(LLC_FLAGS) -o $@

# --- 4. NR post-processing: fixed-register words + fences ---------------
$(BUILD)/$(NAME).encoded.s: $(BUILD)/$(NAME).s $(TOOLS)/ame_to_word.py
	$(PYTHON) $(TOOLS)/ame_to_word.py --gpr-mode=fixed < $< > $@

$(BUILD)/$(NAME).nr.s: $(BUILD)/$(NAME).encoded.s $(TOOLS)/restrict_fpga_assembly.py
	$(PYTHON) $(TOOLS)/restrict_fpga_assembly.py < $< > $@

$(BUILD)/$(NAME).nr.o: $(BUILD)/$(NAME).nr.s $(BUILD_MAKEFILE)
	$(RISCV_CC) $(RISCV_FLAGS) -c $< -o $@

# --- 5. application driver and the common DMA/SPM runtime ----------------
# The application object provides e6_main() plus the NH report hooks; boot,
# platform and UART come from the shared runtime/boscame modules and are
# linked once.
$(BUILD)/$(NAME)_launch.o: $(LAUNCH) $(RUNTIME)/bosc_ame_dma.h \
  $(PLATFORM)/nh_e6_platform.h $(UART)/uart.h $(BUILD_MAKEFILE) | $(BUILD)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

$(BUILD)/bosc_ame_dma.o: $(RUNTIME)/bosc_ame_dma.c $(RUNTIME)/bosc_ame_dma.h $(BUILD_MAKEFILE) | $(BUILD)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

# Startup: NH `_start` at 0x80000000 and E6 `e6_start` at 0x80010000.
$(BUILD)/nh_e6_boot.o: $(PLATFORM)/nh_e6_boot.S $(BUILD_MAKEFILE) | $(BUILD)
	$(RISCV_CC) $(RISCV_FLAGS) -c $< -o $@

# Platform: NH release/wait/verdict, the E6 completion protocol and the E6
# PMP/satp setup with its CSR 0xbc0 cache policies.  The SPM/DMA region values
# come from the runtime header, the DDR window from the link script.
$(BUILD)/nh_e6_platform.o: $(PLATFORM)/nh_e6_platform.c \
  $(PLATFORM)/nh_e6_platform.h $(RUNTIME)/bosc_ame_dma.h $(UART)/uart.h \
  $(BUILD_MAKEFILE) | $(BUILD)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

# Driver: the UART device, shared with the other bare-metal examples.
$(BUILD)/uart.o: $(UART)/uart.c $(UART)/uart.h $(BUILD_MAKEFILE) | $(BUILD)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

# --- 6. structural check of the generated IR and assembly ---------------
# The pass owns the AME configuration and the addresses: SPM pointers, byte
# strides, per-instruction configuration and instruction order.  Check all of
# them on the real output instead of trusting that the pattern did the right
# thing.
#
# The greps below pin the pointers and the byte strides of the registered
# `llvm.riscv.bosc.*` intrinsics; `check_gemm_sequence.py` then walks the IR in
# order and verifies that each tile instruction is preceded by the
# configuration it depends on and uses the phase `mtype` of its datapath
# (0x10010 for i8, 0x10042 for the i32 accumulator).  Registers are *not*
# pinned: LLVM allocates them, and the AME post-processing below maps the
# MAME instruction fields to the board encoding.
check: $(BUILD)/$(NAME).ll $(BUILD)/$(NAME).s $(BUILD)/$(NAME).tile.llvm.mlir
	@echo "== gemm_64x64x64: checking the lowered IR and the assembly =="
	@test "$$(grep -c 'call i64 @bosc_ame_dma_start(' $(BUILD)/$(NAME).ll)" = 4 \
	  || { echo "FAIL: expected four bosc_ame_dma_start calls"; exit 1; }
	@test "$$(grep -c 'call void @bosc_ame_dma_wait(' $(BUILD)/$(NAME).ll)" = 4 \
	  || { echo "FAIL: expected four void bosc_ame_dma_wait calls"; exit 1; }
	@grep -q 'i64 536870912, i64 4096' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: A is not staged at SPM0 base 0x20000000"; exit 1; }
	@grep -q 'i64 541065216, i64 4096' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: B is not staged at SPM1 base 0x20400000"; exit 1; }
	@grep -q 'i64 536879104, i64 16384' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: the accumulator window is not SPM0 + 0x2000"; exit 1; }
	@grep -q 'call <vscale x 32 x i32> @llvm.riscv.bosc.mlce32.m.nxv32i32.i64(ptr inttoptr (i64 536879104 to ptr), i64 256)' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: no mlce32.m from SPM0 + 0x2000 with stride 256"; exit 1; }
	@grep -q 'call <vscale x 128 x i8> @llvm.riscv.bosc.mlae8.m.nxv128i8.i64(ptr inttoptr (i64 536870912 to ptr), i64 64)' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: no mlae8.m from SPM0 base with stride 64"; exit 1; }
	@grep -q 'call <vscale x 128 x i8> @llvm.riscv.bosc.mlbe8.m.nxv128i8.i64(ptr inttoptr (i64 541065216 to ptr), i64 64)' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: no mlbe8.m from SPM1 base with the physical [N,K] row stride 64"; exit 1; }
	@grep -q 'call <vscale x 32 x i32> @llvm.riscv.bosc.mqma.b.mm.nxv32i32.nxv128i8' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: no mqma.b.mm"; exit 1; }
	@grep -q 'call void @llvm.riscv.bosc.msce32.m.nxv32i32.i64(<vscale x 32 x i32>' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: no msce32.m"; exit 1; }
	@grep -q 'i64 65552' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: the i8 phase mtype 0x10010 is not programmed"; exit 1; }
	@grep -q 'i64 65602' $(BUILD)/$(NAME).ll \
	  || { echo "FAIL: the i32 accumulator mtype 0x10042 is not programmed"; exit 1; }
	@grep -q 'mlae8.m' $(BUILD)/$(NAME).s || { echo "FAIL: no MLAE8.M"; exit 1; }
	@grep -q 'mlbe8.m' $(BUILD)/$(NAME).s || { echo "FAIL: no MLBE8.M"; exit 1; }
	@grep -q 'mlce32.m' $(BUILD)/$(NAME).s || { echo "FAIL: no MLCE32.M"; exit 1; }
	@grep -q 'mqma.b.mm' $(BUILD)/$(NAME).s || { echo "FAIL: no MQMA.B.MM"; exit 1; }
	@grep -q 'msce32.m' $(BUILD)/$(NAME).s || { echo "FAIL: no MSCE32.M"; exit 1; }
	# Register numbers are the backend's choice; only the register *file* of
	# each operand is structural (a tile load writes a tile register, a load or
	# store of the accumulator uses an accumulator register, the MMA reads two
	# tiles and one accumulator).
	@grep -qE 'mlae8\.m[[:space:]]+tr[0-7]' $(BUILD)/$(NAME).s || { echo "FAIL: mlae8.m does not write a tile register"; exit 1; }
	@grep -qE 'mlbe8\.m[[:space:]]+tr[0-7]' $(BUILD)/$(NAME).s || { echo "FAIL: mlbe8.m does not write a tile register"; exit 1; }
	@grep -qE 'mlce32\.m[[:space:]]+acc[0-7]' $(BUILD)/$(NAME).s || { echo "FAIL: mlce32.m does not write an accumulator register"; exit 1; }
	@grep -qE 'mqma\.b\.mm[[:space:]]+acc[0-7], tr[0-7], tr[0-7]' $(BUILD)/$(NAME).s \
	  || { echo "FAIL: mqma.b.mm does not read acc/tr/tr and write acc"; exit 1; }
	@grep -qE 'msce32\.m[[:space:]]+acc[0-7]' $(BUILD)/$(NAME).s \
	  || { echo "FAIL: msce32.m does not read an accumulator register"; exit 1; }
	@$(PYTHON) $(TOOLS)/check_gemm_sequence.py $(BUILD)/$(NAME).ll
	@if grep -Eq '^[[:space:]]+v[a-z0-9]+[.]?[a-z0-9]*[[:space:]]' $(BUILD)/$(NAME).s; then \
	  echo "FAIL: AME-only kernel or descriptor wrapper contains RVV"; exit 1; fi
	# `--lower-bosc-ame` and `--lower-bosc-ame-tile` share one lowering body;
	# on this kernel they must produce exactly the same IR.
	@diff -q $(BUILD)/$(NAME).llvm.mlir $(BUILD)/$(NAME).tile.llvm.mlir > /dev/null \
	  || { echo "FAIL: --lower-bosc-ame and --lower-bosc-ame-tile disagree"; \
	       diff -u $(BUILD)/$(NAME).llvm.mlir $(BUILD)/$(NAME).tile.llvm.mlir | head -20; exit 1; }
	@echo "ok: DMA staging, fixed SPM addresses, the AME tile sequence and its configuration"
	@echo "ok: AME register files (LLVM allocated the exact registers)"
	@echo "ok: --lower-bosc-ame and --lower-bosc-ame-tile produce the same IR"

disasm: $(BUILD)/$(NAME).nr.o
	$(OBJDUMP) -d $(BUILD)/$(NAME).nr.o

# --- 7. bare-metal E6/NH image and its NR audit -------------------------
$(BUILD)/$(NAME).elf: $(BUILD)/nh_e6_boot.o $(BUILD)/nh_e6_platform.o \
  $(BUILD)/uart.o $(BUILD)/$(NAME)_launch.o \
  $(BUILD)/$(NAME).nr.o $(BUILD)/bosc_ame_dma.o $(ROOT)/link.ld $(BUILD_MAKEFILE)
	$(LINKER) -m elf64lriscv --gc-sections -T $(ROOT)/link.ld \
	  -Map=$(BUILD)/$(NAME).map -o $@ \
	  $(BUILD)/nh_e6_boot.o $(BUILD)/nh_e6_platform.o $(BUILD)/uart.o \
	  $(BUILD)/$(NAME)_launch.o $(BUILD)/$(NAME).nr.o \
	  $(BUILD)/bosc_ame_dma.o

board: $(BUILD)/$(NAME).elf
	$(OBJCOPY) -O binary $< $(BUILD)/$(NAME).bin
	@echo "built $(BUILD)/$(NAME).bin"

audit: $(BUILD)/$(NAME).elf pmpcheck
	$(PYTHON) $(TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) \
	  --output $(BUILD)/elf-audit.json

# The common platform reads the DDR staging window from absolute symbols that
# link.ld exports and installs the board-validated PMP/satp sequence before the
# application runs.  Check both on the image, so a shrunk window or a reordered
# setup fails here instead of on the board.
pmpcheck: $(BUILD)/$(NAME).elf
	@test "$$($(NM) $< | awk '$$3 == "__nh_e6_ddr_window_start" { print $$1 }')" \
	  = "0000000080100000" \
	  || { echo "FAIL: link.ld must export __nh_e6_ddr_window_start = 0x80100000"; exit 1; }
	@test "$$($(NM) $< | awk '$$3 == "__nh_e6_ddr_window_end" { print $$1 }')" \
	  = "0000000080300000" \
	  || { echo "FAIL: link.ld must export __nh_e6_ddr_window_end = 0x80300000"; exit 1; }
	@$(OBJDUMP) -d $< \
	  | grep -oE 'csrw[[:space:]]+(pmpaddr[0-6]|pmpcfg0|satp)|sfence[.]vma' \
	  | tr -s '[:space:]' ' ' \
	  | grep -q 'csrw pmpaddr0 csrw pmpaddr1 csrw pmpaddr2 csrw pmpaddr3 csrw pmpaddr4 csrw pmpaddr5 csrw pmpaddr6 csrw pmpcfg0 csrw satp sfence.vma' \
	  || { echo "FAIL: the PMP/satp setup sequence changed"; exit 1; }
	@echo "ok: link.ld DDR window exported and the PMP/satp sequence is unchanged"

# The remote runner is shared with the other FPGA examples; this example does
# not carry a copy of it.
run: board
	$(REPO)/examples/BOSCAMEFPGA/fpga_run.sh $(BUILD)/$(NAME).bin \
	  --fpga $(FPGA) --capture-seconds 150

clean:
	rm -rf $(BUILD)
