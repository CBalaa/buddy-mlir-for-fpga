# makefile -- the whole matmul_i8_128x1024x128 pipeline.
#
#     kernels.py (Triton)                      [make ttir]
#       -> kernel.ttir  + kernel.linalg.mlir
#     kernel.ttir --triton-to-linalg-experimental=boscame=true
#       -> kernel.frontend.mlir -> explicit one-shot bufferization
#       -> kernel.bufferized.mlir --linalg-to-boscame       [make boscame]
#       -> kernel.boscame.mlir                 (DDR -> SPM -> AME registers)
#     --lower-bosc-ame + --buddy-to-llvmir     [make llvm]
#       -> kernel.llvm.mlir + kernel.ll
#     llc -> AME fixed-register/fence post     [make asm]
#       -> kernel.nr.s
#     clang -c + ld.lld + audit + objcopy      [make bin]
#       -> build/matmul_i8_128x1024x128/matmul_i8_128x1024x128.bin
#
# `make` runs all of it.  Every stage is a file rule, so re-running only
# rebuilds what changed.
#
# AME=1 (default) uses the experimental BOSCAME path, so the matmul
# runs on the AME tile datapath (DDR -> SPM -> AME registers).  AME=0 skips
# that pass and lowers the same kernel to scalar RISC-V through the
# Triton-to-linalg path, so the E6 core computes the matmul itself.  Both
# images share the platform, the driver, the operands and the oracle, and land
# in different directories, which makes their `stats cycles=` directly
# comparable:
#
#     make             -> build/matmul_i8_128x1024x128/matmul_i8_128x1024x128.bin
#     make AME=0       -> build/matmul_i8_128x1024x128/noame/matmul_i8_128x1024x128.noame.bin
#
# The image runs on the NH/E6 platform (runtime/boscame/nh_e6), which owns the
# boot, the dual-core release/wait, the E6 address-space setup (seven TOR PMP
# regions, `satp = 0`) and the CSR 0xbc0 cache policy.  `link.ld` places the
# DDR staging buffers inside the PMP window the platform exports -- without
# both, the first DMA cannot complete and the run hangs with no diagnostic.
# (The AME=0 kernel issues no DMA; the platform setup is unchanged so the two
# images stay comparable.)
#
# Debug variants (each built into its own directory, combinable with AME=0):
#     make debug          progress marker per program + a 16-row on-board oracle
#     make AME=0 debug    the same for the scalar E6 image
# `make host` keeps the Triton-to-linalg path for the host oracle.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR :=

ROOT := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
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

NAME := matmul_i8_128x1024x128
OUT := $(ROOT)/build/$(NAME)
SCRATCH := $(ROOT)/build/.scratch
CONFIG := $(SCRATCH)/matmul_i8_config.h
RUNTIME := $(REPO)/runtime/boscame/runtime
UART := $(REPO)/runtime/boscame/uart
PLATFORM := $(REPO)/runtime/boscame/nh_e6
NR := $(REPO)/runtime/boscame/nr

# Legacy launcher scratch facade; the experimental kernel stages on its
# aligned DDR stack and does not take this buffer.
SCRATCH_BYTES := 1024

# --- datapath selection: experimental BOSCAME or scalar CPU ---
AME ?= 1
ifeq ($(filter 0 1,$(AME)),)
$(error AME must be 0 or 1, got '$(AME)')
endif

# --- fence policy: board fences around AME/RVV memory instructions -------
# The assembly filter normally inserts a `fence rw, rw` pair around every AME
# and vector memory instruction. This directory defaults to the raw instruction
# stream so the generated kernel.nr.s can be inspected without them; the ELF
# audit is relaxed accordingly. Build with `make FENCES=1` for the fenced image.
FENCES ?= 0
ifeq ($(filter 0 1,$(FENCES)),)
$(error FENCES must be 0 or 1, got '$(FENCES)')
endif
FENCE_FLAGS := $(if $(filter 1,$(FENCES)),,--no-fences)
AUDIT_FENCE_FLAGS := $(if $(filter 1,$(FENCES)),,--allow-unfenced)

# --- build variant (empty for the default production image) -------------
# `VAR` (debug/bisection) and the datapath both select a directory and a file
# suffix, so every variant keeps its own IR, objects, image and audit:
#   make                -> build/<name>/<name>.bin
#   make debug          -> build/<name>/debug/<name>.debug.bin
#   make AME=0          -> build/<name>/noame/<name>.noame.bin
#   make AME=0 debug    -> build/<name>/debug-noame/<name>.debug-noame.bin
VAR ?=
AME_VAR := $(if $(filter 0,$(AME)),noame,)
VARIANT := $(strip $(VAR)$(if $(and $(VAR),$(AME_VAR)),-)$(AME_VAR))
V_DIR := $(OUT)$(if $(VARIANT),/$(VARIANT))
V_NAME := $(NAME)$(if $(VARIANT),.$(VARIANT),)
V_CFLAGS := $(if $(filter debug,$(VAR)),-DMATMUL_I8_TRACE=1 -DMATMUL_I8_CHECK_ROWS=16) \
  $(if $(filter boardcheck,$(VAR)),-DMATMUL_I8_BOARD_CHECK=1 -DMATMUL_I8_TRACE=1 -DMATMUL_I8_CHECK_ROWS=128) \
  -DMATMUL_I8_USE_AME=$(AME)
V_AME := --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias"
# ABI bridge: three typed unranked descriptors + six grid scalars. The new
# kernel needs no caller scratch. Retain the launcher facade for legacy users.
AME_TAG := $(if $(filter 0,$(AME)),noame,ame)
ADAPTER := $(if $(filter 0,$(AME)),linalg-to-boscame,experimental-boscame)
ADAPTER_C := $(SCRATCH)/adapter-$(AME_TAG).c
OBJ := $(V_DIR)/obj
ELF := $(V_DIR)/$(V_NAME).elf
BIN := $(V_DIR)/$(V_NAME).bin

RISCV_FLAGS := -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -fno-pie \
  -ffunction-sections -fdata-sections \
  -I$(ROOT) -I$(RUNTIME) -I$(UART) -I$(PLATFORM) -I$(NR) \
  -I$(SCRATCH)
# Freestanding image: the DMA runtime's programming assertions compile out,
# exactly as in the other FPGA examples; `bosc_ame_dma_wait` keeps its fatal
# hardware checks. Explicit one-dimensional RVV transfers move data between
# SPM and ordinary stack buffers; the existing NR allowlist audits their forms.
C_FLAGS := $(RISCV_FLAGS) -fno-vectorize -fno-slp-vectorize '-DBOSC_AME_DMA_ASSERT(x)='
LLC_FLAGS := -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d \
  -mattr=+m,+a,+f,+d,+c,$(if $(filter 1,$(AME)),+v,-v),+xboscame -code-model=medium

.PHONY: all config ttir boscame llvm asm bin debug boardcheck host check audit simcheck clean FORCE
all: bin

config: $(CONFIG)

$(sort $(OUT) $(V_DIR) $(OBJ) $(SCRATCH)):
	mkdir -p $@

# --- 1. Python -> TTIR (and the shared linalg form for the host oracle) ---
# build.py owns the Triton frontend, the fixed problem size and the toolchain
# configuration; the makefile owns the stages and their order.
$(OUT)/kernel.ttir $(OUT)/kernel.linalg.mlir &: $(ROOT)/kernels.py $(ROOT)/build.py $(BUILD_MAKEFILE) | $(OUT) $(SCRATCH)
	$(PYTHON) -B $(ROOT)/build.py --ttir-only

ttir: $(OUT)/kernel.ttir

# --- 2. TTIR -> BOSCAME (AME=1) -----------------------------------------
ifeq ($(AME),1)
# Loads/stores independently use DMA. RVV computations consume SPM directly;
# marked, explicitly bufferized integer matmuls use typed SPM A/B inputs for AME.
# The explicit ABI promises aligned, nonalias caller buffers.
$(V_DIR)/kernel.frontend.mlir: $(OUT)/kernel.ttir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT)) | $(V_DIR)
	$(BUDDY_OPT) $< $(V_AME) -o $@

$(V_DIR)/kernel.bufferized.mlir: $(V_DIR)/kernel.frontend.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries=true buffer-alignment=64" --canonicalize --cse -o $@

$(V_DIR)/kernel.boscame.mlir: $(V_DIR)/kernel.bufferized.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --linalg-to-boscame -o $@

boscame: $(V_DIR)/kernel.boscame.mlir

# --- 3. BOSCAME -> LLVM dialect -> LLVM IR -----------------------------
# `--lower-bosc-ame` checks the SPM/DMA contract, compiles the bank views to
# fixed addresses, lowers the DMA requests to the runtime ABI, turns the tile
# operations into `bosc_ame.intr.*` with the configuration and byte strides
# the hardware needs, and finishes the memref/func/arith/scf lowering.
$(V_DIR)/kernel.llvm.mlir: $(V_DIR)/kernel.boscame.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --lower-bosc-ame -o $@

$(V_DIR)/kernel.ll: $(V_DIR)/kernel.llvm.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_TRANSLATE))
	$(BUDDY_TRANSLATE) --buddy-to-llvmir $< -o $@
else
# --- 2/3. TTIR -> linalg -> scalar LLVM IR (AME=0) ----------------------
# No `--triton-to-boscame`.  build.py owns the pass list (LOWER and
# llvm_pipeline) and lowers the linalg form it already wrote for the host
# oracle, so this image runs exactly the IR `make host` verifies numerically,
# only compiled for RISC-V instead of x86.  That is the point of the variant:
# everything except the pass and its datapath is held constant.
boscame:
	@echo "AME=0: --triton-to-boscame is skipped, so there is no kernel.boscame.mlir;" \
	  "use AME=1 for the pass output."; exit 1

$(V_DIR)/kernel.ll: $(OUT)/kernel.linalg.mlir $(ROOT)/build.py $(BUILD_MAKEFILE) | $(V_DIR)
	$(PYTHON) -B $(ROOT)/build.py --llvm noame --stage $(V_DIR)
endif

llvm: $(V_DIR)/kernel.ll

# --- 4. RISC-V assembly with the AME instructions -----------------------
$(V_DIR)/kernel.s: $(V_DIR)/kernel.ll $(BUILD_MAKEFILE) $(wildcard $(LLC))
	$(LLC) $< $(LLC_FLAGS) -o $@

# --- 5. NR post-processing: fixed-register words + fences ---------------
$(V_DIR)/kernel.encoded.s: $(V_DIR)/kernel.s $(TOOLS)/ame_to_word.py
	$(PYTHON) $(TOOLS)/ame_to_word.py --gpr-mode=fixed --preserve-matrix-registers < $< > $@

# FORCE keeps the fence policy switch effective without relinking when the
# filtered output is byte-identical (the recipe only replaces the file then).
$(V_DIR)/kernel.nr.s: $(V_DIR)/kernel.encoded.s $(TOOLS)/restrict_fpga_assembly.py FORCE
	@set -eu; \
	tmp="$@.tmp"; \
	trap 'rm -f "$$tmp"' EXIT INT TERM; \
	$(PYTHON) $(TOOLS)/restrict_fpga_assembly.py $(FENCE_FLAGS) --order-vector-copies < $< > "$$tmp"; \
	test -s "$$tmp" || { echo "error: assembly filter produced no output" >&2; exit 1; }; \
	if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; else mv -f "$$tmp" "$@"; fi; \
	trap - EXIT INT TERM

$(V_DIR)/kernel.o: $(V_DIR)/kernel.nr.s $(BUILD_MAKEFILE)
	$(RISCV_CC) $(RISCV_FLAGS) -march=rv64imafdc$(if $(filter 1,$(AME)),v)_zicsr_zifencei -c $< -o $@

asm: $(V_DIR)/kernel.nr.s

# --- 6. application driver, ABI bridge, platform and DMA runtime ---------
# The MLIR wrapper takes three typed unranked descriptor pointers plus grid
# scalars. The launcher adapter validates the DMA ABI and uses element counts.
# Boot, platform, UART and DMA runtime are linked once in either variant.
$(ADAPTER_C): FORCE $(ROOT)/build.py $(BUILD_MAKEFILE) | $(SCRATCH)
	@set -eu; \
	tmp="$@.tmp"; \
	trap 'rm -f "$$tmp"' EXIT INT TERM; \
	$(PYTHON) -B $(ROOT)/build.py --adapter $(ADAPTER) > "$$tmp"; \
	test -s "$$tmp" || { echo "error: adapter generator produced no C source" >&2; exit 1; }; \
	if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; else mv -f "$$tmp" "$@"; fi; \
	trap - EXIT INT TERM

FORCE:

$(CONFIG): FORCE $(ROOT)/build.py $(BUILD_MAKEFILE) | $(SCRATCH)
	@set -eu; \
	tmp="$@.tmp"; \
	trap 'rm -f "$$tmp"' EXIT INT TERM; \
	$(PYTHON) -B $(ROOT)/build.py --config > "$$tmp"; \
	test -s "$$tmp" || { echo "error: shape config generator produced no header" >&2; exit 1; }; \
	if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; else mv -f "$$tmp" "$@"; fi; \
	trap - EXIT INT TERM

$(OBJ)/adapter.o: $(ADAPTER_C) $(CONFIG) $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) $(V_CFLAGS) -c $< -o $@

$(OBJ)/launch.o: $(ROOT)/launch.c $(CONFIG) $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) $(V_CFLAGS) -c $< -o $@

$(OBJ)/bosc_ame_dma.o: $(RUNTIME)/bosc_ame_dma.c $(RUNTIME)/bosc_ame_dma.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

$(OBJ)/nh_e6_boot.o: $(PLATFORM)/nh_e6_boot.S $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(RISCV_FLAGS) -c $< -o $@

$(OBJ)/nh_e6_platform.o: $(PLATFORM)/nh_e6_platform.c $(PLATFORM)/nh_e6_platform.h \
  $(RUNTIME)/bosc_ame_dma.h $(UART)/uart.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

$(OBJ)/uart.o: $(UART)/uart.c $(UART)/uart.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

IMAGE_OBJECTS := $(OBJ)/nh_e6_boot.o $(OBJ)/nh_e6_platform.o \
  $(OBJ)/uart.o $(OBJ)/bosc_ame_dma.o $(OBJ)/launch.o \
  $(OBJ)/adapter.o $(V_DIR)/kernel.o

# --- 7. NH/E6 image, its audit and the flat .bin ------------------------
$(ELF): $(IMAGE_OBJECTS) $(ROOT)/link.ld $(BUILD_MAKEFILE)
	$(LINKER) -m elf64lriscv --gc-sections -T $(ROOT)/link.ld \
	  -Map=$(V_DIR)/$(V_NAME).map -o $@ $(IMAGE_OBJECTS)

# The staging windows must sit inside the PMP region the platform exports, and
# the platform's PMP/satp sequence must be in the image: a memset that drops it
# hangs on the first DMA instead of failing visibly.
audit: $(ELF)
	@test "$$($(NM) $< | awk '$$3 == "__nh_e6_ddr_window_start" { print $$1 }')" \
	  = "0000000080100000" \
	  || { echo "FAIL: link.ld must export __nh_e6_ddr_window_start = 0x80100000"; exit 1; }
	@test "$$($(NM) $< | awk '$$3 == "__nh_e6_ddr_window_end" { print $$1 }')" \
	  = "00000000802e0000" \
	  || { echo "FAIL: link.ld must export __nh_e6_ddr_window_end = 0x802e0000"; exit 1; }
	@$(OBJDUMP) -d $< \
	  | grep -oE 'csrw[[:space:]]+(pmpaddr[0-6]|pmpcfg0|satp)|sfence[.]vma' \
	  | tr -s '[:space:]' ' ' \
	  | grep -q 'csrw pmpaddr0 csrw pmpaddr1 csrw pmpaddr2 csrw pmpaddr3 csrw pmpaddr4 csrw pmpaddr5 csrw pmpaddr6 csrw pmpcfg0 csrw satp sfence.vma' \
	  || { echo "FAIL: the platform's PMP/satp setup sequence is missing"; exit 1; }
	@echo "ok: DDR window exported and the PMP/satp setup sequence is in the image"

$(BIN): $(ELF) $(TOOLS)/check_nr_elf.py
	$(PYTHON) $(TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) $(AUDIT_FENCE_FLAGS) --output $(V_DIR)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
	@echo "built $@"

bin: $(BIN) audit

# --- 8. build variants --------------------------------------------------
# Progress markers per program and a 16-row oracle; `make AME=0 debug` does the
# same for the scalar image (the command-line AME value is inherited).
debug:
	$(MAKE) VAR=debug bin

# Two complete launches, signed-i8 extremes, INT32 wrapping, input/DDR guards
# and a full 128-row oracle. No SPM reset separates the two launches.
boardcheck:
	$(MAKE) VAR=boardcheck check

# --- 9. host oracle from the Triton-to-linalg path ----------------------
# Always the scalar path, whatever AME is: for AME=0 this is the numeric
# reference of the very IR the image carries, for AME=1 it is the
# pass-independent baseline.
host:
	$(PYTHON) -B $(ROOT)/build.py --host

check: bin
ifeq ($(AME),1)
# BK=128 produces two internal 64-wide blocks. C is loaded and stored once
# around the two A/B loads and MMA operations.
	@test "$$(grep -cE '# (mlae8|mlbe8|mlce32|msce32|mqma)' $(V_DIR)/kernel.nr.s)" \
	  = "8" || { echo "FAIL: expected eight AME instructions for two K blocks in the NR asm"; exit 1; }
	@grep -q '# mqma.b.mm' $(V_DIR)/kernel.nr.s || { echo "FAIL: no mqma"; exit 1; }
	$(PYTHON) -B $(ROOT)/build.py --check-abi $(V_DIR)/kernel.ll
else
	@test "$$(grep -cE '# (mlae8|mlbe8|mlce32|msce32|mqma)' $(V_DIR)/kernel.nr.s)" \
	  = "0" || { echo "FAIL: the AME=0 image must not carry AME instructions"; exit 1; }
endif
	@python3 -c "import json,sys; a=json.load(open('$(V_DIR)/elf-audit.json')); \
	  ok = a['status'] == 'PASS' and not a['errors'] and \
	       ($(AME) == 0 or (a['ame_instructions'] > 0 and \
	        (a['ame_store_instructions'] > 0 or a['vector_memory_instructions'] > 0))); \
	  sys.exit(0 if ok else 1)" \
	  || { echo "FAIL: the NR ELF audit did not pass"; exit 1; }
	@echo "ok: $(V_NAME) passed the $(if $(filter 0,$(AME)),scalar,AME) checks and the NR ELF audit"

# --- 10. AME-path functional check that needs no board ------------------
# `make host` covers the scalar (Triton-to-linalg) path only; this runs the
# pass output in tools/boscame_buffer_sim.py (functional DMA/SPM/AME model) for every
# program of the grid and compares all outputs with a NumPy INT32 oracle.
simcheck:
	@test "$(AME)" = "1" || { echo "FAIL: simcheck checks the pass output; build with AME=1 (or use 'make host' for the scalar path)"; exit 1; }
	$(PYTHON) -B $(ROOT)/simcheck.py --module $(V_DIR)/kernel.boscame.mlir

clean:
	rm -rf $(ROOT)/build
