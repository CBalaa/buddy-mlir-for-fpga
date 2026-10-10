# matmul_i8_196x768x768 — Triton INT8 GEMM for the NR FPGA path

A standalone migration of the `matmul_i8` kernel from
[`vit-base-patch16-224/triton/kernels.py`](../vit-base-patch16-224/triton/kernels.py),
together with the Triton-to-binary code that turns the kernel into a flat NR
image (`.bin`) for the BOSCAME FPGA.

```text
A[i8, 196, 768] row-major  @  B[i8, 768, 768] physical [N,K]  ->  C[i32, 196, 768]
```

`tl.dot` accumulates into `C` (AME `mqma.b.mm` has `+=` semantics), so the
launcher pre-loads `C` and checks the result against an independent serial
INT32 reduction. The kernel source ([`kernels.py`](kernels.py)) is unchanged
from the ViT example; the shape is fixed and written down once, in
[`build.py`](build.py).

## Files

| file | role |
|------|------|
| `kernels.py` | `@triton.jit matmul_i8`, copied verbatim from the ViT example |
| `build.py` | everything: shape, toolchain configuration, Triton export, AME lowering, NR runtime objects, NR image, host oracle |
| `makefile` | two-line wrapper (`make`, `make host`, `make run`, `make clean`) |
| `launch.c` | launcher with the independent INT32 oracle; the host `main` lives here under `HOST_TEST` (NR enters through `common/nr/crt.S` and calls `launch()`) |
| `support.c`, `support.h` | NR workspace arena and the UART report helpers |
| `build/` | generated IR, LLVM IR and the final image (ignored by git) |

## Fixed shape (in `build.py`)

| constant | value | why |
|----------|-------|-----|
| `M, N, K` | `196, 768, 768` | logical `C[M,N] = A[M,K] @ B[N,K]^T` |
| `PADDED_M` / `VALID_M` | `208` / `196` | the AME tile has a fixed 16-row M and rejects a short final `msettilem`; the generated adapter stages the padded rows and copies only the logical ones back |
| `BM, BN` | `16, 16` | Triton tile; `BN` is checked against the AME `nr-tile-n` |
| `BK` | `256` | three K blocks for `K=768` |
| `GRID` | `13 × 48 × 1` | `PADDED_M/BM × N/BN` |

## Pipeline (`python build.py`)

```text
kernels.py ──Triton ASTSource → TTIR → triton-riscv──▶ kernel.ttir, kernel.linalg.mlir
      │
      ▼  buddy-opt / buddy-translate / llc
--empty-tensor-to-alloc-tensor
--one-shot-bufferize=… buffer-alignment=64      (tensors → memrefs)
--lower-linalg-to-boscame=target=nr-fpga        (linalg.matmul → bosc_ame.mqma.b.mm)
--promote-buffers-to-stack / convert-strided-memref-copy-to-linalg
--canonicalize --cse --lower-bosc-ame + LOWER    (LLVM dialect)
--buddy-to-llvmir                                (kernel.ll, ABI-checked)
      │
      ├── --host: clang kernel.ll adapter.c launch.c support.c → run oracle
      └── default: llc → ame_to_word.py → restrict_fpga_assembly.py → clang -c
                   (+ common/nr runtime objects) → ld.lld -T common/nr/nr.ld
                   → check_nr_elf.py → llvm-objcopy -O binary
```

## Build artifacts

After `make` the build directory contains **only IR and the final image**:

```text
build/matmul_i8_196x768x768/
├── kernel.ttir               # Triton TTIR
├── kernel.linalg.mlir        # triton-riscv linalg (frontend output)
├── kernel.bufferized.mlir    # one-shot bufferize
├── kernel.ame.mlir           # BOSCAME AME dialect
├── kernel.stack.mlir         # buffer promotion
├── kernel.copies.mlir        # strided copy canonicalization
├── kernel.llvm.mlir          # LLVM dialect
├── kernel.ll                 # LLVM IR handed to llc/translate
└── matmul_i8_196x768x768.bin # final flat NR image for fpga_run.sh
```

Everything else — the generated `adapter.c`, RISC-V assembly, AME-encoded
assembly, object files, link map, ELF, audit report, host oracle binary and its
logs, plus the runtime objects — lives in `build/.scratch/`, which `build.py`
wipes at the start of a build and deletes again after a successful build. A
failed build keeps it for inspection and prints its path. `make host` is a
validation step, so its (non-AME) IR also stays in `.scratch`.

## Build

Use the `boscame` Python 3.12 environment (see
[`common/triton/README.md`](../common/triton/README.md)):

```bash
source /home/chh/miniconda3/etc/profile.d/conda.sh
conda activate boscame
cd examples/FPGA-BOSCAME/matmul_i8_196x768x768

make host         # build and run the host oracle first
make              # build build/matmul_i8_196x768x768/matmul_i8_196x768x768.bin
make run FPGA=5   # upload the flat image through ../fpga_run.sh
```

Other targets: `make clean`; `python build.py --output DIR` relocates the build
root.

## Verification (no board required)

`make host` links the *same* LLVM IR the NR build uses into the host oracle:

```text
verify matmul_i8_196x768x768: PASS errors=00000000 max_abs_error_f32_bits=00000000
```

`make` additionally enforces the NR contracts before the image is accepted:

* `verify_abi` — the entry is `(i64, ptr) × 3 + i32 × 6`, no per-grid heap allocation;
* `bosc_ame.target = "nr-fpga"` present, `bosc_ame.mqma.b.mm` present and no
  remaining `linalg.matmul`;
* `check_nr_elf.py` audit — `status PASS`, no errors, `ame_instructions > 0`.
  For this kernel: 1771 instructions, 21 AME instructions, 0 vector memory
  instructions.

The migration is regression-checked against the original example: the TTIR,
linalg IR, generated adapter, NR assembly, object files and the final `.bin`
are byte-identical to what
`vit-base-patch16-224/triton/build.py --case matmul_i8_196x768x768` produces.

## Configuration

The toolchain is resolved inside `build.py`: repository `build-migrate/bin`
then `build/bin` for `buddy-opt`/`buddy-translate`, repository
`llvm/build-2d26/bin` then `llvm/build/bin` for clang/llc/llvm-objcopy/
llvm-objdump, and `PATH` as the final fallback (same order as
`common/toolchain.mk`). The C runtime objects and the kernel therefore always
share one flag set. Every value can be overridden from the environment:

| variable | default | effect |
|----------|---------|--------|
| `AME_GPR_MODE` | `fixed` | `ame_to_word.py` operand encoding (`fixed`/`direct`) |
| `NR_COALESCE_FENCES` | `0` | experimental adjacent fence sharing; the ELF audit still runs |
| `BUDDY_OPT`, `BUDDY_TRANSLATE` | repo `build-migrate/bin` | Buddy tools |
| `LLVM_BIN` | repo `llvm/build-2d26/bin` | where clang/llc/llvm-objcopy/llvm-objdump are looked up |
| `LLC`, `HOST_CC`, `RISCV_OBJCOPY`, `RISCV_OBJDUMP` | discovered above | individual tool override |
| `RISCV_CC`, `RISCV_LD` | `clang --target=riscv64-unknown-elf`, `ld.lld -m elf64lriscv` | compiler/linker commands (shell-quoted, so extra flags are allowed) |
| `CFLAGS`, `HOST_CFLAGS`, `SCALAR_FLAGS`, `LOWER`, `AME_PASS` | built-in lists | flag-set overrides (shell-quoted) |
| `MATMUL_I8_BUILD_ROOT` | `build/` | build root |
