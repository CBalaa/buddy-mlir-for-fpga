# matmul_i8_128x1024x128 — Triton INT8 GEMM on the NH/E6 FPGA

A standalone migration of the `matmul_i8` kernel used by the ViT-B/16 W8A8
path, together with the code that turns the Triton kernel into a flat image
(`.bin`) for the BOSCAME FPGA.  It is the
[`matmul_i8_196x768x768`](../matmul_i8_196x768x768/README.md) example with the
fixed problem size changed to a small, exactly tile-aligned shape.  The same
kernel body is in that example's
[`kernels.py`](../matmul_i8_196x768x768/kernels.py):

```text
A[i8, 128, 128] row-major  @  B[i8, 1024, 128] physical [N,K]  ->  C[i32, 128, 1024]
```

The matmul is lowered by the TTIR pass `--triton-to-boscame`, so the kernel is
read as Triton IR all the way to the BOSCAME datapath: DDR is staged into SPM
through real DMA transfers, SPM feeds the AME tile registers, `bosc_ame.tile.mma`
accumulates and the result travels back through SPM to DDR.

The TTIR starts its dot accumulator at zero, then adds the original `C` in the
epilogue.  The launcher pre-loads `C` and checks the final result against an
independent serial INT32 reduction.  The kernel source ([`kernels.py`](kernels.py))
is the ViT kernel body; the shape is fixed and written down once, in
[`build.py`](build.py).

## Platform: why this image links `runtime/boscame/nh_e6`

DMA and AME are bus masters, and the delivered E6 needs three things set up on
the *E6 core* before the application touches either of them:

1. **Seven TOR PMP regions** describing the SPM banks, the two DMA register
   windows and the DDR staging window (the DMA engine reads its source through
   the PMP region that `link.ld` exports as
   `__nh_e6_ddr_window_start/end`);
2. **`satp = 0` (bare)** plus the `sfence.vma` that follows it — the samples
   hand DMA/AME physical addresses;
3. the **CSR `0xbc0` cache policy**, which `bosc_ame_dma.c` switches around
   every transfer and CPU DDR access, using the PMP regions from (1).
   DMA wait leaves the read policy active; generated CPU DDR stores explicitly
   restore the write policy, and DMA result loads use the uncached read policy.

All three live in `runtime/boscame/nh_e6/nh_e6_platform.c`, which
`nh_e6_e6_main()` runs *before* `e6_main()`.  An image that links the NR
runtime instead (`crt.S` + `nr_runtime.c` + `nr.ld`) has none of them and also
puts its arena at `0xb8000000`, outside the DDR window: the first
`bosc_ame.dma.start` is issued, its `wait` performs a completion transaction that cannot
drain, and the run hangs with no output.  The two runtimes must not be mixed,
so this example uses the NH/E6 platform, `link.ld` pins `.dma_source` /
`.dma_output` inside the window, and `make audit` fails the build if either the
window symbols or the PMP/`satp` sequence are missing from the image.

## Files

| file | role |
|------|------|
| `kernels.py` | `@triton.jit matmul_i8`, the ViT kernel body unchanged |
| `makefile` | the pipeline: Python → TTIR → BOSCAME → LLVM → RISC-V/AME → image, and the `AME` switch |
| `build.py` | the Triton frontend, the fixed shape, the toolchain defaults, the ABI bridge source and the `--llvm ame|noame` lowering |
| `link.ld` | NH/E6 layout: entry points, DDR staging windows inside the PMP window |
| `launch.c` | the application: operands, the kernel call, the INT32 oracle, the NH report, progress markers; also the `AME=0` `memcpy` shim |
| `simcheck.py` | board-free bit-exact check of the pass output through `tools/boscame_sim.py` |
| `launch-host.c`, `support.c`, `support.h` | host-only oracle (Triton-to-linalg path, no AME) |
| `build/` | generated IR, assembly, object files, ELF and the images (ignored by git) |

## Fixed shape (in `build.py`)

| constant | value | why |
|----------|-------|-----|
| `M, N, K` | `128, 1024, 128` | logical `C[M,N] = A[M,K] @ B[N,K]^T` |
| `PADDED_M` / `VALID_M` | `128` / `128` | `M` is already a multiple of the AME 16-row tile, so no row is padding |
| `BM, BN` | `16, 16` | Triton tile; `BN` is checked against the AME `nr-tile-n` |
| `BK` | `64` | two K blocks; each dot is one 64-wide AME K tile |
| `GRID` | `8 × 64 × 1` | `PADDED_M/BM × N/BN` = 512 programs |
| `SCRATCH_BYTES` | `1024` | `max(BM*alignTo(BN*4,64), max(BM,BN)*alignTo(BK,64))`; `make check` compares it with the pass's `bosc_ame.scratch_bytes` and with `launch.c` |

`BK` is deliberately not the whole `K`: with a single reduction block, Triton's
frontend folds the one-iteration `scf.for` away and merges the zero accumulator
with the `previous` epilogue into one `tt.dot` whose accumulator is a `tt.load`
— a form `--triton-to-boscame` rejects, because it requires a splat initial
accumulator plus an explicit `previous` add.  Two blocks keep the supported
`scf.for` + `tt.dot(..., acc)` + `addi(previous)` + `tt.store` structure.

Because every tile is fully valid, the pass never packs boundary rows: all 512
programs take the direct DDR → SPM DMA path, and the output epilogue scatters
only through aligned, complete tiles.

The caller supplies 64 B aligned, non-overlapping A/B/C roots and **1024 B
scratch**.  The pass initializes scratch on every program and records its
required size in `bosc_ame.scratch_bytes`.  The generated adapter calls the
actual MLIR `_mlir_ciface_triton_matmul_i8_128x1024x128` wrapper with **four
descriptor pointers** and six grid scalars (the `AME=0` bridge takes three
descriptors and no scratch).  See
[the pass documentation](../../../docs/TritonToBOSCAME.md) for the scratch
formula and supported TTIR subset.

## Pipeline (`make`)

The diagram is the `AME=1` flow.  With `AME=0` the `--triton-to-boscame` and
`--lower-bosc-ame` boxes are replaced by the plain linalg → LLVM lowering in
`build.py --llvm noame` (the pass list `make host` also uses); the asm
post-processing, the link and the audit are identical.

```text
kernels.py ──Triton ASTSource → TTIR──▶ kernel.ttir
   │                                        │
   │  (kernel.linalg.mlir also written,     │
   │   for the host oracle)                 ▼  buddy-opt --triton-to-boscame
   │                                   kernel.boscame.mlir
   │                                        │  buddy-opt --lower-bosc-ame
   │                                        ▼  buddy-translate --buddy-to-llvmir
   │                                   kernel.ll
   │                                        │  llc -mattr=...,+xboscame
   │                                        ▼
   │                                   kernel.s
   │                                        │  ame_to_word.py --gpr-mode=fixed
   │                                        │  restrict_fpga_assembly.py
   │                                        ▼
   │                                   kernel.nr.s ──clang -c──▶ kernel.o
   │                                        │  + platform + DMA runtime
   └────────────────────────────────────────┤  + launch.o + the ABI bridge
                                            ▼  ld.lld -T link.ld
                                       matmul_i8_128x1024x128.elf
                                            │  check_nr_elf.py, llvm-objcopy -O binary
                                            ▼
                                       matmul_i8_128x1024x128.bin
```

| target | what it does |
|--------|--------------|
| `make ttir` | Triton frontend: `kernel.ttir` (and `kernel.linalg.mlir`) |
| `make boscame` | `buddy-opt --triton-to-boscame=dma-abi=args_aligned64_nonalias` (AME=1) |
| `make llvm` | AME=1: `--lower-bosc-ame` + `buddy-translate`; AME=0: `build.py --llvm noame` |
| `make asm` | `llc` + the AME fixed-register/fence post-processing |
| `make bin` | platform + runtime + driver objects, `ld.lld -T link.ld`, audit, `.bin` |
| `make audit` | the DDR window symbols and the platform's PMP/`satp` sequence are in the image |
| `make check` | AME=1: five AME instructions in the NR asm + scratch fits; AME=0: zero AME instructions; both: NR ELF audit passed |
| `make debug` | progress marker per program; oracle for the first 16 rows |
| `make simcheck` | the pass output runs bit-exactly in `tools/boscame_sim.py` (AME=1) |
| `make host` | host oracle from the Triton-to-linalg path |
| `make clean` | remove `build/` |

## AME=1 / AME=0: with and without `--triton-to-boscame`

`AME` selects the datapath and therefore the output directory, so both images
can exist side by side:

```bash
make                #  AME=1 (default): pass on,  build/matmul_i8_128x1024x128/matmul_i8_128x1024x128.bin
make AME=0          #  pass off:              build/matmul_i8_128x1024x128/noame/matmul_i8_128x1024x128.noame.bin
make AME=0 debug    #  ... with per-program markers and a 16-row oracle
```

| | `AME=1` (default) | `AME=0` |
|---|---|---|
| TTIR rewrite | `--triton-to-boscame` + `--lower-bosc-ame` | none: `kernel.linalg.mlir` → `build.py --llvm noame` |
| who computes `C = C + A·Bᵀ` | DMA → SPM → AME `mqma.b.mm`, 512 programs | the E6 core, scalar RISC-V loops |
| kernel IR | `kernel.boscame.mlir`, `kernel.llvm.mlir` | `kernel.bufferized.mlir`, `kernel.stack.mlir`, `kernel.copies.mlir` |
| ABI bridge | 4 descriptors (`+` 1024 B scratch) + grid | 3 descriptors + grid (`build.py --adapter linalg-to-boscame`) |
| NR asm / image | 3565 lines, 21 AME instructions, 81648 B | 515 lines, 0 AME instructions, 70664 B |
| console tag | `matmul-i8-128x1024x128` | `matmul-i8-128x1024x128-noame` |

Everything else is held constant on purpose: the same `kernels.py`, the same
fixed shape, the same platform setup (`link.ld`, PMP/`satp`, CSR 0xbc0 policy),
the same driver with the same operands and the same INT32 oracle.  Only the
`e6_stage(0x30)` → `e6_stage(0x31)` span changes, so the two `stats cycles=`
values (and `checked_rows`/`errors`/`max_abs`) are directly comparable.

Two consequences worth knowing before the board run:

* the `AME=0` kernel is ordinary scalar code, so it is much slower — run
  `make AME=0 debug` first if the full-row oracle outlasts the tester's capture
  window;
* the scalar kernel lowers to two 1 KiB `llvm.intr.memcpy` calls (accumulator
  staging).  The host build gets them from libc; the freestanding image gets a
  small word-wise `memcpy` from `launch.c`, compiled in only for `AME=0`.

## Bring-up on the board

```bash
# 1. build (activate the environment containing Triton and triton-riscv)
conda activate boscame
cd examples/FPGA-BOSCAME/matmul_i8_128x1024x128
make debug && make        # debug first: markers + a short oracle, then the real one

# 2. copy and run through the shared tester
cp build/matmul_i8_128x1024x128/debug/matmul_i8_128x1024x128.debug.bin ~/Desktop/fpga-tester/test/
cp build/matmul_i8_128x1024x128/matmul_i8_128x1024x128.bin           ~/Desktop/fpga-tester/test/
cd ~/Desktop/fpga-tester && ./run.sh test=matmul_i8_128x1024x128.debug.bin --fpga=5

# 3. the AME=0 counterpart, for the pass-on/pass-off comparison
cp build/matmul_i8_128x1024x128/debug-noame/matmul_i8_128x1024x128.debug-noame.bin ~/Desktop/fpga-tester/test/
cp build/matmul_i8_128x1024x128/noame/matmul_i8_128x1024x128.noame.bin           ~/Desktop/fpga-tester/test/
```

The platform makes failures visible instead of hanging:

| console output | meaning |
|---|---|
| `E6 progress=0x11` | `e6_main()` entered, operands filled |
| `E6 progress=0x30` | the kernel call starts |
| `E6 progress=0x10xx` (debug image) | the *x*-th program of the 8×64 grid is running (`0x1000 + x*64 + y`); the last one printed before a `TIMEOUT` is where it stopped |
| `E6 progress=0x31` | the kernel returned |
| `E6 progress=0x200+` (debug image) | the oracle is progressing; `checked_rows` grows in the report |
| `E6 progress=0x02` / `0xee` | oracle finished clean / with mismatches |
| `PASS` / `FAIL` + `stats ...` | verdict plus `phase`, `cycles`, `checked_rows`, `errors`, `max_abs` |
| `E6 TRAP` + `mcause`/`mepc`/`mtval` | E6 raised a trap; the DMA failure line adds channel, reason, STATUS and token |
| `TIMEOUT` | E6 never reached a terminal code; the last progress marker says how far it got |

The production image checks all 128 rows (16.7M multiply-adds); the debug image
checks rows 0–15 (2.1M).  A full-row check fits the platform's bounded wait
comfortably at this size, so `make debug` is only needed for the per-program
progress markers.  The `AME=0` image does the same amount of arithmetic on the
E6 core instead of the AME, so it takes much longer — start it with
`make AME=0 debug` and read the kernel time from `stats cycles=`.  The oracle
cost is identical in both images, so it cancels out of the comparison.

## Verification that does not need the board

`make host` links the same kernel through the Triton-to-linalg path into a host
oracle that compares every element against a serial INT32 reduction:

```text
host PASS matmul_i8_128x1024x128
```

For `AME=0` that host oracle is not just a baseline: `build.py --llvm noame`
writes the very IR the `AME=0` image compiles for RISC-V, so compiling that file
for x86 and running it is a numeric check of the image's kernel:

```bash
clang -O2 -ffp-contract=off -DHOST_TEST -I. \
  build/matmul_i8_128x1024x128/noame/kernel.ll \
  build/.scratch/adapter-noame.c launch-host.c support.c -lm -o /tmp/noame-check
/tmp/noame-check     # verify matmul_i8_128x1024x128: PASS errors=0 max_abs=0
```

`make simcheck` runs the *AME* pass output — not the linalg path — in the
functional NH/E6 simulator (`tools/boscame_sim.py`: real DMA alignment/completion
contract, two 128 KiB SPM banks, AME tile semantics) for all 512 programs and
compares the DDR result with a NumPy INT32 oracle, with the scratch window
poisoned so a missing initialization would show up:

```text
simulated 512/512 programs, max_abs=0
SIM CHECK PASS: 131072 outputs match the NumPy INT32 oracle after 512 programs
```

The generated C bridge can also be executed on the host against a stub wrapper:
compiling `build.py --adapter triton-to-boscame` with a recorder in place of
`_mlir_ciface_triton_matmul_i8_128x1024x128` and calling
`_mlir_ciface_kernel_matmul_i8_128x1024x128` once must issue exactly the 512
grid calls, in order, with the correct `num_programs`, program ids and every
descriptor field (including a nonzero incoming offset that has to be folded
into the byte root).  Both the production and the `-DMATMUL_I8_TRACE=1` variant
pass, the latter also checking all 512 progress markers.

`make check` additionally enforces, per variant: the DDR window symbols and the
platform PMP/`satp` sequence (`make audit`); for `AME=1`, the scratch
requirement in `bosc_ame.scratch_bytes` against the 1024 B window in `launch.c`
and the presence of the five AME instructions in the post-processed assembly;
for `AME=0`, that the image carries **no** AME instruction at all; and for both,
the NR ELF audit (`check_nr_elf.py`: `status PASS`, no errors, no RVV).

Measured on this checkout (2026-10-13, before the board run):

| check | result |
|---|---|
| `make host` | `host PASS`, `errors=0` over 131072 outputs |
| `make AME=0` kernel IR on the host | `PASS`, `errors=0`, 131072/131072 outputs |
| `make simcheck` | 512/512 programs, `max_abs=0`, 131072/131072 outputs |
| adapter host stub | 512/512 grid calls, descriptors and markers exact |
| `make check` (AME=1) | 5 AME mnemonics in `kernel.nr.s`; scratch 1024 ≤ 1024; NR ELF audit PASS |
| `make AME=0 check` | 0 AME mnemonics; NR ELF audit PASS |
| ELF audit | 21 AME, 0 RVV, 4775 instructions; `.bin` 81648 B |
| DDR layout | `a_matrix=0x80100000`, `b_matrix=0x80104000`, `scratch=0x80124000`, `c_matrix=0x80200000` |

**The on-board run has not been performed for this example yet.**  The
recorded NH/E6 validation above is offline (host + simulator + ELF audit); the
board verdict for the 8×64 grid is still to be captured.

## Configuration

The toolchain is resolved by the shared
[`runtime/boscame/toolchain.mk`](../../../runtime/boscame/toolchain.mk): the
in-tree LLVM build first, then `PATH`.  Overridable: `BUDDY_OPT`,
`BUDDY_TRANSLATE`, `LLC`, `RISCV_CC`, `RISCV_LD`, `RISCV_OBJCOPY`, `OBJDUMP`,
`NM`, `PYTHON`, `AME` (0 or 1, default 1) and `VAR` (the debug variant).
`build.py` keeps the Triton frontend, the fixed shape, the host oracle and the
`--llvm ame|noame` lowering; `SCRATCH_BYTES` in the makefile and `launch.c`
must stay in sync (and is only used by `AME=1`).
