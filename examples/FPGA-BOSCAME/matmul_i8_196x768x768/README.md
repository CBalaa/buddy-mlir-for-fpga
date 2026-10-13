# matmul_i8_196x768x768 — Triton INT8 GEMM on the NH/E6 FPGA

A standalone migration of the `matmul_i8` kernel from
[`vit-base-patch16-224/triton/kernels.py`](../vit-base-patch16-224/triton/kernels.py),
together with the code that turns the Triton kernel into a flat image (`.bin`)
for the BOSCAME FPGA.  The matmul is lowered by the TTIR pass
`--triton-to-boscame`, so the kernel is read as Triton IR all the way to the
BOSCAME datapath: DDR is staged into SPM through real DMA transfers, SPM feeds
the AME tile registers, `bosc_ame.tile.mma` accumulates and the result travels
back through SPM to DDR.

```text
A[i8, 196, 768] row-major  @  B[i8, 768, 768] physical [N,K]  ->  C[i32, 196, 768]
```

The TTIR starts its dot accumulator at zero, then adds the original `C` in
the epilogue. The launcher pre-loads `C` and checks the final result against an
independent serial INT32 reduction. The kernel source ([`kernels.py`](kernels.py)) is unchanged
from the ViT example; the shape is fixed and written down once, in
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
| `kernels.py` | `@triton.jit matmul_i8`, copied verbatim from the ViT example |
| `makefile` | the pipeline: Python → TTIR → BOSCAME → LLVM → RISC-V/AME → image, and the `AME` switch |
| `build.py` | the Triton frontend, the fixed shape, the toolchain defaults, the ABI bridge source and the `--llvm ame|noame` lowering |
| `link.ld` | NH/E6 layout: entry points, DDR staging windows inside the PMP window |
| `launch.c` | the application: operands, the kernel call, the INT32 oracle, the NH report, progress markers; also the `AME=0` `memcpy` shim |
| `launch-host.c`, `support.c`, `support.h` | host-only oracle (Triton-to-linalg path, no AME) |
| `build/` | generated IR, assembly, object files, ELF and the images (ignored by git) |

## Fixed shape (in `build.py`)

| constant | value | why |
|----------|-------|-----|
| `M, N, K` | `196, 768, 768` | logical `C[M,N] = A[M,K] @ B[N,K]^T` |
| `PADDED_M` / `VALID_M` | `208` / `196` | the AME tile has a fixed 16-row M; rows `196..207` are mask-excluded, so the pass never reads or writes them |
| `BM, BN` | `16, 16` | Triton tile; `BN` is checked against the AME `nr-tile-n` |
| `BK` | `256` | three K blocks for `K=768` |
| `GRID` | `13 × 48 × 1` | `PADDED_M/BM × N/BN` |

`PADDED_M` determines the grid only. A and C retain their original 196 rows.
For boundary tiles the pass copies valid elements into aligned **DDR scratch**,
fills masked lanes with `tt.load`'s `other`, and transfers the packed rows into
SPM. DMA never reads masked original addresses. The final result travels from
SPM to scratch; CPU code adds previous C and writes only valid output elements.
All CPU memory accesses in the generated kernel target DDR.

The caller supplies 64 B aligned, non-overlapping A/B/C roots and **4096 B
scratch**. The pass initializes scratch on every program and records its
required size in `bosc_ame.scratch_bytes`. The generated adapter calls the actual
MLIR `_mlir_ciface_triton_matmul_i8_196x768x768` wrapper with **four descriptor
pointers** and six grid scalars. See [the pass documentation](../../../docs/TritonToBOSCAME.md)
for the scratch formula and supported TTIR subset.

## Pipeline (`make`)

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
                                       matmul_i8_196x768x768.elf
                                            │  check_nr_elf.py, llvm-objcopy -O binary
                                            ▼
                                       matmul_i8_196x768x768.bin
```

| target | what it does |
|--------|--------------|
| `make ttir` | Triton frontend: `kernel.ttir` (and `kernel.linalg.mlir`) |
| `make boscame` | `buddy-opt --triton-to-boscame=dma-abi=args_aligned64_nonalias` (AME=1) |
| `make llvm` | AME=1: `--lower-bosc-ame` + `buddy-translate`; AME=0: `build.py --llvm noame` |
| `make asm` | `llc` + the AME fixed-register/fence post-processing |
| `make bin` | platform + runtime + driver objects, `ld.lld -T link.ld`, audit, `.bin` |
| `make audit` | the DDR window symbols and the platform's PMP/`satp` sequence are in the image |
| `make check` | AME=1: five AME instructions in the NR asm; AME=0: zero AME instructions; both: NR ELF audit passed |
| `make debug` | progress marker per program; oracle for rows 0–15 and 192–195 |
| `make nocheck` | build a separate image that skips the scalar result oracle |
| `make host` | host oracle from the Triton-to-linalg path |
| `make run` | upload through `examples/BOSCAMEFPGA/fpga_run.sh` |
| `make clean` | remove `build/` |

## AME=1 / AME=0: with and without `--triton-to-boscame`

`AME` selects the datapath and therefore the output directory, so both images
can exist side by side:

```bash
make                #  AME=1 (default): pass on,  build/matmul_i8_196x768x768/matmul_i8_196x768x768.bin
make AME=0          #  pass off:              build/matmul_i8_196x768x768/noame/matmul_i8_196x768x768.noame.bin
make AME=0 debug    #  ... with per-program markers and the 20-row oracle
make AME=0 nocheck  #  ... with no oracle at all (kernel time only)
```

| | `AME=1` (default) | `AME=0` |
|---|---|---|
| TTIR rewrite | `--triton-to-boscame` + `--lower-bosc-ame` | none: `kernel.linalg.mlir` → `build.py --llvm noame` |
| who computes `C = C + A·Bᵀ` | DMA → SPM → AME `mqma.b.mm`, 624 programs | the E6 core, scalar RISC-V loops |
| kernel IR | `kernel.boscame.mlir`, `kernel.llvm.mlir` | `kernel.bufferized.mlir`, `kernel.stack.mlir`, `kernel.copies.mlir` |
| ABI bridge | 4 descriptors (`+` 4096 B scratch) + grid | 3 descriptors + grid (`build.py --adapter linalg-to-boscame`) |
| NR asm / image | 5175 lines, 21 AME instructions, 86944 B | 521 lines, 0 AME instructions, 70424 B |
| console tag | `matmul-i8-196x768x768` | `matmul-i8-196x768x768-noame` |

Everything else is held constant on purpose: the same `kernels.py`, the same
fixed shape, the same platform setup (`link.ld`, PMP/`satp`, CSR 0xbc0 policy),
the same driver with the same operands and the same INT32 oracle.  Only the
`e6_stage(0x30)` → `e6_stage(0x31)` span changes, so the two `stats cycles=`
values (and `checked_rows`/`errors`/`max_abs`) are directly comparable.

Two consequences worth knowing before the board run:

* the `AME=0` kernel is ordinary scalar code, so it is much slower here: this
  shape is 115M multiply-adds and the `nocheck` image exists precisely to
  measure the kernel without the equally expensive scalar oracle.  Start with
  `make AME=0 nocheck` (or `make AME=0 debug`), read `stats cycles=`, and add
  the oracle only when the run fits the tester's capture window;
* the scalar kernel lowers to two 1 KiB `llvm.intr.memcpy` calls (accumulator
  staging).  The host build gets them from libc; the freestanding image gets a
  small word-wise `memcpy` from `launch.c`, compiled in only for `AME=0`.

## Bring-up on the board

```bash
# 1. build (activate the environment containing Triton and triton-riscv)
conda activate boscame
cd examples/FPGA-BOSCAME/matmul_i8_196x768x768
make debug && make        # debug first: markers + a short oracle, then the real one

# 2. copy and run through the shared tester
cp build/matmul_i8_196x768x768/debug/matmul_i8_196x768x768.debug.bin ~/Desktop/fpga-tester/test/
cp build/matmul_i8_196x768x768/matmul_i8_196x768x768.bin           ~/Desktop/fpga-tester/test/
cd ~/Desktop/fpga-tester && ./run.sh test=matmul_i8_196x768x768.debug.bin --fpga=5

# 3. the AME=0 counterparts, for the pass-on/pass-off comparison
make AME=0 nocheck        # this shape is slow without AME: kernel time first
cp build/matmul_i8_196x768x768/nocheck-noame/matmul_i8_196x768x768.nocheck-noame.bin ~/Desktop/fpga-tester/test/
cp build/matmul_i8_196x768x768/noame/matmul_i8_196x768x768.noame.bin                 ~/Desktop/fpga-tester/test/
```

The platform makes failures visible instead of hanging:

| console output | meaning |
|---|---|
| `E6 progress=0x11` | `e6_main()` entered, operands filled |
| `E6 progress=0x30` | the kernel call starts |
| `E6 progress=0x10xx` (debug image) | the *x*-th program of the 13×48 grid is running (`0x1000 + x*48 + y`); the last one printed before a `TIMEOUT` is where it stopped |
| `E6 progress=0x31` | the kernel returned |
| `E6 progress=0x200+` (debug image) | the oracle is progressing; `checked_rows` grows in the report |
| `E6 progress=0x02` / `0xee` | oracle finished clean / with mismatches |
| `PASS` / `FAIL` + `stats ...` | verdict plus `phase`, `cycles`, `checked_rows`, `errors`, `max_abs` |
| `E6 TRAP` + `mcause`/`mepc`/`mtval` | E6 raised a trap; the DMA failure line adds channel, reason, STATUS and token |
| `TIMEOUT` | E6 never reached a terminal code; the last progress marker says how far it got |

The oracle re-computes a prefix of `C` (`MATMUL_I8_CHECK_ROWS`) plus the final
partial tile: 20 rows in the debug image, all 196 rows in the production image
(115M multiply-adds). If the
report shows `TIMEOUT` while `checked_rows` was still growing and `errors=0`,
the kernel finished and only the check outran the platform's bounded wait — use
the debug image, or an image built with a smaller `MATMUL_I8_CHECK_ROWS`, to
get a verdict inside the window.

To disable result validation entirely, build the separate `nocheck` image:

```bash
make nocheck
# Upload and run this image through the existing FPGA runner:
make VAR=nocheck run
```

The image is `build/matmul_i8_196x768x768/nocheck/matmul_i8_196x768x768.nocheck.bin`.
It skips `validate()` completely, so no `0x200+` oracle progress markers appear.
The report includes `validation=skipped checked_rows=0`; `PASS` then means the
kernel returned successfully, with no claim of numeric validation. The
`cycles` field still measures the kernel only. Plain `make` retains full
validation, and `make debug` retains the 20-row check. Each variant has its own
output directory, so switching between them requires no clean rebuild. The same
holds for the datapath variants (`AME=0`, `AME=0 debug`, `AME=0 nocheck`).

## Verification that does not need the board

`make host` links the same kernel through the Triton-to-linalg path into a host
oracle that compares every element against a serial INT32 reduction:

```text
host PASS matmul_i8_196x768x768
```

For `AME=0` that host oracle is not just a baseline: `build.py --llvm noame`
writes the very IR the `AME=0` image compiles for RISC-V, so compiling that file
for x86 and running it is a numeric check of the image's kernel:

```bash
clang -O2 -ffp-contract=off -DHOST_TEST -I. -I../../../runtime/boscame/nr \
  build/matmul_i8_196x768x768/noame/kernel.ll \
  build/.scratch/adapter-noame.c launch-host.c support.c -lm -o /tmp/noame-check
/tmp/noame-check     # verify matmul_i8_196x768x768: PASS errors=0 max_abs=0
```

`make bin` additionally enforces, in order: the DDR window symbols and the
platform PMP/`satp` sequence (`make audit`), the NR ELF audit
(`check_nr_elf.py`: `status PASS`, no errors, `ame_instructions > 0`, no RVV),
and the presence of the five AME instructions in the post-processed assembly
(`make check`).  For this kernel the audit reports 21 AME instructions and no
RVV instruction in the whole image.

The pass itself has its own suite (`tests/Conversion/TritonToBOSCAME/`, run by
`ninja -C build check-buddy`): structure, diagnostics and numeric/memory tests
that simulate the pass output and compare it bit for bit against independent
oracles, including this kernel's boundary mask and masked store, nonzero
`other`/accumulator initializers, K padding and reuse of poisoned scratch.
An executable ABI regression checks all descriptor fields and grid IDs across
624 calls to the generated wrapper. It also verifies that downstream lowering
preserves the pass's DMA request count and introduces no CPU SPM access.
The simulator checks the DDR read/write policy, and a regression deliberately
removes the generated transitions to ensure missing ones are detected.

## Recorded board validation

On 2026-10-13 the debug image executed all 624 programs and checked rows 0–15
and 192–195, including the final partial tile: `PASS`, `checked_rows=20`,
`errors=0`, `max_abs=0`. A full-output verifier also passed with
`checked_rows=196`, `errors=0`, `max_abs=0`: every one of the 150,528 INT32
outputs matched an independent NumPy oracle.

The full verifier links the production kernel, adapter, runtime and platform
objects. It replaces only the launcher's slow serial oracle with a generated
expected table. It does not alter the computation or extend the original
196-row allocations. The temporary verifier and evidence are in `build/trace-fix/`:
`prepare_full_oracle.py`, `full-oracle-audit.json`, `full-oracle-board.log`, and
the debug log `final-debug-board.log`. Its ELF audit also reports 21 AME
instructions and zero RVV instructions.

## Configuration

The toolchain is resolved by the shared
[`runtime/boscame/toolchain.mk`](../../runtime/boscame/toolchain.mk): the
in-tree LLVM build first, then `PATH`.  Overridable: `BUDDY_OPT`,
`BUDDY_TRANSLATE`, `LLC`, `RISCV_CC`, `RISCV_LD`, `RISCV_OBJCOPY`, `OBJDUMP`,
`NM`, `PYTHON`, `FPGA`, `AME` (0 or 1, default 1) and `VAR`
(debug/nocheck).  `build.py` keeps the Triton frontend, the fixed shape, the
host oracle and the `--llvm ame|noame` lowering.
