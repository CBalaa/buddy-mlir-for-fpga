# gemm_64x64x64 -- 64x64x64 INT8 GEMM with the BOSCAME tile operations

A single 64x64x64 signed INT8 -> INT32 matrix multiplication that stages its
operands through SPM with the DMA engines and computes with the AME:

```
DDR A/B/Z --DMA0/DMA1--> SPM0/SPM1 --tile.load_a/b--> tile registers
          --tile.mma--> accumulator register --tile.store_acc--> SPM0 --DMA1--> DDR C
```

The kernel is written in the BOSCAME dialect and lowered by `-lower-bosc-ame`.
It contains **only the computation**: no matrix register name, no `mtype`
value, no tile size and no stride.  The five `bosc_ame.tile.*` operations of
`gemm_64x64x64_ame_dma_spm.mlir` speak in matrix shapes and element types:

| operation | signature | meaning |
|-----------|-----------|---------|
| `bosc_ame.tile.load_a` | `memref<MxKxi8, #bosc_ame.spm<b>> -> vector<MxKxi8>` | load the A tile from SPM into the tile register file |
| `bosc_ame.tile.load_b` | `memref<KxNxi8, #bosc_ame.spm<b>> -> vector<KxNxi8>` | load the B tile; the operand is the *physical* `NxK` buffer described as the logical `[K, N]` with strides `[1, K]` |
| `bosc_ame.tile.load_acc` | `memref<MxNxi32, #bosc_ame.spm<b>> -> vector<MxNxi32>` | seed the accumulator from its raw INT32 SPM tile |
| `bosc_ame.tile.mma` | `(vector<MxNxi32>, vector<MxKxi8>, vector<KxNxi8>) -> vector<MxNxi32>` | `C_out[m,n] = C_in[m,n] + sum_k A[m,k] * B[k,n]` |
| `bosc_ame.tile.store_acc` | `vector<MxNxi32>, memref<MxNxi32, #bosc_ame.spm<b>>` | write the INT32 accumulator tile back to SPM |

The accumulator is an SSA value, so the accumulation chain is explicit:

```mlir
%c1 = bosc_ame.tile.mma %c0, %a0, %b0 : vector<64x64xi32>, ...
%c2 = bosc_ame.tile.mma %c1, %a1, %b1 : vector<64x64xi32>, ...
```

**Status.** Lowered, compiled, audited and *run on the board*: the run reports
`PASS` with `status=0`, `errors=0` and `max_abs=0`, i.e. all 4096 INT32 outputs
match the launcher's scalar oracle bit for bit (see "UART output" below).

## What the pass adds around those five operations

* **The hardware configuration.**  Each tile operation carries its shape and
  element type in its operand and result types, so the lowering emits the
  register-form `msettilem/msettilen/msettilek` and `msettype` intrinsics it
  depends on, immediately before the instruction.  A load never runs against
  configuration left over from a different datapath phase; the first version
  deliberately repeats configuration instead of optimizing it away.  The two
  `mtype` values are `0x10010` (i8 x i8 -> i32 MMA) and `0x10042` (i32
  accumulator).
* **The byte stride.**  A and the accumulator use their leading stride, while
  B - stored as an `NxK` buffer with K contiguous but described logically as
  `KxN` - uses the trailing stride, i.e. the physical row pitch.  On this
  kernel all three are 64 bytes for A/B and 256 bytes for the accumulator;
  `make check` asserts that the B load really uses 64 and not `strides[0] = 1`.
* **No registers at all.**  The tile operations name no register and the
  lowering emits the ordinary `bosc_ame.intr.*` operations; which tile
  (`tr0`..`tr7`) and accumulator (`acc0`..`acc7`) register holds a value is
  decided by LLVM's register allocator, exactly as for the upstream AME path.
  The AME post-processing (`ame_to_word.py --gpr-mode=fixed`) then maps the
  assembly to the board-validated `.word` encodings, and `make check` only
  asserts the register *file* of each operand (a tile load writes a tile
  register, the accumulator load/store use an accumulator register).
* **`bosc_ame.spm.view` -> a fixed SPM base address** (`bankBase[bank] +
  offset`), so the kernel ABI has **no SPM workspace argument**: the entry point
  is the four DDR buffers.
* **`bosc_ame.dma.start` / `bosc_ame.dma.wait` -> calls into the runtime ABI.**
  The token is the start call's `i64` result; the wait has no result and the
  runtime performs the completion barrier and the STATUS check itself, taking
  its error path on failure, so a normal wait return means "completed and
  succeeded" and the kernel continues straight into the AME sequence.
* The surrounding memref / func / arith / scf lowering, so the output is LLVM
  dialect ready for `buddy-translate`.

The tile operations declare their memory effects (`Read` for the three loads,
`Write` for the store), which is what lets the SPM/DMA checker reject a tile
access that overlaps a transfer whose `bosc_ame.dma.wait` has not executed yet.

## Memory layout

| window | contents |
|--------|----------|
| SPM0 + 0x0000 | A, 4096 bytes, row-major `[M, K] = [64, 64]` |
| SPM1 + 0x0000 | B, 4096 bytes, **physical** `[N, K] = [64, 64]` with K contiguous |
| SPM0 + 0x2000 | the 16384-byte INT32 accumulator tile |

DMA is byte-oriented, so every endpoint is a 1-D `i8` memref and the tile
operations read 2-D views of the same storage: `memref.view` re-types the A and
the accumulator windows (the accumulator changes element type, `i8 -> i32`,
which is exactly what `memref.view` is for), and the B view is an i8 -> i8
*layout* change expressed with `memref.reinterpret_cast` (the element type does
not change there).

`Z` is a permanent 16 KiB zero tile staged into the accumulator window: the
validated FPGA subset has no "zero the accumulator" instruction, so the kernel
seeds the accumulator with a zero tile and lets `tile.mma` accumulate on top.

## Files

| file | role |
|------|------|
| `gemm_64x64x64_ame_dma_spm.mlir` | the BOSCAME-dialect kernel: the five tile operations, the SPM views and the DMA staging |
| `gemm_64x64x64_launch.c` | the only application source: `e6_main()` fills A/B, builds the memref descriptors, calls `_mlir_ciface_gemm_64x64x64` and checks all 4096 outputs against a scalar oracle that keeps the physical `B[N, K]` layout; `nh_e6_app_tag()`/`nh_e6_app_report()` publish the GEMM statistics to NH |
| `makefile` | lower -> translate -> llc -> NR filters -> link -> audit |
| `link.ld` | fixed address layout: entries, BSS, two stacks, DMA windows, and the exported `__nh_e6_ddr_window_start`/`__nh_e6_ddr_window_end` PMP window symbols |
| `../../../runtime/boscame/nh_e6/*` | common NH/E6 platform: startup (`nh_e6_boot.S`), release/wait/verdict/report and the E6 PMP/`satp` setup plus the CSR `0xbc0` cache policies |
| `../../../runtime/boscame/uart/uart.c` | common UART driver used by the NH report |
| `../../../runtime/boscame/runtime/bosc_ame_dma.c` | DMA/SPM helper (SRC/DST/LEN/START/STATUS, CSR `0xbc0`, fences) |
| `../tools/ame_to_word.py`, `../tools/nr_isa.py` | fixed-register `.word` encodings for the board AME image |
| `../tools/restrict_fpga_assembly.py` | NR assembly validation and the fences around every AME word |
| `../tools/check_nr_elf.py` | image audit (instruction count, AME/RVV census) |
| `../tools/check_gemm_sequence.py` | verifies the lowered AME sequence: instruction order, per-instruction configuration, `mtype` phases and byte strides |

## Build

```sh
make all      # lower, assemble, structural check, link
make board    # link build/gemm_64x64x64.bin
make audit    # PMP/window guard (pmpcheck) + tools/check_nr_elf.py on the ELF
make run FPGA=4
```

`all` is the compiler pipeline in full:

```
buddy-opt  --lower-bosc-ame            # checks + tile/SPM/DMA + standard LLVM
buddy-translate --buddy-to-llvmir      # LLVM dialect -> LLVM IR
llc -mattr=-v,+xboscame                # AME instructions, scalar descriptor copies
ame_to_word.py --gpr-mode=fixed        # fixed-register `.word` encodings
restrict_fpga_assembly.py              # validate + fence every AME word
clang -c && ld.lld                     # E6/NH image
```

`-lower-bosc-ame` lowers the tile operations to the ordinary
`llvm.riscv.bosc.*` intrinsics, so `llc` needs nothing beyond
`-mattr=+xboscame` (the module is not annotated with a private subtarget
feature and register allocation stays with the backend).  This AME-only kernel
does not need RVV.  Disabling `v` in
llc also keeps the generated C-interface wrapper's memref descriptor copies
scalar; the structural check rejects RVV instructions in the generated
assembly.

## Which lowering pass

The shipped rule uses `--lower-bosc-ame`, the unified entry point, because that
is the pass every pipeline runs and the one that has to handle the tile
operations: it owns the SPM/DMA checks, the tile lowering and the standard
LLVM conversion in a single pass.  The example also carries the FPGA tile entry
point `--lower-bosc-ame-tile`, which shares that whole lowering body (same
`LLVMTypeConverter`, same conversion target, same pattern set) and differs only
in requiring the module to select a contract this build implements even before
tile operations appear.  `make check` lowers this kernel through *both* entry
points and fails if the results differ, so the two cannot drift and the tile
entry point is exercised end to end on the real kernel, not only in the lit
tests:

```sh
buddy-opt gemm_64x64x64_ame_dma_spm.mlir --lower-bosc-ame       -o build/a.mlir
buddy-opt gemm_64x64x64_ame_dma_spm.mlir --lower-bosc-ame-tile  -o build/b.mlir
diff build/a.mlir build/b.mlir     # identical
```

## What the pass generates

The DMA requests become runtime calls with the fixed SPM addresses already
folded in (`build/gemm_64x64x64.ll`):

```llvm
call i64 @bosc_ame_dma_start(i32 0, i64 %a, i64 536870912, i64 4096)   ; A -> SPM0
call i64 @bosc_ame_dma_start(i32 1, i64 %b, i64 541065216, i64 4096)   ; B -> SPM1
...
call i64 @bosc_ame_dma_start(i32 0, i64 %z, i64 536879104, i64 16384)  ; zero tile -> SPM0+0x2000
...
call i64 @bosc_ame_dma_start(i32 1, i64 536879104, i64 %c, i64 16384)  ; C: SPM0+0x2000 -> DDR
```

`536870912` = `0x20000000` (bank 0), `541065216` = `0x20400000` (bank 1),
`536879104` = `0x20002000` (bank 0 + 8192).  No root argument is passed or
needed.

The AME part of the linked image, with the v0.5 words the tools emit:

| step | instruction | word |
|------|-------------|------|
| 1 | `msettilem zero, <M>` | `04055077` |
| 2 | `msettilen zero, <N>` | `04054077` |
| 3 | `msettype zero, <e32>` | `00054077` |
| 4 | `mlce32.m acc<k>, (spm_c), 256` | `00b52077` |
| 5 | `msettilem/msettilek zero, <M/K>` | `04055077` / `04056077` |
| 6 | `msettype zero, <e8>` | `00054077` |
| 7 | `mlae8.m tr<k>, (spm_a), 64` | `04b50077` |
| 8 | `msettilen/msettilek zero, <N/K>` | `04054077` / `04056077` |
| 9 | `msettype zero, <e8>` | `00054077` |
| 10 | `mlbe8.m tr<k>, (spm_b), 64` | `08b500f7` |
| 11 | `msettilem/n/k zero, <M/N/K>`, `msettype zero, <e8>` | `04055077` / `04054077` / `04056077` / `00054077` |
| 12 | `mqma.b.mm acc<k>, tr<k>, tr<k>` | `28180877` |
| 13 | `msettilem/msettilen zero, <M/N>`, `msettype zero, <e32>` | `04055077` / `04054077` / `00054077` |
| 14 | `msce32.m acc<k>, (spm_c), 256` | `02b52077` |

This is the same AME instruction sequence as the hand-written reference
kernel; the difference is that every configuration word and every byte stride
in it was derived from the tile operations.  The register numbers of the
operands are LLVM's choice (`ame_to_word.py` rewrites them to the fixed board
fields), which is why the table writes them as `tr<k>`/`acc<k>`.

## DMA completion and error handling

The kernel has no status handling of its own.  `bosc_ame.dma.wait` lowers to
`void bosc_ame_dma_wait(channel, token)`, and the runtime validates the token
against the launch record, performs the completion barrier, reads `STATUS`
(`0x20`: bit 0 `busy`, bit 1 `bus_error`) and takes the non-returning failure
hook on error.  The shipped default hook traps; this application installs
`nh_e6_publish_dma_failure`, which first saves channel/reason/raw STATUS/token
to the platform diagnostic area, then traps.  See
`../../../runtime/boscame/nh_e6/README.md` for the platform side.

## UART output

Measured on the board (`make run FPGA=<n>`), a passing run:

```text
[5.5296MHz][ame-64x64x64] start
[5.5296MHz][ame-64x64x64] E6 progress=0x00000011
[5.5296MHz][ame-64x64x64] E6 progress=0x00000030
[5.5296MHz][ame-64x64x64] E6 progress=0x00000031
[5.5296MHz][ame-64x64x64] E6 progress=0x00000002
[5.5296MHz][ame-64x64x64] PASS
[5.5296MHz][ame-64x64x64] stats status=0x00000000 cycles=0x0000000000006d01 errors=0 max_abs=0 acc_errors=0 acc_max_abs=0[ame-64x64x64] tr/acc size probe
  msettile readback ok=0 raw_m=0 raw_n=0 raw_k=0
  max_m=64 max_n=64 max_k=64
  tr_a_bytes=4096 tr_b_bytes=4096 acc_bytes=16384
  mlenb=8192 mrlenb=256 malenb=16384
  clamp_m=0 clamp_n=0 clamp_k=0 probe_ok=0
```

What each line proves:

* `E6 progress` markers are the launcher's phase boundaries, printed as
  `0x%08x` by the platform: `0x11` after the setup, `0x30` before the kernel
  call, `0x31` after the kernel returned `status == 0`, and `0x02` after the
  scalar oracle found zero mismatches.  The `0x02` marker is only reached on the
  success path (`0xee` would mean mismatches, `0xe1` a non-zero kernel status).
* `PASS` is the NH verdict (`e6_main` returned 0), which requires both
  `status == 0` and `errors == 0`.
* `status=0x00000000` is the kernel return value and `errors=0 max_abs=0` means
  all 4096 INT32 outputs match the scalar oracle bit for bit.  With
  `max_abs = 0` there is no rounding or saturation difference at all: every
  output equals the independent reference computation.  This is the board
  validation of the whole tile path - the SPM/DMA staging, the derived
  `msettile*`/`msettype` configuration, the physical B row stride, the fixed
  backend-allocated registers and the MMA accumulation.
* `cycles=0x6d01` (27905 cycles) is the measured kernel time between the `0x30`
  and `0x31` markers.  It is informational: the example does not claim a
  throughput target.
* The `tr/acc size probe` block reports the capacities this example *assumes*
  rather than measures: `raw_*`, `readback`, `clamp_*` and `probe_ok` stay 0
  because a fixed 64x64x64 tile does not run the tile-capacity probe.
* The `stats ...` line and the `tr/acc size probe` header share one physical
  line: the probe header carries no leading newline.  A capture that appears to
  cut the second prefix short is a capture artifact, not a device fault.

The AME sequence that produced this result is the one in "What the pass
generates" above: it is generated from `bosc_ame.tile.*`, not written by hand.

`make run` uses the shared `examples/BOSCAMEFPGA/fpga_run.sh` remote runner; the
capture ends at `--capture-seconds`, so read the `PASS`/`FAIL` line in the log.

## Verify without hardware

```sh
make all      # lower, compile, assemble, structural check, link
make audit    # pmpcheck (link.ld DDR window + PMP/satp order) + NR ELF audit

# The shared platform flow (PMP/satp setup, CSR 0xbc0 policies,
# release/wait/progress/verdict/report) on the host:
cc -std=c11 -Wall -Wextra -Werror \
   -I ../../../runtime/boscame/runtime -I ../../../runtime/boscame/uart \
   -I ../../../runtime/boscame/nh_e6 \
   ../../../runtime/boscame/nh_e6/test_nh_e6_platform.c -o /tmp/nh_e6_platform_test \
   && /tmp/nh_e6_platform_test
```

`make audit` reports the image census (instruction count, AME instructions,
RVV instructions).  The local checks cannot replace the board run of the AME
sequence, the PMP setup and the CSR `0xbc0` cache policy; the run above is the
board evidence for this fixed 64x64x64 problem, and other shapes or inputs
still need their own board run.

## Changes needed to run on a board

* `link.ld` fixes the E6/NH addresses and the DDR staging window
  (`0x80100000`..`0x80300000`); it matches the platform's PMP setup.
* The launcher's four DDR buffers are 4096-byte aligned and pairwise disjoint,
  which is what `bosc_ame.dma.abi = "args_aligned64_nonalias"` promises.
* `make run FPGA=<n>` needs the remote-runner environment of
  `examples/BOSCAMEFPGA/tools/fpga_run.py` (SSH host, remote directory).
