# gemm_64x64x64_ame_dma_spm -- 64x64x64 INT8 GEMM written in the BOSCAME dialect

A single 64x64x64 signed INT8 -> INT32 matrix multiplication that stages its
operands through SPM with the DMA engines and computes with the AME:

```
DDR A/B/Z --DMA0/DMA1--> SPM0/SPM1 --MLAE8/MLBE8--> Tr0/Tr4
          --MQMA.B.MM--> Acc0 --MSCE32--> SPM0 --DMA1--> DDR C
```

The kernel is written in the BOSCAME dialect and lowered by the unified
`-lower-bosc-ame` pass.  The point of the example is to show, on one readable
kernel, everything the pass owns:

* `bosc_ame.spm.view` -> a fixed SPM base address (`bankBase[bank] + offset`),
  so the kernel ABI has **no SPM workspace argument**;
* `bosc_ame.dma.start` / `bosc_ame.dma.wait` -> calls into the runtime ABI.  The
  token is the start call's `i64` result; the wait has **no result** and the
  runtime performs the completion barrier and the STATUS check itself, taking
  its error path on failure, so a normal wait return already means "completed
  and succeeded" and the kernel continues straight into the AME sequence;
* `bosc_ame.mlae8.m` / `mlbe8.m` / `mlce32.m` / `mqma.b.mm` / `msce32.m` ->
  `bosc_ame.intr.fpga.*` with `tr0`/`tr4`/`acc0` and the fixed-slot contract;
* the surrounding memref / func / arith / scf lowering, so the output is LLVM
  dialect ready for `buddy-translate`.

## Files

| file | role |
|------|------|
| `gemm_64x64x64_ame_dma_spm.mlir` | the BOSCAME-dialect kernel, the whole point of the example |
| `gemm_64x64x64_launch.c` | the only application source: `e6_main()` fills A/B, builds the memref descriptors, calls `_mlir_ciface_gemm_64x64x64` and checks all 4096 outputs against a scalar oracle; `nh_e6_app_tag()`/`nh_e6_app_report()` publish the GEMM statistics to NH |
| `makefile` | lower -> translate -> llc -> NR filters -> link -> audit |
| `link.ld` | fixed address layout: entries, BSS, two stacks, DMA windows, and the exported `__nh_e6_ddr_window_start`/`__nh_e6_ddr_window_end` PMP window symbols |
| `../../common/nh_e6/*` | common NH/E6 platform: startup (`nh_e6_boot.S`), release/wait/verdict/report and the E6 PMP/`satp` setup plus the CSR `0xbc0` cache policies (`nh_e6_platform.c`, `.h`) |
| `../../common/uart/uart.c` | common UART driver used by the NH report |
| `../../common/runtime/bosc_ame_dma.c` | DMA/SPM helper (SRC/DST/LEN/START/STATUS, CSR `0xbc0`, fences) |

## Build

```sh
make all      # lower, assemble, structural check
make board    # link build/gemm_64x64x64.bin
make audit    # PMP/window guard (pmpcheck) + tools/check_nr_elf.py on the ELF
make run FPGA=4
```

`all` is the compiler pipeline in full:

```
buddy-opt  --lower-bosc-ame            # checks + AME/SPM/DMA + standard LLVM
buddy-translate --buddy-to-llvmir      # LLVM dialect -> LLVM IR
llc -mattr=-v,+xboscame                # AME instructions, scalar descriptor copies
ame_to_word.py --gpr-mode=fixed        # fixed-register `.word` encodings
restrict_fpga_assembly.py              # validate + fence every AME word
clang -c && ld.lld                     # E6/NH image
```

This AME-only kernel does not need RVV.  Disabling `v` in llc also keeps the
generated C-interface wrapper's memref descriptor copies scalar.  With `+v`,
llc can introduce `e64,m2` vector loads/stores even when the C driver has
auto-vectorization disabled.  The structural check rejects RVV instructions
and the final ELF audit should report 11 AME instructions and 0 RVV.

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

The AME part of `gemm_64x64x64` in the linked image, with the verified v0.5
words:

| step | instruction | word |
|------|-------------|------|
| 1 | `msettilem x0, 64` | `04055077` |
| 2 | `msettilen x0, 64` | `04054077` |
| 3 | `msettilek x0, 64` | `04056077` |
| 4 | `msettype x0, <e32>` | `00054077` |
| 5 | `mlce32.m acc0, (spm_c)` | `00b52077` |
| 6 | `msettype x0, <e8>` | `00054077` |
| 7 | `mlae8.m tr0, (spm_a)` | `04b50077` |
| 8 | `mlbe8.m tr4, (spm_b)` | `08b500f7` |
| 9 | `mqma.b.mm acc0, tr0, tr4` | `28180877` |
| 10 | `msettype x0, <e32>` | `00054077` |
| 11 | `msce32.m acc0, (spm_c)` | `02b52077` |

These are the v0.5 AME words for `MLAE8`/`MLBE8`/`MQMA.B.MM`/`MSCE32`, emitted
by the pass from the dialect instead of hand-written `.word` directives.

## DMA completion and error handling

The kernel has no status handling of its own.  `bosc_ame.dma.wait` lowers to
`void bosc_ame_dma_wait(channel, token)`, and the runtime:

1. validates the token against the source saved by `start` in software, then
   performs the completion barrier (the saved SRC write-back, fence),
2. reads `STATUS` once (`0x20`: bit 0 `busy`, bit 1 `bus_error`), fences,
3. checks the defined bits and, on failure, calls the non-returning
   `bosc_ame_dma_fail_hook` (channel, reason, raw status, token).

SRC/DST/LEN/START are write-only in the hardware contract.  The runtime never
reads them; only STATUS is read.  Each channel has one software record, consumed
by its wait, and requires serialized calls with one wait per start before reuse.
On RISC-V, accesses also use the delivered hardware reference's cache policy:
`fence iorw, iorw` then CSR `0xbc0` before configuring DMA or observing STATUS.
The delivered E6 needs write-through stores and uncached loads; an uncached
store miss cannot complete.  These policy values assume the reference's PMP
region setup; on this platform the common E6 flow installs it before `e6_main()`
runs, and the driver selects the policies through
`nh_e6_cache_writes()`/`nh_e6_cache_uncached_loads()` instead of writing CSR
`0xbc0` itself.

The shipped default hook traps.  This application installs
`nh_e6_publish_dma_failure` (common/nh_e6) as the hook: it first saves
channel/reason/raw STATUS/token to the platform diagnostic area, then traps.
The platform's E6 trap entry captures `mcause`, `mepc` and `mtval` before
publishing `0xed`; NH prints those cells and the DMA context over UART.  The
check is independent of `BOSC_AME_DMA_ASSERT`/`NDEBUG`.  A normal return from
wait means "completed and succeeded", which is why the GEMM code continues into
the AME sequence.

The runtime host test in
`examples/FPGA-BOSCAME/common/runtime/test_bosc_ame_dma.c` covers a clean
completion, `bus_error`, `busy` after the barrier, a bad channel, the
`start` "not launched" token (0), a token mismatch, missing launches and
duplicate waits, in both debug and `-DNDEBUG` builds.  It also rejects reads
of write-only registers and checks that a mismatched token writes the original
launched source back for the completion barrier.

## Notes on the kernel

* **Platform setup.**  The E6 address-space setup lives in the common NH/E6
  platform (`nh_e6_platform_init()`, run by `nh_e6_e6_main()` *before*
  `e6_main()`), not in this driver.  It installs the seven TOR PMP regions from
  the patch7 `test/src/cases/nh_e6_dma_spm/src/main.c` reference -- the DMA MMIO
  range `[0x60000000, 0x60020000)`, both SPM banks (from `bosc_ame_dma.h`) and
  the DDR staging window -- then `fence iorw, iorw`, `satp=0` and `sfence.vma`,
  because this example passes physical addresses to DMA and AME in bare mode.
  The window boundaries are the absolute symbols `link.ld` exports
  (`__nh_e6_ddr_window_start`/`__nh_e6_ddr_window_end`), so the PMP region can
  not drift from the layout; the `.dma_source`/`.dma_output` sections stay
  inside it and it is deliberately not shrunk to their size.  The driver
  itself only switches the cache policy at its phase boundaries
  (`nh_e6_cache_writes()`, via `e6_stage()`) and back to uncached loads
  (`nh_e6_cache_uncached_loads()`) before the scalar oracle reads the
  DMA-updated C buffer.  No kernel argument or page table is involved.
* **Boot and verdict.**  `_start` (NH), `e6_start` (E6), the dual-core release,
  the bounded wait, the progress/timeout handling and the PASS/FAIL/TRAP verdict
  live in `../../common/nh_e6`, shared with any future example on this platform.
  This example contributes `e6_main()`, the tag and the report body only; it no
  longer carries its own `launcher.S`.
* **`nr-fpga` target.**  The module carries `bosc_ame.target = "nr-fpga"`, so
  the pass selects the validated W8A8 register-file convention: register-form
  `msettilem/msettilen/msettilek` and `msettype`, the `bosc_ame.fpga.slot`
  attributes (`tr0` for A, `tr4` for B, `acc0` for the accumulator) and the
  `+xboscame-fpga` target feature on the lowered function.  The `nr-fpga`
  verifier also requires the accumulator memory to be `i32`, which is why the
  kernel re-types the byte SPM window with `memref.view`:
  `memref<16384xi8, spm<0>>` -> `memref<64x64xi32, spm<0>>`.
* **Zero accumulator.**  The validated subset has no "zero Acc" instruction, so
  the kernel stages a permanent 16 KiB zero DDR tile (`Z`) into the SPM
  accumulator window and seeds `Acc0` with `mlce32` before the MMA.  Passing a
  zero `C` is the reference behaviour (see `linear_ame_dma_spm.c`).
* **Byte-oriented output.**  The DMA contract is 1-D `i8`, so the output
  argument is `memref<16384xi8>`: a 64x64 INT32 tile is 16384 bytes.
* **ABI assertion.**  `bosc_ame.dma.abi = "args_aligned64_nonalias"` is the
  caller's promise that every DDR pointer is 64-byte aligned and that distinct
  pointers do not overlap.  `e6_main()` honours it with `aligned(4096)` buffers.

## Expected UART output

```text
[5.5296MHz][ame-64x64x64] start
[5.5296MHz][ame-64x64x64] E6 progress=0x11
[5.5296MHz][ame-64x64x64] E6 progress=0x30
[5.5296MHz][ame-64x64x64] E6 progress=0x31
[5.5296MHz][ame-64x64x64] PASS
[5.5296MHz][ame-64x64x64] stats status=0x0 cycles=0x... errors=0 max_abs=0 acc_errors=0 acc_max_abs=0[ame-64x64x64] tr/acc size probe
  msettile readback ok=0 raw_m=0 raw_n=0 raw_k=0
  max_m=64 max_n=64 max_k=64
  tr_a_bytes=4096 tr_b_bytes=4096 acc_bytes=16384
  mlenb=8192 mrlenb=256 malenb=16384
  clamp_m=0 clamp_n=0 clamp_k=0 probe_ok=0
```

The two report lines are printed by `nh_e6_app_report()`; note that the
`stats ...` line and the `tr/acc size probe` header share one physical line
because the probe header carries no leading newline -- the migration from
`launcher.S` preserved that byte-for-byte.  The platform owns the `start`,
`E6 progress`, verdict and trap/DMA lines.

`errors = 0` means all 4096 INT32 outputs match the scalar oracle; the earlier
`PASS` is the NH verdict (`e6_main` returned 0).  `acc_errors`/`probe_*` stay 0
because this example runs the fixed 64x64x64 tile only; it does not run a
tile-capacity probe.

`make run` waits for the last field of that report (`probe_ok=`) as its
completion marker, so the capture ends after the full report is printed.  A
FAIL/TRAP run reaches the same report; read the `PASS`/`FAIL` line in the log
rather than relying on the exit status alone.

On a trap, NH additionally reports:

```text
[ame-64x64x64] trap mcause=0x... mepc=0x... mtval=0x...
[ame-64x64x64] DMA failure channel=... reason=... status=0x... src_token=0x...
```

The DMA line appears only when the runtime failure hook was reached.  Reasons
are 1 = bad channel, 2 = invalid token or missing launch, 3 = token mismatch,
4 = bus_error, 5 = busy after the completion barrier.  STATUS bit 0 is busy
and bit 1 is bus_error.  For a direct instruction/MMIO fault, use `mepc` with
the ELF disassembly to identify the instruction; synchronous `mcause` 2 is an
illegal instruction, 5 is a load access fault and 7 is a store/AMO access fault.
The software hook also deliberately triggers an illegal instruction, so the
DMA context distinguishes it from an unsupported kernel instruction.

The existing `stats status` field is the kernel return value, not DMA STATUS.
It and `cycles` are written only after the kernel returns; they cannot establish
DMA success or identify the fault when the kernel traps.  The `readback=0` and
`probe_ok=0` fields are expected here because this example does not run the
capacity probe.

## Verify without hardware

```sh
make all      # re-runs the structural check
make audit    # pmpcheck (link.ld DDR window + PMP/satp order) + NR AME audit

# common NH/E6 platform flow (PMP/satp setup, CSR 0xbc0 policies,
# release/wait/progress/verdict/report) on the host
cc -std=c11 -Wall -Wextra -Werror \
   -I ../common/runtime -I ../common/uart -I ../common/nh_e6 \
   ../common/nh_e6/test_nh_e6_platform.c -o /tmp/nh_e6_platform_test \
   && /tmp/nh_e6_platform_test
```

`make board` builds `build/gemm_64x64x64.bin`; local checks cannot
replace the FPGA run of the AME sequence, the PMP setup and the CSR `0xbc0`
cache policy.
