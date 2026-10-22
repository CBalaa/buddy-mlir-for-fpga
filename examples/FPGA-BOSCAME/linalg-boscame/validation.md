# Standalone backend validation record

This record covers the real TTIR frontend and canonical saved buffer-form
inputs. It does not reuse the older Triton GEMM example's FPGA results.
The current lowering uses typed DMA for pure SPM-to-SPM copies. The retained
board images passed with independent final-DDR readback on NR patch9: frontend
GEMM and saved-buffer LayerNorm on FPGA4, frontend LayerNorm and saved-buffer
GEMM on FPGA6. The LayerNorm images predate the current select-zero correction:
their padded centered X lanes were `0 - mean` and contributed `mean²` to
variance. Those board results validate the earlier behavior, not the current
source semantics. GEMM results are unaffected. Historical images retain their
original identities, counters and timings separately below. No comparable
baseline speedup is available.

## Genuine frontend compiler and functional checks

`kernels.py` was exported through Triton's actual AST/TTIR compiler, then
converted to tensor Linalg, bufferized separately, lowered by the independent
backend, and translated through LLVM dialect to LLVM IR. All intermediate
artifacts are saved under `build/examples/linalg-boscame/frontend/`.
The final rebuilt `.boscame.mlir`, `.llvm.mlir` and `.ll` artifacts are
byte-identical to those used by the successful numeric simulations.

| Field | GEMM | LayerNorm |
| --- | --- | --- |
| Source shape | M128 / N128 / K1024, BM64 / BN128 / BK1024 | 197 rows, 768 valid / 1024 logical lanes |
| Explicit grid | `(2,1,1)` | `(197,1,1)` |
| Source ABI | Three typed unranked full roots + six grid/pid i32 values | Four typed unranked full roots + six grid/pid i32 values |
| Generated C wrapper / RV64 adapter link / host descriptor check | Passed | Passed |
| LLVM verifier on linked adapter bitcode | Passed | Passed |
| Actual backend numeric simulation | 16384 / 16384 exact wrapping-i32 values | 151296 / 151296 values bit-identical to padded ordered-f32 oracle |
| Launch order | Program 1, then 0 | Rows 196 down to 0 |
| Guards, inputs, authorized DDR ranges, persistent poisoned SPM | Passed | Passed, including last row and shared Weight/Bias tails |
| User DMA requests / bytes per program | 4 / 262144 | 4 / 12288 |
| SPM-to-SPM DMA requests / bytes per program | 4 / 262144 | 4 / 12288 |
| Ordinary scratch DMA requests / bytes per program | 12 / 655360 | 25 / 98304 |
| Total DMA starts = waits / bytes per program | 20 / 1179648 | 33 / 122880 |
| Peak emitted SPM extent in banks 0 / 1 | 131072 / 131072 bytes | 4096 / 4096 bytes |
| Ordinary scratch | Four arrays totaling 262144 bytes | Five 4096-byte arrays, two scalar f32 values |
| Tile MMA operations per program | 32 | 0 |

Reproduce in the Triton/triton-riscv Python environment:

```bash
python examples/FPGA-BOSCAME/linalg-boscame/frontend.py --check-abi
python3 examples/FPGA-BOSCAME/linalg-boscame/simcheck_frontend.py
```

The real frontend retains load snapshots in ordinary memory. Its additional
scratch traffic is reported explicitly and is not included in the saved
buffer fixture's four-transfer GEMM result. Both paths preserve physical
K-contiguous B and add the original C once. LayerNorm keeps 0/1/0 load padding,
the 1024-lane reductions, division by 768 and the scalar rsqrt operation. Its
current centered select zeros padded X lanes before squaring, so they
contribute zero to variance.
The scalar LLVM library call is `rsqrtf`: it rounds `sqrtf(value)` and then
`1.0f/root` separately. Both ordinary and BOSCAME simulators use that emitted
runtime contract. A regression checks `7.0f`, where this result is
`0x3ec18490` and the single-round `1/sqrt(double)` result is `0x3ec1848f`.
The simulation model/oracle correction does not change generated arithmetic.

`check_board_test.py` validates the post-execution DDR checker with synthetic
data for both kernels and both ABI variants. Negative checks reject changed
input/output/guard bytes, incorrect total and user DMA counts/bytes,
equal-but-incorrect starts/waits, trap diagnostics, application time shorter
than both launches, missing/truncated DDR, and stale or incorrectly tagged
UART verdicts. These checker tests contain no board execution evidence.
Manifest tests also reject changed ELF, binary and LLVM artifacts, a wrong
ABI variant and missing compiler identity. Reports use compiler/source hashes
captured at image build rather than identifying an image with whichever
`buddy-opt` happens to be installed when the dump is checked.

## SPM-to-SPM copy lowering and historical RVV probes

After staging, the independent backend emits exact typed DMA packets for
`memref.copy` and identity-mapped parallel Linalg copies that only yield the
input. Each packet has an immediate wait. The copied bytes are exactly the
valid subviews, with no masked padding transfer or arithmetic conversion.
Arithmetic and reduction operations retain their existing lowering and
numerics. This implements the SPM-to-SPM path permitted by section 2.5 of the
patch6 and patch7 hardware handoffs.

Both endpoints must have matching shapes and element types and the same
physical contiguous axis. Short/unaligned packets, physical transposes and
any source/destination overlap across the entire copy are rejected. Disjoint
constant views in the same bank are supported. This preserves bits and does
not assume `memmove` semantics. Legacy defaults remain unchanged.

Controlled probes reproduced an incomplete byte copy with the exact loop
emitted for the frontend GEMM. A probe placing the independent scalar counter
increment after `vse8.v`, with a nop preserving the store's original
instruction address, passed the full 65536-byte 64-lane copy and signed-byte
oracle. The historical frontend GEMM image using that byte-copy workaround
has no completed numerical result. These historical measurements do not
establish a processor root cause or compatibility of every reordered variant.

The optional `restrict_fpga_assembly.py --order-vector-copies` tool remains
available, but the canonical builder no longer enables it. It accepts only
an unmasked byte load/store pair using
the same vector register, separated by a self-increment whose destination is
independent of both address registers. Register aliases are normalized.
Only comments and blank lines may otherwise separate the three instructions;
labels, directives and other instructions block the rewrite. It preserves
the original instructions and fence pairs, changing no values, byte ranges,
DMA traffic or arithmetic order. The filter default retains the original
sequence.

The separate `--order-vector-adds` compatibility option is selected by the
canonical NR builder for the measured `e32/16/m4` integer-add loop. Its strict
matcher requires two e32 loads, an in-place `vadd.vv` and store, independent
address/counter registers, 64-byte pointer steps, a 16-lane counter step and
the matching loop branch. It replaces the original counter increment with
`c.nop` and places that increment after the store's trailing fence. Vector
arithmetic, i32 wrapping, addresses and DMA counts are unchanged. The
unscheduled image and scheduling prototype are recorded separately below;
the integrated final image passed its complete board and final-DDR checks.

## Compiler regression checks

Current targeted checks pass all 34 tests in
`tests/Conversion/LowerLinalgToBOSCAME`, including the new integer-add schedule
test, and all 17 tests in `tests/Conversion/BOSCAME`.
Typed same-bank/liveness checks passed 2/2 and checker regressions passed 6/6.
The final `ninja -C build check-buddy` run completed with exit status 0 after
the integer-add matcher and its regression were added:

| Suite | Passed | Unsupported | Expected failures | Unexpected failures |
| --- | --- | --- | --- | --- |
| Examples | 239 | 2 | 0 | 0 |
| Regressions | 465 | 2 | 1 | 0 |

The regression suite discovered 468 tests. Coverage includes SPM signature
rejection, direct and transitive parallel-call rejection, preservation of
explicit caller SPM windows across staged helper calls, typed SPM DMA copies,
same-bank ranges and cross-packet overlap, both strict instruction schedule
matchers, actual O2 DMA endpoints and RVV fill code generation, explicit-window overlap
rejection, view transfers and ordinary f16 fallback.
The BOSCAME checker regressions also cover unknown SPM users, unknown memory
effects during DMA and live AME accumulators, including reused and loop-carried
values.
Changed-file pre-commit checks and `git diff --check` also passed.

## Canonical GEMM compiler and functional checks

| Field | Result |
| --- | --- |
| Logical M / N / K | 128 / 128 / 1024 |
| BM / BN / BK | 64 / 128 / 1024 |
| Grid and program-ID source | 2×1, serial caller loop in `launch.c` |
| A layout | row-major `[128,1024]`, strides `[1024,1]` |
| B layout | logical `[1024,128]`, physical `[128,1024]`, logical strides `[1,1024]` |
| C layout | row-major `[128,128]`, strides `[128,1]` |
| Kernel ABI | three ranked rank-two memrefs, flattened to 21 LLVM arguments |
| Per-call descriptor offsets | zero; caller advances A by 65536 bytes and C by 32768 bytes |
| Independent lowering / LLVM translation | passed |
| RV64 C adapter + launcher LLVM link | passed |
| Host stub ABI check | passed, including both call offsets and invalid root rejection |
| `llc` raw RISC-V assembly generation | passed |
| Static DMA starts / waits per program | 4 / 4 |
| Dynamic user DDR transfers per program | A read 65536 B, B read 131072 B, C read 32768 B, C write 32768 B |
| SPM bank 0 | B at byte offset 0, 131072 bytes |
| SPM bank 1 | A at byte offset 0, 65536 bytes; C at byte offset 65536, 32768 bytes |
| Additional AME scratch | none; 3 SPM views and 0 ordinary allocations |
| Functional integer oracle | 16384 / 16384 outputs match exactly after i32 wrapping |
| Simulator launch order | `m0=64`, then `m0=0`, with persistent poisoned SPM |
| Source numerics | signed i8 product, original nonzero C added once, i32 wrap preserved |
| DDR guards / authorized per-program ranges | passed |

Reproduce with:

```bash
python3 examples/FPGA-BOSCAME/linalg-boscame/build.py --check-abi --codegen
python3 examples/FPGA-BOSCAME/linalg-boscame/simcheck.py
```

## LayerNorm compiler and functional checks

| Field | Result |
| --- | --- |
| Rows / valid f32 lanes / logical lanes | 197 / 768 / 1024 |
| X / Weight / Bias / Out arguments | ranked `memref<768xf32>`, zero offset, 64-byte aligned pointers |
| Row slicing | caller advances X/Out by `row*3072` bytes; Weight/Bias shared |
| User-buffer DMA per row | 3 reads and 1 write, each 3072 bytes |
| SPM-to-SPM DMA per row | 4 requests, 12288 bytes |
| Total generated DMA per row | 25 starts / 25 waits, 90112 bytes |
| Ordinary scratch DMA per row | 17 requests, 65536 bytes |
| SPM peak | 8192 bytes per bank |
| Ordinary scratch | six 4096-byte buffers plus one f32 sum |
| Numerics | ordered 1024-lane f32 sums, division by 768, scalar rsqrt, padded centered lanes selected to zero |
| Functional oracle | 151296 / 151296 outputs bit-identical to ordinary buffer Linalg and independent ordered-f32 oracle |
| Guards / last row / shared Weight and Bias tails / persistent SPM | passed |
| Historical image board status | PASS on NR patch9 FPGA4, with independent live-DDR check; image used the earlier padded `0 - mean` semantics |

## Retained DMA-copy board measurements

All four retained images completed two different input rounds with SPM
poisoned once and retained across reverse-order programs. Every on-board
round and independent host check passed: GEMM checked 32768 values across
both rounds and 16384 final-DDR values exactly; LayerNorm checked 302592
values across both rounds and 151296 final-DDR values exactly. Output and
guard errors are zero, with maximum absolute error 0.0. The runner obtained
live post-execution DDR, observed the matching terminal PASS and completed
normal idle cleanup with exit status 0. The reports bind
the compiler/source hashes, LLVM input, ELF and binary to their build
manifests. The clock is 5529600 Hz; kernel and application timings use the E6
counter with the distinct scopes described in the historical record below.

At the time of those board runs, both LayerNorm variants were independently
rebuilt with the final canonical builder, including `--order-vector-adds`.
That matcher has no match in these kernels. The recorded LLVM input, ELF and
binary hashes are identical to the board-tested artifacts, but describe the
earlier padded-lane semantics. The
[frontend equivalence record](../../../docs/validation/buffer-form-boscame-board/dma-copy/frontend/layernorm/final-image-equivalence.json)
and [saved-buffer equivalence record](../../../docs/validation/buffer-form-boscame-board/dma-copy/buffer/layernorm/final-image-equivalence.json)
bind the original board report and tested manifest to the final rebuilt
manifest. The original report and manifest remain unchanged.

| Path | FPGA | Kernel E6 cycles, rounds 0 / 1 | Application-total E6 cycles | Final DDR values exact | Status |
| --- | --- | --- | --- | --- | --- |
| [Real frontend GEMM, integrated add schedule](../../../docs/validation/buffer-form-boscame-board/dma-copy/frontend/gemm/fpga4.report.json) | 4 | 1431133 / 1430833 | 285125893 | 16384 / 16384 | PASS |
| [Real frontend LayerNorm, earlier semantics](../../../docs/validation/buffer-form-boscame-board/dma-copy/frontend/layernorm/fpga6.report.json) | 6 | 43474526 / 43471018 | 152600654 | 151296 / 151296 | PASS |
| [Saved buffer GEMM](../../../docs/validation/buffer-form-boscame-board/dma-copy/buffer/gemm/fpga6.report.json) | 6 | 552127 / 552215 | 283379815 | 16384 / 16384 | PASS |
| [Saved buffer LayerNorm, earlier semantics](../../../docs/validation/buffer-form-boscame-board/dma-copy/buffer/layernorm/fpga4.report.json) | 4 | 36344165 / 36339318 | 138361646 | 151296 / 151296 | PASS |

| Path | DMA starts = waits per launch | Total bytes | User bytes | SPM-copy bytes | Ordinary scratch bytes |
| --- | --- | --- | --- | --- | --- |
| Real frontend GEMM | 40 | 2359296 | 524288 | 524288 | 1310720 |
| Real frontend LayerNorm | 6501 | 24207360 | 2412736 | 2412736 | 19365888 |
| Saved buffer GEMM | 8 | 524288 | 524288 | 0 | 0 |
| Saved buffer LayerNorm | 4925 | 17752064 | 2412736 | 2412736 | 12910592 |

The report field `scratch_dma_bytes` is total minus user traffic, so it
includes both SPM-copy and ordinary scratch bytes: 1835008 for frontend
GEMM, 21786624 for frontend LayerNorm, zero for saved-buffer GEMM and 15331328
for saved-buffer LayerNorm. Each round has the same exact counters.

Frontend GEMM takes 0.258813115 / 0.258758861 seconds per two-program kernel
launch and 51.563565719 seconds for the application scope; saved-buffer GEMM
takes 0.099849356 / 0.099865271 seconds per launch and 51.247796405 seconds
for the application scope. Frontend LayerNorm takes 7.862146629 / 7.861512225
seconds per 197-row kernel launch and 27.597051143 seconds for the application
scope; saved-buffer
LayerNorm takes 6.572657154 / 6.571780599 seconds per launch and 25.021999060
seconds for the application scope. These are individual measurements tied
to the linked images, without a statistical performance or speedup claim.

The earlier DMA-copy frontend GEMM image without the integer-add schedule
completed 16 DMA starts and waits, totaling 1048576 bytes, but did not complete its
first program or produce a terminal verdict or final-DDR readback. A live
diagnostic bounded execution after the complete AME product and sixteenth
DMA wait, before the seventeenth start returned; this interval includes the
original-C/product RVV addition. The 8192 product values and the A/B/C
snapshots checked exactly in that diagnostic. These are partial-state
observations, not a completed kernel result. The
[incomplete record](../../../docs/validation/buffer-form-boscame-board/dma-copy/frontend/gemm/unordered-add/fpga4.incomplete.json)
retains this image's identity and execution bound.

A prototype that changed only the integer-add schedule completed two rounds
with 32768 exact GEMM checks, zero guard errors, and 40 DMA starts and waits /
2359296 bytes per launch. Its independent final-DDR oracle found all 16384
final values exact. The final integrated image subsequently passed the
separate full board run and final-DDR oracle linked above. These observations
support the narrow compatibility matcher without establishing a hardware
root cause.

## Historical board measurement record

The following results come from the exact historical standalone images and
their live post-execution DDR dumps. They do not validate images rebuilt with
SPM-to-SPM DMA lowering. The historical LayerNorm images used 29 frontend or
21 saved-buffer DMA requests per row; the current lowering uses 33 and 25.
Saved-buffer GEMM's four-transfer contract is unchanged, but its measurement
still identifies the retained image. Sanitized UART, reports and build
manifests are retained in the [board evidence directory](../../../docs/validation/buffer-form-boscame-board/README.md).

The drivers record per-round kernel E6 cycles around each complete adapter
launch, including its serial reverse program loop and actual DMA wrappers.
Application-total E6 cycles cover both rounds, SPM poisoning, initialization
and independent C validation. Both use E6 `rdcycle`; they do not measure NH
cycles. Raw cycle counts must retain these distinct scopes when converted
using the reported clock frequency.

The clock is 5529600 Hz. Each image performs two different input rounds with
no warmup; rows or GEMM programs run in reverse order, and SPM persists between
rounds. Raw measurements are reported individually, without a statistical
performance claim.

| Path | FPGA | Kernel E6 cycles, rounds 0 / 1 | Application-total E6 cycles | Final DDR values exact | Status |
| --- | --- | --- | --- | --- | --- |
| Historical real frontend GEMM before DMA-copy lowering | 4 | unavailable | unavailable | unavailable | INCOMPLETE |
| Historical real frontend LayerNorm | 6 | 45132010 / 45131952 | 155919201 | 151296 / 151296 | Historical image PASS |
| Historical saved buffer GEMM | 4 | 552136 / 552205 | 283379428 | 16384 / 16384 | Historical image PASS |
| Historical saved buffer LayerNorm | 4 | 38291478 / 38291181 | 142260974 | 151296 / 151296 | Historical image PASS |

An additional historical frontend GEMM retry on FPGA4 after the board became
available loaded successfully and printed `start`, but produced no terminal verdict in
the runner's 120-second inactivity window. The runner released the board with
exit status 0; there was no final DDR readback. A separate live diagnostic of
the exact same binary found five DMA starts and five waits, totaling 425984
bytes, with no completed program or checked output. This bounds execution
after the C DDR-to-SPM wait and before the sixth DMA start returns: the interval
includes the C e32 SPM copy and the next start helper, before the first AME
instruction. Trap and DMA error records were zero. The diagnostic snapshot
does not cover the high stack scratch and does not validate results. The
[incomplete retry record](../../../docs/validation/buffer-form-boscame-board/frontend/gemm/fpga4.incomplete.json)
retains that historical image's identity and these limits; it has no completed
numerical result.

The historical saved GEMM image takes 0.099850984 / 0.099863462 seconds per
complete two-program launch, and 51.247726418 seconds for the application scope. Each launch
records eight DMA starts and waits, 524288 user bytes and zero scratch bytes.
Both on-board rounds check 32768 values exactly, with zero output/guard
errors; the independent final-DDR oracle finds all 16384 values exact.

The historical frontend LayerNorm image takes 8.161894170 / 8.161883681
seconds per complete 197-row launch, and 28.197193468 seconds for the
application scope. Each launch
records 5713 DMA starts and waits, 21786624 bytes total: 2412736 user bytes
and 19365888 scratch bytes. The on-board oracle checks all 302592 values over
both rounds with zero output/guard errors. The independent final-DDR oracle
finds zero mismatches and maximum absolute error 0.0, including the retained
padded-lane variance semantics.

The historical saved LayerNorm image takes 6.924818793 / 6.924765082 seconds
per complete 197-row launch, and 25.727172671 seconds for the application
scope. Each launch
records 4137 DMA starts and waits, 15331328 bytes total: 2412736 user bytes
and 12910592 scratch bytes. All 302592 on-board checks are exact over both
rounds with zero output/guard errors. The independent final-DDR check finds
all 151296 values exact with maximum absolute error 0.0.

The retained report binds its compiler binary and source hashes, LLVM input,
ELF and binary image to the build manifest. The runner completed live 3 MiB
DDR readback, natural idle cleanup and a zero remote process exit status.
No equivalent canonical baseline FPGA timing is available, so no speedup is
claimed.
