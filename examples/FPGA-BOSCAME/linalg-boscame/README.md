# Linalg to BOSCAME examples

This directory contains a real Triton frontend pipeline and independent saved
buffer-form inputs. Both use the canonical GEMM and padded LayerNorm shapes.
Their generated artifacts and C ABIs are explicit and separate.

## Real TTIR through the independent backend

Activate the repository's Triton/triton-riscv Python environment, then run:

```bash
python examples/FPGA-BOSCAME/linalg-boscame/frontend.py --check-abi
python3 examples/FPGA-BOSCAME/linalg-boscame/simcheck_frontend.py
```

`kernels.py` is the actual Triton source. `frontend.py` exports its TTIR and
saves every boundary under `build/examples/linalg-boscame/frontend/`, with
separate `gemm/` and `layernorm/` directories:

| Artifact | Producer and contract |
| --- | --- |
| `kernel.ttir` | Triton AST exporter; runtime program IDs remain explicit |
| `kernel.linalg.mlir` | Experimental frontend; ordinary tensor Linalg, physical pointer views and load snapshots, plus backend configuration/provenance |
| `kernel.bufferized.mlir` | Separate one-shot bufferization with 64-byte alignment |
| `kernel.boscame.mlir` | Independent `--linalg-to-boscame` backend |
| `kernel.llvm.mlir`, `kernel.ll` | `--lower-bosc-ame`, followed by LLVM translation |
| `contract.json` | Source shapes, grid, masks and full-root ABI |

The frontend option `boscame=true` preserves configuration and provenance; it
does not create SPM windows or invoke the independent backend. `tt.load` values
are materialized in ordinary memory, including the original C snapshot. B
retains logical `[K,N]` indexing with physical strides `[1,1024]`; no transpose
or alternate B packing is introduced. Bufferization is an explicit command
between frontend conversion and the independent pass. Use `--stop-after` to
inspect `ttir`, `linalg`, `buffer`, `boscame` or `llvm`; `--kernel` selects
`gemm`, `layernorm` or `all`.

The generated frontend functions take typed unranked memrefs followed by six
i32 arguments: grid X/Y/Z and program ID X/Y/Z. `frontend_adapter.c` constructs
rank-one descriptors for complete root buffers and calls the actual emitted
`_mlir_ciface_triton_gemm` or `_mlir_ciface_triton_layernorm` wrappers. Its public
entries are `boscame_frontend_gemm_launch` and
`boscame_frontend_layernorm_launch`, declared in `frontend_abi.h`. They validate
64-byte alignment, address overflow and disjoint full ranges, then run the
explicit grids `(2,1,1)` and `(197,1,1)` serially in reverse program order.
`--check-abi` compares emitted LLVM wrapper types with the RV64 adapter,
LLVM-links them, and runs a host stub checking all descriptors, program IDs
and invalid-root returns. It does not execute AME on the host.

The frontend functional check covers every output: all 16384 GEMM i32 values
match independent int64 products with explicit i32 wrapping and nonzero initial
C; all 151296 LayerNorm values are bit-identical to the padded ordered-f32
oracle. Programs run in reverse order with persistent poisoned SPM. Input,
output and excluded-region guards and exact user DDR access ranges are checked;
`--quick` limits LayerNorm to rows 196, 0 and 1.

The recorded frontend LayerNorm image with SPM-to-SPM DMA predates the
select-zero correction: its padded centered lanes were `0 - mean`, contributing
`mean²` each to variance. Its board pass validates that earlier numerics, not
the current frontend source. That image passed two complete reverse-order
launches on NR patch9 FPGA6. Live post-execution DDR readback
contains all 151296 final values bit-identical to the ordered-f32 oracle;
both on-board rounds checked 302592 values with zero output or guard errors.
Each launch records 6501 DMA starts and waits and 24207360 bytes. Kernel E6
cycles are 43474526 / 43471018; application-total E6 cycles are 152600654.
The [historical report](../../../docs/validation/buffer-form-boscame-board/dma-copy/frontend/layernorm/fpga6.report.json)
and [validation record](validation.md) retain the exact image identity and
timing scopes. Older image measurements remain explicitly historical.
The [final-image equivalence record](../../../docs/validation/buffer-form-boscame-board/dma-copy/frontend/layernorm/final-image-equivalence.json)
confirms that the historical source regenerated identical LLVM input, ELF and
binary; it does not bind the later select-zero source. The integer-add matcher
does not apply to this kernel.

Both LLVM paths call the scalar `rsqrtf` runtime entry, which computes
`1.0f / sqrtf(value)`. The simulator and oracle round the square root and
division separately to f32, matching the ordinary compiled path. The
single-round result of `1/sqrt(double)` can differ and is not the contract.

The independent backend lowers pure SPM-to-SPM copies to exact typed DMA
packets after staging, waiting immediately for each packet. This follows
section 2.5 of the patch6 and patch7 hardware handoffs. The copies preserve
bits and transfer only the valid subviews; initialized masked padding is not
copied. Arithmetic and reductions keep their existing computation paths.
The canonical builder no longer uses `restrict_fpga_assembly.py
--order-vector-copies`. That optional sequence workaround and its historical
probe results remain available. The filter default and legacy backend
defaults are unchanged.

The canonical NR builder selects the separate `--order-vector-adds` option
for the narrowly matched `e32/16/m4` two-load, in-place integer add loop.
It moves the independent 16-lane counter increment after the store's trailing
fence, retaining `c.nop` in its original position, the vector arithmetic,
pointer steps and DMA traffic. The final integrated frontend GEMM image passed
on NR patch9 FPGA4: both rounds checked all 32768 GEMM values exactly with
zero guard errors and 40 DMA requests / 2359296 bytes per launch. Independent
final-DDR readback confirms all 16384 final values exactly. Kernel E6 cycles
are 1431133 / 1430833; application-total E6 cycles are 285125893. The
[current report](../../../docs/validation/buffer-form-boscame-board/dma-copy/frontend/gemm/fpga4.report.json)
binds these results to the final image. The earlier incomplete image and the
successful compatibility result do not establish a hardware root cause.

Materializing TTIR load values introduces ordinary scratch arrays and DMA
traffic beyond the four user-buffer transfers. Counts below are per program
or per LayerNorm row in the generated BOSCAME IR:

| Path | User requests / bytes | SPM-to-SPM requests / bytes | Ordinary scratch requests / bytes | Total starts = waits / bytes | Peak SPM per bank | Ordinary scratch |
| --- | --- | --- | --- | --- | --- | --- |
| Real frontend GEMM | 4 / 262144 | 4 / 262144 | 12 / 655360 | 20 / 1179648 | 131072 / 131072 bytes | 262144 bytes in four aligned arrays |
| Real frontend LayerNorm | 4 / 12288 | 4 / 12288 | 25 / 98304 | 33 / 122880 | 4096 / 4096 bytes | Five 4096-byte arrays and two scalar f32 values |
| Saved buffer GEMM | 4 / 262144 | 0 / 0 | 0 / 0 | 4 / 262144 | 131072 / 98304 bytes | None |
| Saved buffer LayerNorm | 4 / 12288 | 4 / 12288 | 17 / 65536 | 25 / 90112 | 8192 / 8192 bytes | Six 4096-byte arrays and one scalar f32 value |

Both GEMM paths emit 32 tile MMA operations per program. These are artifact
and functional-model facts; board measurements are recorded separately in
`validation.md`.

## Supported and rejected forms

| Form at the independent buffer boundary | Behavior |
| --- | --- |
| Static rank-one/rank-two physically rectangular memrefs with proven 64-byte packet alignment | Typed exact-range DMA staging |
| Explicit filled padding and valid-region subviews | Preserved; excluded bytes are not transferred |
| Pure SPM-to-SPM copies, including disjoint constant views in the same bank | Exact typed DMA packets with immediate waits; preserves bits |
| Marked signed i8 × i8 → i32 matmul with K-contiguous physical B | AME tiles, original C once, i32 wrapping |
| Unmarked matmul, including floating point | Ordinary Linalg loop lowering |
| Serial loops/branches and serial calls | Shared fixed SPM windows with checked lifetimes |
| Tensor SSA or unresolved conversion casts | Rejected; bufferize before invoking this pass |
| Dynamic extents, negative/unknown strides, overlapping physical rows, no unit-stride axis, unsupported masks or unaligned packets | Rejected with a diagnostic; use an explicit ordinary-memory fallback before this boundary |
| Marked AME with row-major logical `[K,N]` B or unsupported element types | Rejected; no hidden transpose or numerical conversion |
| SPM copies with short/unaligned packets, differing physical contiguous axes, or any source/destination overlap across the whole copy | Rejected; no physical transpose or `memmove` semantics |
| SPM in function signatures, scalar SPM loads/stores, concurrent or unknown execution contexts | Rejected |
| Live windows exceeding either 128 KiB bank, conflicting configuration or ABI promises, incomplete DMA/AME lowering | Rejected |

See [the backend architecture](../../../docs/BufferFormBOSCAME.md) for the exact
layout, lifetime and configuration rules.

## Saved buffer-form GEMM

`gemm.bufferized.mlir` is a saved input for the independent
`--linalg-to-boscame` pass. It contains ranked memrefs and a marked
signed i8 × i8 → i32 `linalg.matmul`; it requires no Triton frontend or
bufferization inside this pass.

See [the backend architecture](../../../docs/BufferFormBOSCAME.md) for the pass
boundary, supported buffer accesses and workspace rules.

The complete operation is `C = C_initial + A × B`, with signed i8 inputs and
i32 wrapping accumulation. The original, potentially nonzero C is added once.

| Buffer | Logical shape | Physical shape | Strides in elements |
| --- | --- | --- | --- |
| A | `[128,1024]` | `[128,1024]` | `[1024,1]` |
| B | `[1024,128]` | `[128,1024]` | `[1,1024]` logically |
| C | `[128,128]` | `[128,128]` | `[128,1]` |

B has K-contiguous columns: logical `B[k,n]` is stored at physical
`B_physical[n*1024+k]`. The input already provides this layout. A conventional
row-major logical `[K,N]` buffer is a different contract; the example does not
request CPU packing.

## Program and caller contract

`BM=64`, `BN=128`, `BK=1024`, with grid **2×1**. One call updates a complete
64×128 C slab. Its two internal 64-column AME groups have `n0=0` and `n0=64`.
This explicitly fuses N at the program boundary; a backend must not extend a
64-column program into an adjacent program's C region.

The saved function is a per-program entry. The caller makes two sequential
calls using the following aligned pointers and zero descriptor offsets:

| Argument | Program 0 (`m0=0`) | Program 1 (`m0=64`) | Ranked descriptor shape |
| --- | --- | --- | --- |
| A | A root | A root + 65536 bytes | `[64,1024]` |
| B | B root | B root | `[1024,128]`, strides `[1,1024]` |
| C | C root | C root + 32768 bytes | `[64,128]` |

The ABI is ranked memref descriptors, not unranked memrefs or bare pointers.
The caller owns program IDs and pointer slicing; the compiler does not infer a
grid from a runtime ID. Distinct A/B/C roots must not overlap, and their aligned
pointers must be 64-byte aligned. Each C slab must contain its original values
before entry.

`abi.h` defines the dimensions, grid, typed rank-two descriptors and status
codes, with static assertions for the canonical byte counts and ABI layout.
`adapter.c` builds the three per-program descriptors with zero offsets and
flattens them into the emitted `gemm_program` function's 21 arguments: allocated
pointer, aligned pointer, offset, two sizes and two strides for each memref.
All metadata is i64. There is no `_mlir_ciface` wrapper or runtime program-ID
argument. `launch.c` validates the full root ranges, then calls the adapter
twice with the explicit slices above. It preserves C's initial values.

The public entry is `boscame_gemm_launch(a, b_physical, c)`. It rejects null,
unaligned or overlapping buffers before calling the kernel; allocation
capacity remains the caller's responsibility. Calls use fixed SPM addresses
and execute serially. Platform setup and the DMA runtime must be initialized
by the integrating application.

## Lowering

Run from the repository root after building the Buddy tools:

```bash
python3 examples/FPGA-BOSCAME/linalg-boscame/build.py --kernel all --check-abi --codegen
python3 examples/FPGA-BOSCAME/linalg-boscame/simcheck.py
```

The build script saves each selected kernel's `.bufferized.mlir`,
`.boscame.mlir`, `.llvm.mlir`, `.ll` and `.contract.json`, plus the selected
`contract.json`, under
`build/examples/linalg-boscame/`. It invokes the `--linalg-to-boscame` alias
with the explicit options below. `--check-abi` checks the flattened LLVM
signature, compiles `adapter.c` and `launch.c` to RV64 LLVM IR, and links them
with the kernel into `gemm.with-launch.bc`. It also runs `abi_check.c` on the
host with a kernel stub to check descriptor fields, both calls and ABI error
returns. That host check does not execute AME. `--codegen` additionally creates
raw RISC-V `gemm.s` with `llc`; platform encoding, runtime linking and a board
image are separate integration steps. Omit the options to produce only IR.
Tool overrides are available through `--buddy-opt`, `--buddy-translate`,
`--llc`, `--clang`, `--llvm-link` and `--host-cc`; `--out-dir` changes the output
directory. Defaults resolve tools from the repository build trees or PATH.

The equivalent manual pipeline uses the retained
`--lower-linalg-to-boscame` spelling with the same opt-in configuration:

```bash
mkdir -p build/examples/linalg-boscame
build/bin/buddy-opt \
  examples/FPGA-BOSCAME/linalg-boscame/gemm.bufferized.mlir \
  --lower-linalg-to-boscame="enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga bm=64 bn=128 bk=1024" \
  -o build/examples/linalg-boscame/gemm.boscame.mlir
build/bin/buddy-opt build/examples/linalg-boscame/gemm.boscame.mlir \
  --lower-bosc-ame \
  -o build/examples/linalg-boscame/gemm.llvm.mlir
build/bin/buddy-translate build/examples/linalg-boscame/gemm.llvm.mlir \
  --buddy-to-llvmir \
  -o build/examples/linalg-boscame/gemm.ll
```

The first command is the independent backend boundary. The default
experimental frontend and the legacy `--triton-to-boscame` entry are separate
pipelines.

After producing `gemm.boscame.mlir`, run the functional check (requires NumPy):

```bash
python3 examples/FPGA-BOSCAME/linalg-boscame/simcheck.py
```

It checks all 16384 output elements against an independent int64 product
followed by i32 wrapping. Both programs run in reverse order with persistent
poisoned SPM. Inputs include signed i8 extrema and nonzero C near i32 limits;
guards and access logs check that each program touches only its authorized
A/C slab and the shared B buffer. This is a functional model check.

## Emitted transfers and workspace

For this canonical input, the generated BOSCAME IR contains four DMA starts
and four matching waits per program:

| Direction | Buffer | DDR byte range | Bytes |
| --- | --- | --- | --- |
| DDR → SPM | B | complete physical `[128,1024]` | 131072 |
| DDR → SPM | A | `A + m0*1024` elements | 65536 |
| DDR → SPM | C | `C + m0*128` elements | 32768 |
| SPM → DDR | C | same C slab | 32768 |

The emitted workspace places B in bank 0 at offset 0 and A in bank 1 at offset
0, followed by C in bank 1 at byte offset 65536. Thus bank 0 uses 131072 bytes
and bank 1 uses 98304 bytes. These
sizes fit the two 128 KiB banks and 64-byte alignment defined in
`midend/include/Dialect/BOSCAME/SpmDma.h`. Tile computation must share this same
budget. The generated IR has three `bosc_ame.spm.view` operations and no
`memref.alloc` operations or additional AME scratch windows. It has 32
`bosc_ame.tile.mma` operations: two N groups, each with sixteen 64-wide K
steps.

The functional check passes with four complete DDR transfers for each of the
two programs, all 16384 i32 outputs matching the oracle, and guards intact.
The final saved GEMM image passed on NR patch9 FPGA6: both rounds checked
32768 outputs exactly, and live DDR readback independently confirms all 16384
final outputs. Each two-program launch records eight DMA starts and waits,
524288 bytes and no scratch traffic. Kernel E6 cycles are 552127 / 552215;
application-total E6 cycles are 283379815, including both rounds and the
independent on-board oracle. The
[current report](../../../docs/validation/buffer-form-boscame-board/dma-copy/buffer/gemm/fpga6.report.json)
binds the result to the final image. See [validation.md](validation.md) for
timing scopes and the separately retained historical measurements.

## LayerNorm buffer-form input

`layernorm.bufferized.mlir` is a separate per-row input with ranked
`memref<768xf32>` X, Weight, Bias and Out arguments. Each descriptor has zero
offset and a 64-byte aligned pointer. The caller advances X/Out by
`row*3072` bytes for each of 197 rows; Weight and Bias are shared. The logical
computation keeps 1024 lanes, with padded X/Bias equal to 0 and padded Weight
equal to 1. Both ordered f32 sums cover all 1024 lanes and divide by 768. The
centered select sets padded X lanes to zero before squaring, so they contribute
zero to variance. The scalar rsqrt follows the source operation order.

`layernorm_abi.h` and `layernorm_adapter.c` expose the ranked per-row entry
`boscame_layernorm_run_program`. It validates four disjoint 64-byte-aligned
768-element ranges and builds four flattened rank-one memrefs, for 20 LLVM
arguments. The board caller supplies the explicit X/Out row slices; Weight
and Bias are shared. `build.py --kernel layernorm --check-abi --codegen` verifies
this signature, RV64 link and host descriptor/error-return checks.

```bash
build/bin/buddy-opt \
  examples/FPGA-BOSCAME/linalg-boscame/layernorm.bufferized.mlir \
  --linalg-to-boscame="enable-spm-dma=true dma-abi=args_aligned64_nonalias" \
  -o build/examples/linalg-boscame/layernorm.boscame.mlir
build/bin/buddy-opt build/examples/linalg-boscame/layernorm.boscame.mlir \
  --lower-bosc-ame \
  -o build/examples/linalg-boscame/layernorm.llvm.mlir
python3 examples/FPGA-BOSCAME/linalg-boscame/simcheck_layernorm.py
```

The checker compares the ordinary buffer Linalg execution and the BOSCAME
execution with an independent sequential-f32 oracle; `--quick` runs three rows.
The complete 197-row simulation passes: all 151296 outputs are bit-identical
to both references, with persistent SPM, the last row and DDR guards checked.
The four user-buffer transfers are three 3072-byte reads and one 3072-byte
write. The current lowering adds four SPM-to-SPM requests totaling 12288
bytes, and 17 ordinary scratch requests totaling 65536 bytes: 25 requests
and 90112 bytes per row in total. It uses 8192 SPM bytes per bank, six ordinary
4096-byte scratch buffers and one ordinary f32 sum. Copies use exact valid
subviews and leave the masked padding initialized by the fills.

The saved LayerNorm image with SPM-to-SPM DMA passed two full launches on NR
patch9 FPGA4 before the select-zero correction. That image used `0 - mean` for
padded centered lanes, so its board result validates the earlier `mean²`
variance contribution, not the current input semantics. All 302592 on-board
checks in that historical run were exact with zero output/guard errors, and
live DDR readback independently confirms all 151296 final values.
Kernel E6 cycles are 36344165 / 36339318, with application-total E6 cycles
138361646. Each launch records 4925 DMA starts and waits and 17752064 bytes:
2412736 user bytes, 2412736 SPM-copy bytes and 12910592 ordinary scratch
bytes. The [historical report](../../../docs/validation/buffer-form-boscame-board/dma-copy/buffer/layernorm/fpga4.report.json)
binds the result to its exact image. See [validation.md](validation.md) for
timing scopes and the separately retained historical measurements.
The [final-image equivalence record](../../../docs/validation/buffer-form-boscame-board/dma-copy/buffer/layernorm/final-image-equivalence.json)
confirms identical LLVM input, ELF and binary after rebuilding the historical
source with the final canonical builder; it does not bind the later
select-zero input. The integer-add matcher has no match in this kernel.

See [validation.md](validation.md) for compiler checks and board results for
each input boundary.

## Board result checking

Build complete NH/E6 images for both input boundaries:

Use the Triton Python environment for frontend builds, or select its interpreter
with the make variable `PYTHON`.

```bash
make -C examples/FPGA-BOSCAME/linalg-boscame bin KERNEL=gemm
make -C examples/FPGA-BOSCAME/linalg-boscame bin KERNEL=layernorm
make -C examples/FPGA-BOSCAME/linalg-boscame bin FRONTEND=1 KERNEL=gemm
make -C examples/FPGA-BOSCAME/linalg-boscame bin FRONTEND=1 KERNEL=layernorm
```

Images, ELF audits, exact LLVM inputs and build manifests are under
`build/examples/linalg-boscame/board/{buffer,frontend}/{gemm,layernorm}/`.
The linker supplies a 1 MiB E6 stack for the frontend's explicit ordinary
snapshots. The drivers poison SPM once, execute two different input rounds
without clearing it, launch programs in reverse order, check every result and
guard, and count actual runtime DMA requests separately for user and scratch
memory. The E6 `rdcycle` counter measures each complete kernel launch, including
the adapter, reverse program loop and DMA wrappers. Application-total E6 cycles
also include SPM poisoning, input initialization and the independent C oracle
for both rounds. These are two scopes of the E6 counter; no separate NH cycle
measurement is reported.

Set `BOSCAME_FPGA_SSH_TARGET` and `BOSCAME_FPGA_WORKDIR` to the existing SSH
target and directory containing `run.sh`, then run an image:

```bash
python3 examples/FPGA-BOSCAME/linalg-boscame/run_board.py gemm \
  --variant frontend --fpga 4 \
  --image-dir build/examples/linalg-boscame/board/frontend/gemm \
  --host "$BOSCAME_FPGA_SSH_TARGET" --remote-root "$BOSCAME_FPGA_WORKDIR"
```

The runner verifies the uploaded binary, waits for the matching terminal
verdict and diagnostics, then issues a live `readmem` for the final 3 MiB of
DDR before releasing the session. It downloads that dump and runs the host
checker. Loader-time readback cannot substitute for this final dump. Raw
transport logs stay in a temporary directory; saved UART evidence contains
application lines only. A failed verdict, incomplete readback or image change
during execution makes the command fail.
The remote runner's inactivity window defaults to 60 seconds and can be set
with `--idle-seconds`; choose a window longer than the longest silent kernel
or oracle interval. After readback, the runner waits for natural idle cleanup
and requires a zero process status.

`check_board.py` requires the exact ELF, post-execution DDR dump, UART capture,
kernel and ABI variant. Its default build manifest is the ELF's sibling
`build-manifest.json`; `--manifest` overrides that path. It verifies the ELF,
sibling `<kernel>.bin` and saved `kernel.ll` hashes against the manifest and
reports the build-time compiler and source hashes. Relinking Buddy later
does not reassign a compiler identity to an older board image.

The checker requires every output to match the independent host oracle bit
for bit, both rounds' exact counters and guards, matching per-variant DMA
starts/waits/bytes, positive kernel cycles, application cycles covering both
launches, zero trap/DMA errors and one correctly tagged terminal UART PASS.
`python3 examples/FPGA-BOSCAME/linalg-boscame/check_board_test.py` checks that
corrupted DDR, counts, manifests, artifacts and terminal logs fail without
contacting a board.
