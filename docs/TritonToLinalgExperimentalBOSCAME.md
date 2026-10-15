# Experimental Triton lowering to BOSCAME

The experimental driver can stage independent loads and stores through DMA,
including kernels without a dot, and lower signed i8 × i8 → i32 dots to AME.
The existing `--triton-to-boscame` entry remains available.

```bash
export PATH="$PWD/build/bin:$PATH"
buddy-opt kernel.ttir \
  --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias" \
  -o kernel.boscame.mlir
buddy-opt kernel.boscame.mlir --lower-bosc-ame -o kernel.llvm.mlir
buddy-translate kernel.llvm.mlir --buddy-to-llvmir -o kernel.ll
```

`boscame=false` is the default. It retains the existing tensor output and pass
sequence, without internal bufferization or BOSCAME operations. Both required
examples' explicit-false output was compared byte for byte with output saved
before this implementation.

## Contract and supported accesses

BOSCAME mode requires the CLI contract `dma-abi=args_aligned64_nonalias` or
`bosc_ame.dma.abi = "args_aligned64_nonalias"` on the module. Conflicting or
invalid declarations fail. The caller promises that distinct DDR arguments do
not overlap and their aligned pointers are 64-byte aligned. This promise does
not prove an arbitrary descriptor offset: its alignment is checked separately.
The example adapters also check alignment, nonalias ranges and physical strides.

DMA uses the existing `bosc_ame.dma.start` and `bosc_ame.dma.wait` operations.
Each endpoint is a static, rank-one, unit-stride memref of
`i8/i16/i32/i64/f16/bf16/f32/f64`. At least one endpoint is SPM. Byte capacity
is element count times element width. Effective address is
`alignedPointer + descriptorOffset * elementBytes`. A dynamic offset must have
an SSA divisibility proof; no alignment assumptions are inserted.

A `subview` adds offsets times the **source** strides. A `reinterpret_cast`
replaces the descriptor offset, while retaining the aligned pointer. Typed
ranked and unranked DDR roots are traced. Unknown DDR intervals conservatively
alias intervals of the same root. Distinct function arguments are disjoint
only under the explicit ABI; fresh allocations are distinct storage. SPM
intervals must remain statically bounded inside their bank.

The first implementation supports:

- Rank-one contiguous unmasked accesses and static prefix masks whose valid
  byte count is a positive multiple of 64.
- Static rank-two rectangles with a unit-stride physical axis and positive,
  nonoverlapping static strides. DMA operates on rows/columns of that physical
  axis, preserving logical tensor indexing, including physical `[N,K]` B.
- Rectangular masks recognized by TritonToStructured. A dynamic extent along
  the contiguous axis must prove 64-byte granularity; complete packets are
  guarded by the original valid extents. Zero valid extents issue no DMA.
- Static signed i8 × i8 → i32 dots, including nonzero original accumulators,
  K blocking and AME padding. Floating dots remain ordinary Linalg matmuls.

Unsupported gathers/scatters, split/wrapping pointers, scalar or atomic Triton
memory accesses, short or unaligned DMA ranges, unprovable masks/offsets,
unsupported integer dots and exhausted SPM budgets fail with a nonzero exit
status. Store lengths are never rounded up to touch excluded user bytes.

## Pipeline and memory ownership

The driver validates the contract, runs TritonToStructured and cleanup,
TritonToUnstructured, TritonArithToLinalg, StructuredToMemref,
UnstructuredToMemref, the common pointer/function conversions and cast cleanup.
Tensor-first preconversion fusion is disabled in BOSCAME mode because it can
consume loads/stores before DMA staging sees them.

Loads prepare a bounded SPM workspace covering the logical shape, initialize
masked lanes with the original `other` using one-dimensional vector stores,
DMA only the authorized DDR region, and wait. A real
`bufferization.to_tensor` exposes that SPM buffer to the remaining tensor
computation. There is no unconditional SPM-to-ordinary-buffer load copy.
Structured conversion resolves loads before stores so store buffer inference
can retain the producer's actual memory space and layout. Stores use a
read-only `bufferization.to_buffer`: compatible SPM results DMA directly to
the authorized DDR region; ordinary results explicitly stage through vectors.

After standard empty-tensor and one-shot bufferization, the driver plans static
local computation buffers in the shared SPM workspace. Non-escaping static
rank-one/rank-two allocations connected to SPM computations are eligible;
rank-zero reduction accumulators stay ordinary. SPM fill, elementwise Linalg
generic, copy and reduction operations are then lowered internally, before the
strict SPM exit checker. Elementwise operations use one-dimensional 64-byte
vectors, static matching shapes and identity maps with a contiguous final
axis. Supported computation elements are f32 and native integers; arithmetic
that would require unsupported runtime calls is rejected. Short vector tails,
unsupported layouts and unsupported SPM consumers fail explicitly. This does
not weaken the separate typed DMA element support.

The f32 rank-one sum keeps its scalar accumulator in `scf.for` iter_args.
Each iteration reads SPM with `vector.load` of `vector<1xf32>`, extracts lane
zero and clones the original scalar `arith.addf`, without reassociation.
This preserves one f32 rounding per addition and the original operand order;
it deliberately does not use `vector.reduction`.
The tested FPGA image's reduction-instruction results matched a model that
rounds once per 16-lane packet instead of once per lane, so the instruction
mnemonic alone is insufficient evidence of numerical equivalence. Extracting
all lanes from a wider vector generated unsupported slide instructions, so the
current sum uses single-lane vector reads. Other vector transfers remain
64-byte blocks. No scalar SPM loads, ordinary accumulator accesses or policy
calls occur inside the reduction loop. Scalar rsqrt remains between the
vector loops. This sum prioritizes source rounding semantics; its small vector
reads can cost more than a hardware reduction.

Integer dots first become standard tensor `linalg.matmul` operations carrying
the unit attribute `bosc_ame.lower_to_ame`. After bufferization, the driver
verifies the markers survived and calls a C++ helper on the buffer form.
Compatible SPM A/B buffers go directly to `tile.load_a/b`; the required example
uses A strides `[64,1]` and logical B strides `[1,64]`. The B workspace already
has the AME physical `[N,K]` layout and needs no second pack or DMA. Ordinary
inputs, larger K blocks and incompatible SPM layouts are materialized/packed
by the matmul consumer: SPM snapshots use complete one-dimensional vectors,
never scalar SPM accesses. Every snapshot range must fit the real `spm.view`.
C is initialized and materialized at each dot boundary. K blocks remain 64
wide; padding and MatmulConverter's separate original-C addition are preserved.
Unmarked floating matmuls keep their standard Linalg arithmetic. Their consumer
explicitly snapshots any SPM operands into ordinary buffers with bounded
one-dimensional vector copies, then copies an SPM result back after the matmul.
This keeps the subsequent standard scalar loop lowering away from SPM without
converting floating dots to integer AME.

All DMA/cache-policy calls finish outside AME register live ranges. Final
lowering inserts the existing DDR read/write policies at the actual ordinary
CPU accesses. A write policy for an SPM-to-DDR vector copy precedes its adjacent
SPM vector load, keeping the RVV value out of a runtime-call live range.

Every static access point has a non-reusing bank/offset allocation, aligned to
64 bytes. Memory and matmul helpers share the two-bank budget. K blocks may
reuse their own windows after tile store; access points do not share windows.
There are no SPM allocations: storage is expressed by `bosc_ame.spm.view`.
Static ordinary temporary buffers are promoted to stack when their lifetime
allows it. On the NH/E6 image, user DDR occupies PMP entry 5 at
`0x80100000..0x802e0000`; the E6 stack occupies
`0x802e0000..0x802f0000` in entry 6. Entry 6 retains the `0x0f` write-through
policy during both DDR read and write phases. Internal stack packs are DMA
sources, while results return through explicit one-dimensional vector copies.
They do not require the stack to share user DDR's alternating cache policy.

Keeping this boundary matters for compiler-generated spills. With the stack
inside entry 5, a diagnostic board run reached the second program and trapped
while dereferencing an A descriptor at address 9. The descriptor's stack slot
had held the boolean value 1 late in the previous program. The next entry
stored its descriptor under the policy left by `dma_wait`, then switched back
to cached loads and observed the old value 1. Separating the stack into entry
6 avoids changing its cache policy between those accesses; linker assertions
keep user data and guards below that boundary.

The exit check rejects residual Triton/tensors/bridges/casts/target matmuls,
SPM allocation/free, scalar SPM access, ordinary memref copies or Linalg on
SPM, and unmatched/unsafe DMA tokens. `lower-bosc-ame` owns standard-to-LLVM
conversion; the experimental driver does not duplicate it.

## Required examples and numerics

```bash
make -C examples/FPGA-BOSCAME/layernorm_197x768 llvm adapter asm simcheck
make -C examples/FPGA-BOSCAME/matmul_i8_128x1024x128 check simcheck
make -C examples/FPGA-BOSCAME/layernorm_197x768 check bin
make -C examples/FPGA-BOSCAME/matmul_i8_128x1024x128 boardcheck
```

These commands use the real Triton frontend (Triton/triton-riscv Python
environment), the in-tree Buddy/LLVM tools, and NumPy for the functional check.
Each example also accepts tool overrides in its makefile.

LayerNorm keeps a logical tensor of 1024 f32 lanes. Each of X, Weight and Bias
reads exactly 768 f32 = **3072 bytes**; Out writes exactly 3072 bytes. X/Out row
offsets are `programId * 768` elements = `programId * 3072` bytes. Invalid X,
Weight and Bias lanes are respectively 0, 1 and 0. Both reductions still use
all 1024 logical lanes: the centered padded X lanes contribute `mean²` to
variance. This is the original TTIR computation, including its division by 768.

The functional tests execute all 197 LayerNorm programs in an interleaved order,
with persistent poisoned SPM and DDR sentinels after the last row and after
Weight/Bias. They compare exactly with the unchanged experimental buffer IR
and an independent sequential-f32 padded TTIR oracle. Maximum absolute error
against a high-precision padded formula on the deterministic input was
2.61e-5. No reduction order or source arithmetic was changed. This functional
comparison does not assert bit identity for actual board rsqrt/FMA instructions.
The freestanding LayerNorm image links `runtime/boscame/nr/nr_math.c` for
`sqrtf` and `rsqrtf`; the latter evaluates `1.0f / sqrtf(value)` with separate
f32 rounding. The kernel and oracle builds disable floating-point contraction.

GEMM executes all 512 programs and compares all 131072 i32 outputs exactly with
an independent wide-integer NumPy product followed by i32 wrap. Signed extrema,
asymmetric physical B, nonzero previous C, outer K accumulation, persistent
SPM and DDR guards are checked. A separate 64×64×256 test checks four internal
K blocks, a nonzero dot initializer and dynamic rectangular masks/nonzero
`other`; excluded DDR addresses are inspected in the access log.

LayerNorm computes directly in SPM, including its 1024-lane padded values;
compatible GEMM A/B loads also remain in SPM until AME consumes them. General
matmul packing and C materialization still use ordinary temporary buffers and
extra DMA/vector copies. Results are materialized at every dot boundary rather
than kept in accumulator registers across the outer K loop. No baseline latency
or speedup comparison has been made; the board drivers record cycles for
diagnostics. An ELF containing RVV/AME instructions establishes
code generation; numerical board validation requires execution and final-DDR
comparison as described below.

## Board execution and post-execution readback

The direct-SPM revision passed actual FPGA execution and independent final-DDR
comparison on 2026-10-15: LayerNorm on FPGA6 (NR patch9) and GEMM on FPGA4
(NR patch7). All final-round values matched their independent oracles exactly,
with unchanged inputs and guards. See the [current direct-SPM validation](validation/experimental-boscame-direct-spm/README.md)
for image hashes, numerical details, remaining movement and source binding.
The earlier reports below describe the previous ordinary-buffer staging path.

`make check bin` for LayerNorm builds the NH/E6 launcher, typed adapter,
freestanding math, DMA runtime and audited kernel into
`build/layernorm_197x768/layernorm_197x768.bin`. The launcher runs two different
inputs, each with all 197 programs in reverse row order: 394 programs and
302592 checked output values. SPM is poisoned once before the first launch
and then retained. Every input/output has before/after guards, including the
last row and the Weight/Bias tails; inputs are checked for modification.

GEMM's `make boardcheck` produces
`build/matmul_i8_128x1024x128/boardcheck/matmul_i8_128x1024x128.boardcheck.bin`.
It executes two complete 512-program launches and checks all 262144 output
values. The operands include signed-i8 extremes and asymmetric physical B;
initial C values exercise i32 wrap. The independent C oracle uses unsigned
wrap arithmetic, and checks inputs and all DDR guards after each round.
The second launch reuses SPM without clearing it. `make debug` remains a
shorter diagnostic variant and is not this full board check.

Upload the desired image with the platform's usual transfer command, then run
it from the FPGA test runner directory:

```bash
./run.sh test=test/layernorm_197x768.bin
./run.sh test=test/matmul_i8_128x1024x128.boardcheck.bin
```

Keep the runner active until the application prints its terminal verdict;
increase its documented idle delay if a quiet computation would otherwise end
the session. After that verdict, read back DDR starting at `0x80000000` through
the output and diagnostic symbols. A 3 MiB dump covers both supplied layouts.
The runner's loader mirror is captured before program execution and cannot
validate output. Use the exact ELF linked into the uploaded image:

```bash
python3 examples/FPGA-BOSCAME/tools/check_experimental_board.py layernorm \
  --elf examples/FPGA-BOSCAME/layernorm_197x768/build/layernorm_197x768/layernorm_197x768.elf \
  --dump layernorm.final-ddr.bin --report layernorm.board.json
python3 examples/FPGA-BOSCAME/tools/check_experimental_board.py gemm \
  --elf examples/FPGA-BOSCAME/matmul_i8_128x1024x128/build/matmul_i8_128x1024x128/boardcheck/matmul_i8_128x1024x128.boardcheck.elf \
  --dump gemm.final-ddr.bin --report gemm.board.json
```

The checker resolves addresses and sizes from ELF symbols, validates both
rounds' diagnostic counters, verifies final inputs and guards, and independently
recomputes every final-round result. GEMM must match every i32 bit. LayerNorm
retains the 1024-lane padded TTIR variance and uses the f32 tolerance
`2e-5 * (1 + abs(expected))`; the report also records exact matches and maximum
absolute error. Missing symbols, incomplete dumps, invalid counters, changed
guards/inputs or numerical errors return nonzero. The report includes ELF and
dump SHA256 values.

On 2026-10-14, both examples passed actual NH/E6 execution and independent
post-execution DDR comparison. LayerNorm on FPGA5 and FPGA6 each checked
302592 values across two rounds, with zero errors and guard errors; all 151296 final-round values
matched the sequential-f32 oracle bit for bit (max absolute error 0 for these
inputs, 2.993e-5 against the f64 padded formula). GEMM on FPGA4 checked 262144
values across two rounds, with zero errors and guard/input errors; all 131072
final-round i32 values matched the independent wide-integer/wrap oracle
exactly. See [the retained board reports, UART verdicts and image hashes](validation/experimental-boscame/README.md).

## Implementation and tests

| Files | Responsibility |
| --- | --- |
| `TritonLoweringConfig.h`, experimental `Passes.td` and driver | Options, contract, common pipeline, internal bufferization, marker preservation, exit checks |
| TritonArithToLinalg headers/source | Configuration, marked standard matmul, function attribute preservation, rejection of unresolved memory accesses |
| StructuredToMemref headers/source | Independent masked/unmasked DMA load/store branches using existing address/mask parsing |
| UnstructuredToMemref headers/source | Explicit failure for unsupported accesses, preventing CPU fallback |
| `Conversion/BOSCAMEHelpers` headers/source | Shared workspace, policies, complete vector copies, rank-one and rectangular accesses, buffer matmul AME lowering |
| `SpmDma.h`, `DmaAlignment.h`, `BOSCAME.td`, `BOSCAMEDialect.cpp` | Typed byte widths/capacities and effective-address alignment proof |
| `SpmDmaCheck.cpp`, `LegalizeForLLVMExport.cpp`, `buddy-opt.cpp` | Footprints, token/SPM safety, typed pointer export, one-dimensional vector conversion and check registration |
| Conversion/Transforms CMake files, legacy TritonToBOSCAME source | Shared-helper linkage without a driver dependency cycle; legacy policy helpers remain compatible |
| Both example makefiles/build adapters/READMEs and linkers | New commands, typed ABI, audited NH/E6 images and stable stack cache policy |
| Both example `launch.c` files and `simcheck.py` files | Two-round board drivers, independent numerical oracles, input/output guards and functional verification |
| `runtime/boscame/nr/nr_math.c` | Freestanding scalar `sqrtf`/`rsqrtf` for the LayerNorm image |
| `examples/FPGA-BOSCAME/tools/check_experimental_board.py` | Independent post-execution DDR validation, ELF symbol bounds and machine-readable evidence |
| `tools/boscame_buffer_sim.py` and experimental/BOSCAME tests | Fail-closed functional model, independent oracles, guards, failure exit codes, ABI and default regressions |

`boscame-board-readback.py` tests the final-DDR checker with valid synthetic
images and corrupted data/guards/counters, and `boscame-freestanding-math.py`
tests the freestanding math entry points. These host tests support the board
workflow and do not replace hardware execution.

Use `ninja -C build buddy-opt`, `ninja -C build check-buddy`, and
`pre-commit run --files <changed-files>`. The new tests complement the retained
experimental and legacy BOSCAME suites; no legacy oracle or test is removed.
