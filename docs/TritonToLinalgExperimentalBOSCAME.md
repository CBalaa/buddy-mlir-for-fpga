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

Loads DMA only the authorized DDR region into a bounded SPM view, wait, copy
complete 64-byte blocks with **one-dimensional** vector loads/stores into an
aligned ordinary buffer, then create a real `bufferization.to_tensor`.
Masked-off ordinary lanes retain the original `other` value. Stores use a real
`bufferization.to_buffer`, vector-copy valid lanes into SPM, DMA only the
original valid DDR region and wait. Ordinary arithmetic, reduction and rsqrt
therefore access ordinary memory even when lowered to scalar loops.

Integer dots first become standard tensor `linalg.matmul` operations carrying
the unit attribute `bosc_ame.lower_to_ame`. After empty-tensor and one-shot
bufferization, the driver verifies the number of marked matmuls survived and
calls a C++ helper on the buffer form. The helper packs ordinary A/B/C buffers,
DMA-stages them, emits tile load/mma/store, and materializes each dot result
back into the original ordinary output buffer. K blocks are 64 wide; partial
AME tiles are padded in ordinary memory. MatmulConverter's separate original-C
addition is preserved and is not repeated inside the AME helper.

All DMA/cache-policy calls finish outside AME register live ranges. Final
lowering inserts the existing DDR read/write policies at the actual ordinary
CPU accesses. A write policy for an SPM-to-DDR vector copy precedes its adjacent
SPM vector load, keeping the RVV value out of a runtime-call live range.

Every static access point has a non-reusing bank/offset allocation, aligned to
64 bytes. Memory and matmul helpers share the two-bank budget. K blocks may
reuse their own windows after tile store; access points do not share windows.
There are no SPM allocations: storage is expressed by `bosc_ame.spm.view`.
Static ordinary temporary buffers are promoted to stack when their lifetime
allows it. On the NH/E6 image, DMA packs on the E6 stack require that stack to
lie inside the exported DDR/PMP window.

The exit check rejects residual Triton/tensors/bridges/casts/target matmuls,
SPM allocation/free, scalar SPM access, ordinary memref copies or Linalg on
SPM, and unmatched/unsafe DMA tokens. `lower-bosc-ame` owns standard-to-LLVM
conversion; the experimental driver does not duplicate it.

## Required examples and numerics

```bash
make -C examples/FPGA-BOSCAME/layernorm_197x768 llvm adapter asm simcheck
make -C examples/FPGA-BOSCAME/matmul_i8_128x1024x128 check simcheck
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

GEMM executes all 512 programs and compares all 131072 i32 outputs exactly with
an independent wide-integer NumPy product followed by i32 wrap. Signed extrema,
asymmetric physical B, nonzero previous C, outer K accumulation, persistent
SPM and DDR guards are checked. A separate 64×64×256 test checks four internal
K blocks, a nonzero dot initializer and dynamic rectangular masks/nonzero
`other`; excluded DDR addresses are inspected in the access log.

This implementation deliberately adds ordinary temporary buffers, packing and
extra DMA/vector copies. It materializes results at every dot boundary instead
of keeping an accumulator across the outer K loop. Latency, cycles and speedup
have not been measured. **Board execution and readback have not been verified.**
An ELF containing RVV/AME instructions is a code-generation check, not a
numerical board result.

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
| Both example makefiles/build adapters/READMEs and GEMM linker/simcheck | New commands, typed ABI, authorized DMA stack and functional verification |
| `tools/boscame_buffer_sim.py` and experimental/BOSCAME tests | Fail-closed functional model, independent oracles, guards, failure exit codes, ABI and default regressions |

Use `ninja -C build buddy-opt`, `ninja -C build check-buddy`, and
`pre-commit run --files <changed-files>`. The new tests complement the retained
experimental and legacy BOSCAME suites; no legacy oracle or test is removed.
