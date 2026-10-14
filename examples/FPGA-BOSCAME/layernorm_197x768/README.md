# LayerNorm through independent DMA load/store

The example retains the original Triton LayerNorm computation for 197 rows of
768 f32 elements. It contains no dot. The experimental BOSCAME path therefore
uses independent DMA loads/stores while Linalg/Arith/Math handle computation.

```bash
make ttir
make llvm adapter asm
make simcheck
```

Use the Triton/triton-riscv Python environment. Buddy/LLVM tools default to the
in-tree builds and can be overridden in the makefile. The two compiler stages
are directly connected:

```bash
buddy-opt build/layernorm_197x768/kernel.ttir \
  --triton-to-linalg-experimental="boscame=true dma-abi=args_aligned64_nonalias" \
  -o build/layernorm_197x768/kernel.boscame.mlir
buddy-opt build/layernorm_197x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/layernorm_197x768/kernel.llvm.mlir
buddy-translate build/layernorm_197x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/layernorm_197x768/kernel.ll
```

The logical tensor has 1024 lanes; the valid DDR prefix has 768 f32 elements,
so **each DMA transfers 3072 bytes**. X and Out have dynamic row offsets
`programId * 768` elements, proved to be `programId * 3072` aligned bytes.

| Buffer | DDR address | Padded ordinary lanes |
| --- | --- | --- |
| X | X + row × 768 | 0 |
| Weight | Weight | 1 |
| Bias | Bias | 0 |
| Out | Out + row × 768 | No DDR write |

Each load waits for DMA, then vector-copies SPM to an ordinary 1024-element
buffer. Stores vector-stage only the valid 768 elements into SPM and DMA them
to Out. Neither scalar computation nor ordinary Linalg touches SPM, and no
4096-byte user DDR access is emitted.

Both reductions still use all 1024 logical lanes. In particular, the centered
padded X lanes contribute `mean²` to variance. The compiler preserves this
original TTIR behavior and does not replace it with a 768-lane reduction.

`make adapter` generates a bridge with four typed unranked f32 descriptors
and six grid/program scalars. Descriptor sizes and offsets are elements. The
bridge checks 64-byte root alignment, nonalias ranges, capacities and strides,
then launches all 197 rows. Generated ordinary buffers live on an aligned
stack; a board application must place any DMA staging stack inside its
DDR/PMP window and link the existing DMA/cache-policy runtime.

`make simcheck` executes all rows in reversed order with persistent poisoned
SPM. It compares with an independent padded sequential-f32 oracle and checks
last-row and Weight/Bias tail sentinels plus the exact four DMA ranges per row.
Compiler tests also compare exactly against the unchanged experimental buffer
path; maximum absolute error against a high-precision padded formula was
2.61e-5 on the deterministic input. No arithmetic or reduction order changed.

Ordinary temporary buffers and vector copies add work. Performance is
unmeasured. `make asm` validates generated RVV forms and fences, but **board
execution/readback is not verified**; board floating-point instruction rounding
has not been compared with the functional oracle.

See [the compiler contract and tests](../../../docs/TritonToLinalgExperimentalBOSCAME.md).
