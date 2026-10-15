# Direct-SPM revision: changed files

Paths below are relative to the repository root. This inventory covers the
direct-SPM revision after `5a980622`; the earlier option/typed-DMA implementation
has its own [inventory](../experimental-boscame/changed-files.md).

| File | Responsibility |
| --- | --- |
| `midend/include/Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h` | Shared vector-fill, store-buffer inference, placement and computation interfaces |
| `midend/lib/Conversion/BOSCAMEHelpers/BOSCAMEHelpers.cpp` | Rank-one DMA loads retain logical SPM buffers; vector `other` initialization; direct SPM stores |
| `midend/lib/Conversion/BOSCAMEHelpers/StructuredMemory.cpp` | Rank-two direct-SPM loads/stores preserve logical shape and physical pitch |
| `midend/lib/Conversion/BOSCAMEHelpers/SPMPlacement.cpp` | Static local computation-buffer placement in the common bounded workspace; retain ordinary scalar accumulators |
| `midend/lib/Conversion/BOSCAMEHelpers/SPMComputations.cpp` | One-dimensional SPM vector fills/arithmetic/copies; source-order single-lane f32 sum; reject unsupported consumers |
| `midend/lib/Conversion/BOSCAMEHelpers/MarkedMatmul.cpp` | Direct compatible SPM A/B tile loads; bounded layout-preserving snapshots for general packing/C; floating matmul boundary |
| `midend/lib/Conversion/BOSCAMEHelpers/CMakeLists.txt` | Build and link the placement/computation helpers |
| `midend/lib/Conversion/StructuredToMemref/StructuredToMemrefPass.cpp` | Resolve loads before stores so buffer inference retains the producer's memory space/layout |
| `midend/lib/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimentalPass.cpp` | Schedule placement, matmul and SPM consumer lowering after internal bufferization |
| `examples/FPGA-BOSCAME/tools/boscame_buffer_sim.py` | Execute one-dimensional vector constants/broadcasts/arithmetic/extracts with per-operation rounding and strict memory checks |
| `tests/Conversion/TritonToLinalgExperimental/boscame-spm-compute.py` | Numeric/IR/LLVM/assembly checks for SPM arithmetic, ordered cancellation, wrap and explicit rejection |
| `tests/Conversion/TritonToLinalgExperimental/boscame-spm-matmul.py` | Direct integer tiles, both B layouts, general K/padding/C, bounded snapshot failures and ordinary floating matmuls |
| `tests/Conversion/TritonToLinalgExperimental/boscame-vector-simulator.py` | Independent semantic tests for vector execution and invalid operations |
| `tests/Conversion/TritonToLinalgExperimental/boscame-gemm.py` | Direct A/B SPM structure checks alongside the retained independent exact GEMM oracle |
| `tests/Conversion/TritonToLinalgExperimental/boscame-k-blocked.py` | Direct-SPM structure checks alongside retained K-blocking/mask/other/access-bound checks |
| `tests/Conversion/TritonToLinalgExperimental/boscame-layernorm.py` | Direct SPM arithmetic and source-order reduction checks; retained full-row/default/independent numerical comparisons |
| `tests/Conversion/TritonToLinalgExperimental/boscame-contracts.py` | Update the generalized matmul-buffer diagnostic expectation |
| `tests/Conversion/TritonToLinalgExperimental/boscame-load-store.mlir` | Expect DMA with direct SPM lifetime and no unconditional vector bounce |
| `tests/Conversion/TritonToLinalgExperimental/boscame-options.mlir` | Direct-SPM output expectations; retain false/default compatibility checks |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/makefile` | Describe the current direct-SPM integer data flow |
| `examples/FPGA-BOSCAME/layernorm_197x768/README.md` | Direct-SPM LayerNorm flow, ordered sum, workspace and current board evidence |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/README.md` | Direct A/B AME path, remaining consumer packing and current board evidence |
| `docs/TritonToLinalgExperimentalBOSCAME.md` | Pipeline, supported consumers, numerical constraints and direct-SPM validation link |
| `docs/validation/experimental-boscame-direct-spm/README.md` | Current hardware execution, numerical results, limits and reproducible commands |
| `docs/validation/experimental-boscame-direct-spm/changed-files.md` | This inventory |
| `docs/validation/experimental-boscame-direct-spm/layernorm-fpga6.json` | Independent post-execution LayerNorm DDR report and ELF/dump hashes |
| `docs/validation/experimental-boscame-direct-spm/layernorm-fpga6.uart.txt` | Sanitized application verdict and counters |
| `docs/validation/experimental-boscame-direct-spm/gemm-fpga4.json` | Independent post-execution exact GEMM DDR report and ELF/dump hashes |
| `docs/validation/experimental-boscame-direct-spm/gemm-fpga4.uart.txt` | Sanitized application verdict and counters |
| `docs/validation/experimental-boscame-direct-spm/layernorm-final-equivalence.json` | Final compiler regeneration matches the executed candidate's complete kernel code |
| `docs/validation/experimental-boscame-direct-spm/gemm-final-equivalence.json` | Final rebuild matches the executed GEMM ELF and flat binary exactly |

The local GEMM launcher edit, removed legacy example directory, and untracked
user examples were preserved and are not part of this revision.
