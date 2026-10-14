# Changed files and duties

Inventory for this task, relative to its starting revision. Existing unrelated workspace changes are excluded.

| File | Responsibility |
| --- | --- |
| `docs/TritonToLinalgExperimentalBOSCAME.md` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/README.md` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/changed-files.md` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/gemm-fpga4.json` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/gemm-fpga4.uart.txt` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/layernorm-fpga5.json` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/layernorm-fpga5.uart.txt` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/layernorm-fpga6.json` | Document contracts, commands, implementation and measured validation. |
| `docs/validation/experimental-boscame/layernorm-fpga6.uart.txt` | Document contracts, commands, implementation and measured validation. |
| `examples/FPGA-BOSCAME/layernorm_197x768/.gitignore` | Keep generated example artifacts out of version control. |
| `examples/FPGA-BOSCAME/layernorm_197x768/README.md` | Document contracts, commands, implementation and measured validation. |
| `examples/FPGA-BOSCAME/layernorm_197x768/build.py` | Export TTIR, emit checked typed/legacy ABI adapters and validate wrapper signatures. |
| `examples/FPGA-BOSCAME/layernorm_197x768/kernel.py` | Retain the original padded LayerNorm Triton computation. |
| `examples/FPGA-BOSCAME/layernorm_197x768/launch.c` | Run NH/E6 kernels and complete independent two-round numerical/input/guard checks. |
| `examples/FPGA-BOSCAME/layernorm_197x768/link.ld` | Place user DDR and CPU stack in distinct PMP cache-policy regions with bounds checks. |
| `examples/FPGA-BOSCAME/layernorm_197x768/makefile` | Connect experimental lowering directly to LLVM and build/audit executable board variants. |
| `examples/FPGA-BOSCAME/layernorm_197x768/simcheck.py` | Run complete program grids against independent functional oracles and DDR guards. |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/README.md` | Document contracts, commands, implementation and measured validation. |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/build.py` | Export TTIR, emit checked typed/legacy ABI adapters and validate wrapper signatures. |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/launch.c` | Run NH/E6 kernels and complete independent two-round numerical/input/guard checks. |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/link.ld` | Place user DDR and CPU stack in distinct PMP cache-policy regions with bounds checks. |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/makefile` | Connect experimental lowering directly to LLVM and build/audit executable board variants. |
| `examples/FPGA-BOSCAME/matmul_i8_128x1024x128/simcheck.py` | Run complete program grids against independent functional oracles and DDR guards. |
| `examples/FPGA-BOSCAME/tools/boscame_buffer_sim.py` | Execute DMA/SPM/vector/AME buffer IR with fail-closed memory checks. |
| `examples/FPGA-BOSCAME/tools/check_experimental_board.py` | Independently compare post-execution DDR using exact ELF symbols and numerical oracles. |
| `midend/include/Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h` | Share workspace budgets, DDR policies and complete one-dimensional vector copies. |
| `midend/include/Conversion/StructuredToMemref/StructuredToMemref.h` | Thread backend configuration and independently stage masked/unmasked loads and stores. |
| `midend/include/Conversion/TritonArithToLinalg/ConversionPatterns.hpp` | Thread configuration, mark eligible integer matmuls and preserve common function conversion. |
| `midend/include/Conversion/TritonArithToLinalg/TritonArithToLinalg.h` | Thread configuration, mark eligible integer matmuls and preserve common function conversion. |
| `midend/include/Conversion/TritonLoweringConfig.h` | Define backend configuration and default-compatible pass options. |
| `midend/include/Conversion/TritonToLinalgExperimental/Passes.td` | Define backend configuration and default-compatible pass options. |
| `midend/include/Conversion/UnstructuredToMemref/UnstructuredToMemref.h` | Reject unsupported memory operations instead of silently bypassing DMA. |
| `midend/include/Dialect/BOSCAME/BOSCAME.td` | Validate typed contiguous DMA endpoints, capacities and existing op contracts. |
| `midend/include/Dialect/BOSCAME/DmaAlignment.h` | Prove effective byte-address alignment, including divisible dynamic offsets. |
| `midend/include/Dialect/BOSCAME/SpmDma.h` | Validate typed contiguous DMA endpoints, capacities and existing op contracts. |
| `midend/lib/Conversion/BOSCAMEHelpers/BOSCAMEHelpers.cpp` | Share workspace budgets, DDR policies and complete one-dimensional vector copies. |
| `midend/lib/Conversion/BOSCAMEHelpers/CMakeLists.txt` | Link shared helpers and register lowering/checks without a driver dependency cycle. |
| `midend/lib/Conversion/BOSCAMEHelpers/MarkedMatmul.cpp` | Pack buffer-form marked integer matmuls and emit blocked AME operations. |
| `midend/lib/Conversion/BOSCAMEHelpers/StructuredMemory.cpp` | Preserve physical rank-two layout while staging valid rectangles through DMA. |
| `midend/lib/Conversion/CMakeLists.txt` | Link shared helpers and register lowering/checks without a driver dependency cycle. |
| `midend/lib/Conversion/StructuredToMemref/CMakeLists.txt` | Link shared helpers and register lowering/checks without a driver dependency cycle. |
| `midend/lib/Conversion/StructuredToMemref/StructuredToMemref.cpp` | Thread backend configuration and independently stage masked/unmasked loads and stores. |
| `midend/lib/Conversion/StructuredToMemref/StructuredToMemrefPass.cpp` | Thread backend configuration and independently stage masked/unmasked loads and stores. |
| `midend/lib/Conversion/TritonArithToLinalg/TritonArithToLinalg.cpp` | Thread configuration, mark eligible integer matmuls and preserve common function conversion. |
| `midend/lib/Conversion/TritonArithToLinalg/TritonArithToLinalgPass.cpp` | Thread configuration, mark eligible integer matmuls and preserve common function conversion. |
| `midend/lib/Conversion/TritonToBOSCAME/CMakeLists.txt` | Link shared helpers and register lowering/checks without a driver dependency cycle. |
| `midend/lib/Conversion/TritonToBOSCAME/TritonToBOSCAME.cpp` | Retain the legacy entry while sharing DDR policy helpers. |
| `midend/lib/Conversion/TritonToLinalgExperimental/CMakeLists.txt` | Link shared helpers and register lowering/checks without a driver dependency cycle. |
| `midend/lib/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimentalPass.cpp` | Validate ABI, organize lowering, bufferize, lower marked matmuls and check exit IR. |
| `midend/lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp` | Reject unsupported memory operations instead of silently bypassing DMA. |
| `midend/lib/Dialect/BOSCAME/IR/BOSCAMEDialect.cpp` | Validate typed contiguous DMA endpoints, capacities and existing op contracts. |
| `midend/lib/Dialect/BOSCAME/Transforms/CMakeLists.txt` | Link shared helpers and register lowering/checks without a driver dependency cycle. |
| `midend/lib/Dialect/BOSCAME/Transforms/LegalizeForLLVMExport.cpp` | Export typed DMA descriptors with element-scaled effective addresses. |
| `midend/lib/Dialect/BOSCAME/Transforms/SpmDmaCheck.cpp` | Trace typed roots/footprints and enforce DMA waits, aliasing and SPM safety. |
| `runtime/boscame/nh_e6/README.md` | Document contracts, commands, implementation and measured validation. |
| `runtime/boscame/nh_e6/nh_e6_platform.h` | Supply freestanding f32 math or document the stable stack/PMP cache policy. |
| `runtime/boscame/nr/README.md` | Document contracts, commands, implementation and measured validation. |
| `runtime/boscame/nr/nr_math.c` | Supply freestanding f32 math or document the stable stack/PMP cache policy. |
| `runtime/boscame/nr/nr_runtime.h` | Supply freestanding f32 math or document the stable stack/PMP cache policy. |
| `tests/Conversion/BOSCAME/typed-dma-contracts.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/BOSCAME/typed-dma.mlir` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/Inputs/layernorm.ttir` | Provide portable TTIR fixtures retaining the original kernel computation. |
| `tests/Conversion/TritonToLinalgExperimental/Inputs/matmul-i8.ttir` | Provide portable TTIR fixtures retaining the original kernel computation. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-adapter.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-board-readback.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-codegen.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-contracts.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-freestanding-math.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-gemm.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-k-blocked.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-layernorm.py` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-load-store.mlir` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-options.mlir` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tests/Conversion/TritonToLinalgExperimental/boscame-spm-store.mlir` | Verify contracts, numerical oracles, ABI/code generation, guards and nonzero failures. |
| `tools/buddy-opt/buddy-opt.cpp` | Register vector lowering and BOSCAME contract checks. |
