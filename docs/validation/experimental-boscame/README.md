# Experimental BOSCAME board validation

Measured on 2026-10-14 on the NH/E6 NR patch7 FPGA image at 5.5296 MHz.
Both binaries come from the new experimental entry and pass directly through
`--lower-bosc-ame`, LLVM translation, assembly and the retained NR ELF audit.
The source TTIR arithmetic and reduction sizes are unchanged.

| Kernel | FPGA | Programs across two rounds | E6 output values checked | Host final-round comparison | Guards and input changes |
| --- | --- | --- | --- | --- | --- |
| LayerNorm 197×768 | 5 and 6, separate complete runs | 394 per board | 302592 per board, all exact against C oracle | 151296 per board, all bit-identical to independent sequential-f32 oracle; max abs error 0 | 0 |
| GEMM 128×1024×128 | 4 | 1024 | 262144, all exact | 131072, all exact against independent i64 product plus i32 wrap | 0 |

LayerNorm includes all 1024 logical lanes in variance, including the centered
256 padded X lanes. X/Weight/Bias/Out each use exactly 3072 authorized DDR
bytes per program. Programs execute in reverse row order, SPM is poisoned
once, and two rounds use different inputs without clearing SPM. Both matrix
and affine buffers have 1024-byte guards on both sides. The f32 oracle models
sequential reductions and two rounded scalar operations for rsqrt
(`1.0f / sqrtf(value)`). Equality is a result for these inputs; it is not a
universal floating-point bit-identity claim. The final-round maximum absolute
error against the independent f64 padded formula is 2.992675160307101e-5.

GEMM includes -128 and 127, physically asymmetric `[N,K]` B, nonzero C near
signed-i32 bounds, two outer K blocks and persistent SPM. Its C oracle uses
unsigned modular addition, and the host oracle independently computes in i64
then explicitly wraps to i32. All A/B/C before/after guards and all input bytes
are checked. No output sampling replaces the complete oracle.

The final DDR dumps contain 3 MiB starting at `0x80000000`, captured after the
application's PASS. The runner's pre-execution image mirror is not used for
numerical validation. Reports resolve symbols from the exact uploaded ELF:

- [GEMM FPGA4 report](gemm-fpga4.json) and [UART verdict](gemm-fpga4.uart.txt).
- [LayerNorm FPGA5 report](layernorm-fpga5.json) and [UART verdict](layernorm-fpga5.uart.txt).
- [LayerNorm FPGA6 report](layernorm-fpga6.json) and [UART verdict](layernorm-fpga6.uart.txt).

| Image | SHA256 of uploaded flat binary |
| --- | --- |
| LayerNorm | `2ef4f3c1acfe11f6b1624ad6322bdb546acc7cecc5169bf705d5cbc40856be6b` |
| GEMM boardcheck | `c8ad59c4ab9b5f319f6f02d16fa8e37a946ca8b64e1935261075ac9a08e82515` |

Each JSON report records the ELF and final DDR dump SHA256. Rebuilding after
formatting and the final Buddy build produced the same uploaded binary hashes.
Raw DDR dumps are kept with the run artifacts, rather than as large fixtures.

The first collection scripts sent UVHS `exit` immediately after readback.
`run.sh` then classified the harmless summary `Total ERROR: 0` as a platform
error and returned 1 for the FPGA4/FPGA5 collection sessions. Both applications
printed PASS, UVHS reported zero fatal/errors, and both independent DDR
checkers returned 0. The FPGA6 repeat used the runner's default idle cleanup
and returned 0 as well as passing the full DDR checker. Runner cleanup status
is kept distinct from application and numerical results.

Build and run from the respective example and runner directories:

```bash
make -C examples/FPGA-BOSCAME/layernorm_197x768 check bin
make -C examples/FPGA-BOSCAME/matmul_i8_128x1024x128 boardcheck
# After uploading each .bin using the platform's normal transfer command:
./run.sh --fpga=5 test=test/layernorm_197x768.bin
./run.sh --fpga=4 test=test/matmul_i8_128x1024x128.boardcheck.bin
```

Use the Triton Python environment for the builds. Capture post-execution DDR
through the platform's existing readmem facility, then run the independent
checker described in the [pipeline documentation](../../TritonToLinalgExperimentalBOSCAME.md#board-execution-and-post-execution-readback).
The checker returns nonzero for incomplete rounds, changed inputs/guards,
missing symbols, short dumps or numerical differences. Do not send `exit`
during the runner's live error scan; its default idle cleanup closes the run.

Validation completed: `ninja -C build buddy-opt`, `ninja -C build check-buddy`
(239 examples and 423 regressions passed; no unexpected failures), changed-file
pre-commit, both complete image audits, and GEMM default/debug/scalar variants.
The existing NH/E6 platform host test and DMA runtime tests (debug/NDEBUG)
also pass. Host negative tests exercise damaged DDR, missing/overlapping ELF symbols,
incomplete execution and failure exit codes. Freestanding math tests cover
10000 random positive f32 inputs, endpoints and special values.

The implementation uses ordinary stack buffers, packing and extra DMA/vector
copies. It materializes each dot result. No baseline speedup comparison was
made. The board drivers' cycle counters are retained in the UART records as
diagnostics, not a performance comparison.

The [complete changed-file inventory](changed-files.md) records file duties.
The [implementation document](../../TritonToLinalgExperimentalBOSCAME.md) records
supported/rejected access forms, default compatibility and the compiler checks.
