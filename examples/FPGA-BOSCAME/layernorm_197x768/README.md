# Triton LayerNorm on the NH/E6 FPGA

默认启用 **f32 有序规约的标量绕过**：在 SSA 层用顺序标量累加，最终 ELF 不含 `vfredosum.vs`。
使用 `make ALLOW_VECTOR_ORDERED_REDUCTIONS=1` 可显式恢复原向量指令；
切回 `=0` 无需 clean。支持范围、栈开销及验证状态见[绕过说明](../tools/README.scalar-ordered-reductions.md)。
2026-10-28 正式主镜像已上板两轮通过：302592 个输出逐位一致，errors/guards 均为0；
cycle 为9851570 / 9847844。详见绕过说明中的任务证据。
当前默认 planner 使用直接 DDR 路径；`make check` 校验该路径或旧版四条 DMA
路径的完整外部访问范围，下面的旧性能/栈记录对应各自历史镜像。


This directory consumes the checked-in `kernel.frontend.mlir` standard MLIR
Contract. It builds and checks the complete backend image without a Triton
Python installation. The original kernel and frontend exporter live in
`triton-riscv/examples/BOSCAMEFPGA/layernorm_197x768`.
See [the Contract handoff guide](../../../docs/TritonContract.md).

This example lowers the ViT-B/16 LayerNorm operator over a `[197,768] f32`
hidden state through the experimental BOSCAME backend. The fixed problem size (`ROWS=197`, `WIDTH=768`,
`BLOCK=1024`, `EPS=1e-12`) and the one-program-per-row launch grid are
defined in [build.py](build.py). The operator has no dot: the reductions,
elementwise and select operations stay in Linalg and lower to SPM vector code.

The only lowering route in this directory is:

```text
kernel.frontend.mlir -> Contract verify -> bosc-ame-plan -> verified planned IR
     -> bosc-ame-emit -> lower-bosc-ame -> LLVM IR -> RVV assembly -> ELF
```

## Quick start

```bash
make boscame llvm    # Contract -> planned/BOSCAME IR -> LLVM IR
make adapter asm     # typed ABI bridge and the restricted NR assembly
make check           # typed ABI, boundary DMA and NR ELF audits (no board)
make simcheck        # functional SPM/DMA model over all 197 programs
make                 # full pipeline -> build/layernorm_197x768/*.bin
```

`make boscame`, `make llvm` and `make asm` stop at the named stage.
Only ordinary Python and NumPy are needed for the backend helpers and
simulator. Select tools explicitly or through PATH:

```sh
make BUDDY_MLIR_BINARY_DIR=/path/to/buddy-build/bin \
  LLVM_BIN=/path/to/llvm-build/bin PYTHON=python3 all check simcheck
```

To consume a fresh export without copying it into this directory:

```sh
make FRONTEND_IR=/path/to/handoff/kernel.frontend.mlir all check simcheck
```

The input must match this example's fixed entry symbol, dimensions, launch grid,
aligned/disjoint buffer promises and launcher ABI. Generate a new handoff in the
frontend repository with `make ttir` or `build.py --export-contract`; this
makefile starts at Contract verification and has no Python-to-TTIR target.

## Pipeline

The conversion commands used by `make boscame llvm` are:

```bash
buddy-opt build/layernorm_197x768/kernel.frontend.mlir \
  --kernel-contract-verify -o build/layernorm_197x768/kernel.bufferized.mlir
buddy-opt build/layernorm_197x768/kernel.bufferized.mlir \
  --bosc-ame-plan --bosc-ame-plan-verify \
  -o build/layernorm_197x768/kernel.planned.mlir
buddy-opt build/layernorm_197x768/kernel.planned.mlir \
  --bosc-ame-emit -o build/layernorm_197x768/kernel.boscame.mlir
buddy-opt build/layernorm_197x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/layernorm_197x768/kernel.llvm.mlir
buddy-translate build/layernorm_197x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/layernorm_197x768/kernel.ll
```

Contract mode performs integer normalization, static mask specialization and
safe bufferization inside the frontend. `kernel.frontend.mlir` already contains
standard buffer-form Linalg/MemRef/SCF and target-neutral `contract.*` facts;
`kernel.bufferized.mlir` is its verified copy, retained as a familiar artifact
name. No second bufferization is needed. The planner binds actual operands,
selects each DDR/SPM route, rebuilds content versions and dependencies, and
records concrete `plan.*` decisions in `kernel.planned.mlir`. Emission verifies
that plan and expands it mechanically. There is no global storage-policy option.

The contract can be produced by another frontend without Triton or BOSCAME
dialect dependencies. See the [pipeline guide](../../../docs/BOSCAMEKernelContractPipeline.md)
and [regenerated-example validation](../../../docs/BOSCAMEKernelContractValidation.md)
for the interface, resource counts and artifact hashes.

## Memory and DMA

The planner places five 4 KiB float32 roots in SPM: a shared mean/invstd
broadcast buffer, X/centered/normalized/output, Weight, Bias and squares.
Bank 0 uses offsets 0, 4096 and 8192; bank 1 uses 0 and 4096. The bank
high-water marks are 12288 B and 8192 B, totaling **20480 B**. Safe frontend
bufferization preserves centered while squares is produced, then updates the
X/output root in place. No general lifetime-based physical window overlap is
assumed.

The logical tensor has 1024 lanes while only the first 768 are valid. Each
external DMA moves the 3072-byte prefix: X, Weight and Bias read, and Out
writes. Each load first fills all 1024 SPM lanes (X/Bias padding +0, Weight
padding 1). The output stays on SPM/DMA because the store covers only the valid
prefix of the private 1024-lane tile. Each program performs four DMA events,
9216 read bytes and 3072 write bytes; private arrays do not round-trip through
DDR.

| Buffer | DDR location | External DMA |
| --- | --- | --- |
| X | row × 3072 bytes | 3072-byte read |
| Weight | 3072 bytes | 3072-byte read |
| Bias | 3072 bytes | 3072-byte read |
| Out | row × 3072 bytes | 3072-byte write |

Both reductions use all 1024 logical lanes and divide by 768. The centered
prefix keeps the original subtraction; a subview fill writes exact +0 to
`[768,1024)` before squaring. Two ordered f32 reductions use 16-lane chunks
and `vfredosum.vs`, preserving the established reduction order and tolerance.
They write two private host rank-zero buffers, which emission converts to
aligned stack storage. Explicit scalar-load tasks capture the results before
the div/rsqrt chains. Mean and invstd are broadcast through the same SPM root
at different times; this version does not fuse those broadcasts into adjacent
RVV computations.

The plan has 21 tasks, versus 19 in the saved integrated baseline. The new
contract explicitly records two scalar-load events, two host initializations
and exact tail fill dependencies. Its five physical windows replace the
baseline's five windows with a 4 KiB gap: total bank high-water drops from
24576 B to 20480 B. The baseline's register broadcast forwarding is not part
of the current mechanical emitter, so the internal vector traffic differs.
External DMA ranges, floating-point operation order and output bytes remain
unchanged for the recorded full-grid validation.

## ABI and image memory

`build.py --adapter` emits the typed C bridge to
`_mlir_ciface_triton_layernorm_197x768`. The bridge takes **four typed
unranked f32 memref descriptor pointers** and six i32 grid/program scalars,
checks 64-byte root alignment, nonalias ranges, capacities and row-major
`[ROWS,WIDTH]` / `[WIDTH]` strides, then launches all 197 row programs
in reverse order with progress markers.

`link.ld` places user DDR at `0x80100000..0x802e0000` in PMP entry 5, and
the E6 stack at `0x802e0000..0x802f0000` in entry 6. Entry 6 keeps the
`0x0f` write-through policy during both DDR phases, so descriptor or stack
spills cannot be read through a stale cached entry after a policy switch. The
freestanding image links the NH/E6 boot/platform code, the DMA/cache-policy
runtime and `nr_math.c` for `sqrtf`/`rsqrtf`; `-ffp-contract=off` keeps the
square root and division separately rounded.

## Validation

2026-10-28 的正式 SSA 标量镜像已通过 FPGA4：两组输入、394 个 row programs，
302592 个输出全部逐位一致，errors=0、guard_errors=0；两轮 cycle 分别为
9851570 / 9847844。任务 ID 为661e98c04c27457ba59273b9adb8ce83。
镜像 SHA256 为58fd9d39faa5124c8b7348a79345144460899651ea7c8233ae4f953dd05de04c。
完整任务、UART 和诊断历史见[验证报告](../../../reference/rvv_reduction_probe/scalar-workaround/README.md)。

make check 验证 typed LLVM ABI、无 heap allocation、实际 IR 的全197行访问覆盖和
完整 ELF。当前镜像1911条静态指令、29条 RVV、0条 AME、0条 vfredosum；
规约 seed/result 没有 vfmv 中转。调用链栈审计为1056/65536字节。

make simcheck 按逆序执行所有197行，使用顺序 f32 oracle，atol=rtol=2e-5。
当前 direct-DDR 路径访问覆盖为 X×3 / Weight×1 / Bias×1 / Out×1，guards 完好，
最大绝对误差0；旧四 DMA 路径仍按自身契约校验。
--write-output FILE 保存605184字节输出，--compare-output FILE 与已有输出逐字节比较。
功能模拟和指令审计不代替上述硬件数值验收。

在主仓库根目录运行：

~~~bash
conda activate boscame
export BUDDY_ROOT="$PWD/buddy-mlir"
export BUDDY_MLIR_BINARY_DIR="$BUDDY_ROOT/build/bin"
export LLVM_BIN="$BUDDY_ROOT/llvm/build/bin"
make -C examples/BOSCAMEFPGA/layernorm_197x768 check simcheck
python examples/BOSCAMEFPGA/run_bin.py examples/BOSCAMEFPGA/layernorm_197x768/build/layernorm_197x768/layernorm_197x768.bin
~~~

复用已有 SSH 隧道，通过现有调度队列执行并保留原客户端至结束。
120秒为执行期限，排队另算；记录实际 task/node/FPGA 和镜像 SHA256。
该镜像由板端 oracle 完整验证，未声明 DDR output；不将装载读回当作运行结果。
旧独立 DDR dump 的 checker 仍可用于已有历史数据，不是本次验收的数据来源。
服务未提供实测时钟，cycle 不根据 UART 的频率标签换算时间。

## Files

| File | Role |
| --- | --- |
| `kernel.frontend.mlir` | Standard Contract input, exported in triton-riscv |
| `build.py` | Triton frontend export, fixed shape, typed ABI adapter and LLVM ABI check |
| `makefile` | TTIR -> BOSCAME -> LLVM -> assembly -> ELF/image pipeline |
| `link.ld` | NH/E6 entries, user DDR windows and stack cache policy |
| `launch.c` | Operands, two-round guarded f32 board oracle and NH report |
| `simcheck.py` | Whole-program functional execution, numerical oracle and guard checks |
| `../tools/restrict_fpga_assembly.py` | Restricted NR instruction stream from the `llc` assembly |
| `../tools/check_nr_elf.py` | Complete-ELF instruction audit |

See the [pipeline guide](../../../docs/BOSCAMEKernelContractPipeline.md)
and [current validation record](../../../docs/BOSCAMEKernelContractValidation.md).
