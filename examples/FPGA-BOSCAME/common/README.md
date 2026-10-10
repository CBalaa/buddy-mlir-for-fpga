# 公共平台资源

这些源文件供 `FPGA-BOSCAME` 下的样例复用，不依赖本地参考仓库或临时目录。
目录按“启动／平台／驱动／运行时／应用”五类职责组织；应用代码（各算子的
`launch.c`、`support.c`）不在本目录。

## 模块职责

| 路径 | 类别 | 职责 |
| --- | --- | --- |
| `uart/uart.c`、`uart/uart.h` | 驱动 | NR UART 设备：基地址 `0x310b0000`、32 位 MMIO、TX 偏移 `0x20`，按 `freq / (baud × 16)` 四舍五入计算除数；只负责设备寄存器访问，不含双核启动或算法报告。除 `print_uart`／`print_uart_int`／`print_uart_addr` 外还提供 NH/E6 报告用的 `print_uart_hex32`／`hex64`（小写定宽）与 `print_uart_dec32`；`UART_BASE`、`UART_PUTC` 可在 host 测试中覆盖 |
| `runtime/crt_uart.S` | 启动 | 通用单核启动：寄存器清零、FS/VS/XS、FPU 初始化、gp／栈、BSS／TBSS 清零、trap 入口，最后进入 C `_init` |
| `runtime/encoding.h` | 启动 | CRT 所需的 RISC-V CSR 定义 |
| `runtime/bare_runtime.c` | 启动 | C `_init`（复位堆 → `init_uart` → banner → `main`）、weak 启动钩子、终止 trap 入口 `handle_trap` |
| `runtime/bare_runtime_memory.c` | 运行时 | 单核 bump heap（可选 `BARE_RUNTIME_RECLAIM_FREE`）、`malloc`／`free`、`memcpy`／`memmove`／`memset`／`strlen`／`strcmp` |
| `runtime/bare_runtime_memref.c` | 运行时 | MLIR `memrefCopy` ABI：描述符布局、offset／stride、连续特化与通用路径 |
| `runtime/bare_runtime_trace.c` | 运行时 | `_mlir_ciface_buddyTraceCycle*` ABI、私有 trace 记录与周期汇总 |
| `runtime/bare_runtime_ame.c` | 运行时 | `ame_fence()` 与 AME resync 序列，仅 `BARE_RUNTIME_ENABLE_AME=1` 时编译 |
| `runtime/bare_runtime_w8a8.c` | 运行时 | 编译器调用的 `_mlir_ciface_buddy_w8a8_rvv_accumulate_n64` helper ABI |
| `runtime/bare_runtime_internal.h` | 内部 | 上述模块共享的 `BARE_RUNTIME_*` 编译宏默认值与 `bare_byte_copy`；不属于对外 ABI |
| `runtime/bosc_ame_dma.c`、`.h` | 驱动 | DMA／SPM 寄存器访问（SRC/DST/LEN/START/STATUS、CSR `0xbc0`、fence）与 lowering ABI，保留为单一模块 |
| `runtime/test_bosc_ame_dma.c` | 测试 | DMA helper 的 host 测试，直接 include `bosc_ame_dma.c` |
| `nr/*` | 平台 | NR 的 NH→RA 完整运行环境，见 [`nr/README.md`](nr/README.md) |
| `nh_e6/nh_e6_boot.S` | 启动 | NH/E6 双核启动：NH `_start`（栈、`mtvec`）、E6 `e6_start`（独立栈、FS/VS/XS/AME 状态、BSS 清零）、两个 trap 入口与停机循环；异常 CSR 捕获不依赖栈 |
| `nh_e6/nh_e6_platform.c`、`.h` | 平台 | RA SYSCTRL 的 NH→E6 启动与完成协议：释放 E6、有界等待、进度、超时、通用 PASS/FAIL/TRAP 判定、trap／DMA 诊断输出、E6 完成发布；E6 侧在调用应用 `e6_main()` 前完成七个 TOR PMP 区域与 `satp=0` 的地址空间设置，并封装 CSR `0xbc0` 的写通／uncached-load 缓存策略；通过 `nh_e6_app_tag()`／`nh_e6_app_report()` 调用应用报告，不含矩阵尺寸等应用字段 |
| `nh_e6/test_nh_e6_platform.c` | 测试 | NH/E6 平台的 host 测试：把 SYSCTRL 与 UART 换成内存假寄存器，直接 include 生产实现，覆盖 PMP/satp 顺序、CSR `0xbc0` 取值与启动释放流程 |
| `nh_e6/README.md` | 平台 | NH/E6 平台的硬件约定、接口与验证说明 |
| `toolchain.mk` | 构建 | 可配置的 LLVM 工具选择；从样例 makefile 引入 |

对外的头文件路径（`bare_runtime.h`、`bosc_ame_dma.h`、`uart.h`、`nr/nr_runtime.h`、
`nh_e6/nh_e6_platform.h`）和导出符号保持不变：模块拆分只改变实现文件，不改变接口。

### 依赖方向

```text
应用 launch.c / support.c
        │  只包含对外头文件
        ▼
启动  crt_uart.S / nr/crt.S / nh_e6/nh_e6_boot.S
        │                    │
        ▼                    ▼
平台  nr/nr_runtime.c（NH 启动 RA、mailbox、日志、trap）
      nh_e6/nh_e6_platform.c（NH 启动 E6、进度、超时、判定、报告；
                               E6 PMP/satp 设置与 CSR 0xbc0 缓存策略）
        │
        ▼
运行时  bare_runtime*.c、nr/nr_memory.c、nr_math.c、nr_copy.S、ame_sync.c
        │
        ▼
驱动  uart.c、bosc_ame_dma.c（寄存器访问，被平台和运行时调用）
```

- 启动代码先建立执行环境（栈、gp、BSS、trap 向量），再进入 C 入口；C 入口不
  承担初始化汇编环境。
- 平台层拥有共享状态的所有权约定：NR 中物理 UART 由 NH 访问，计算核日志经
  共享缓冲由 NH 转发；`print_uart*` 在 NR 上是日志转发接口，不是设备驱动。
  NH/E6 平台中 NH 独占物理 UART 并持有最终判定，E6 只把测量值写进 DDR；
  `nh_e6/nh_e6_platform.c` 通过 `nh_e6_app_report()` 回调应用，不知道任何
  应用字段；E6 的 PMP／`satp` 地址空间设置和 CSR `0xbc0` 策略由平台在
  `e6_main()` 之前完成，应用只使用 `nh_e6_cache_writes()`／
  `nh_e6_cache_uncached_loads()`，不再自己写 PMP 或缓存 CSR。
- 运行时层不直接访问设备寄存器。`nr_memory.c` 的致命错误只通过
  `nr_finish()`（平台报告，声明在 `nr_runtime_internal.h`）发布，不自己写
  mailbox。
- 驱动层不包含双核启动或算法专属报告。

### 每个镜像只链接一套实现

| 后端 | 启动 | 平台 | 运行时内存 | 日志 |
| --- | --- | --- | --- | --- |
| 单核 bare（`hello`） | `crt_uart.S` + `bare_runtime.c` | 无（NH 单核直接进入 C 运行时） | `bare_runtime_memory.c`、`bare_runtime_memref.c` | `uart.c` + `print_uart*`，经 `bare_runtime.c` 调用 |
| NR（`common/nr`） | `nr/crt.S` | `nr/nr_runtime.c` | `nr/nr_memory.c` | NR 控制台转发（`nr_console.inc`，NH 独占 UART） |
| NH/E6 双核（`gemm_64x64x64`） | `nh_e6/nh_e6_boot.S` | `nh_e6/nh_e6_platform.c` | 无（算子直接使用 DDR staging） | `uart.c` + `print_uart*`，由 NH 独占 |

`hello` 的链接清单在 [`hello/makefile`](../hello/makefile)，NR 的清单在
[`nr/nr.mk`](nr/nr.mk) 的 `NR_OBJECTS`，NH/E6 的清单在
[`gemm_64x64x64/makefile`](../gemm_64x64x64/makefile)（`nh_e6_boot.o`、
`nh_e6_platform.o`、`uart.o`）。算子集合再通过
`qwen3-0.6b/common.mk`、`vit-base-patch16-224/common.mk` 的 `RUNTIME_OBJS`
或 Python 镜像构建器复用同一份列表。不要把 bare 的 `bare_runtime.c` 与 NR 的
`nr_runtime.c`／`nr_memory.c` 同时链接：两套启动、内存和日志实现会重复定义。

`common/nr` 与 `common/nh_e6` 是两个不同的平台，不能互相替代：NR 用
`0x80010000` 作为 DDR 信号 mailbox，完成状态编码和链接布局与 NH/E6 的
RA SYSCTRL（`0x50000120`／`0x50000140`／`0x50000160`）不同。同一个镜像
不允许同时链接两套启动、平台或日志实现。

bare 与 NR 的分配器策略不同，不共用实现：bare 是固定 128 KiB 缓冲的 bump
heap（可选回收式 free），NR 是低 DDR 的 bump heap + `nr_heap_mark()`／
`nr_heap_reset()`，`free` 不回收、耗尽即 `FAIL`。两者的对齐、耗尽和 reset
语义各自保留。

## 来源

UART 驱动沿用 [ModelZoo](https://gitlink.org.cn/michaelcjl/ModelZoo) 的
`thirdparty/nr/src/uart.c` 和 `thirdparty/nr/include/uart.h` 的 NR 寄存器约定；
本地适配接入上游 `init_uart(freq, baud)` 接口，运行时按输入时钟和目标波特率
计算除数，不再把除数锁死为 8。
启动文件和 CSR 定义来自其 `thirdparty/platform-v01/`；C 运行时来自
`examples/tools/bare_runtime.c`、`bare_runtime.h`。
来源提交为 `8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3`；启动文件、CSR 定义和
C 运行时按该快照迁入，UART C 文件在本地保留 NR 的寄存器访问并接入上游的
动态除数接口。原文件中的归属说明予以保留。
CRT 原有注释引用 `riscv-dnn/include/common/crt.S`；不为来源文件补造许可证声明。

`crt_uart.S` 不含固定 UART 或 DDR 地址，通过链接符号设置内存区域。
当前复用方式为 NH 单核入口直接进入 C 运行时，不包含 NR 的 NH→RA 启动调度。
样例需提供链接脚本中的 `_start`、`__global_pointer$`、`__stack_base`、
BSS／TBSS 边界，栈大小与 CRT 的 1 MiB 约定一致。

`hello` 设置 `BARE_RUNTIME_ENABLE_AME=0`，链接时移除未使用的函数和数据。
公共运行时内保留的 AME／RVV 算子代码并不因此获得当前平台的数值验收；
需要 AME 的新样例应按自身 ABI 和硬件单独验证。

`common/nh_e6` 的 NH/E6 启动与报告流程来自 GEMM 示例原先自带的
`gemm_64x64x64/launcher.S`（该文件已随拆分删除），硬件约定取自交付参考的
patch6／patch7 `test/src/cases/nh_e6_dma_spm` 用例：入口 `0x80000000`／
`0x80010000`，完成码 `2`／`0xee`／`0xed`，SYSCTRL `0x50000120`／`0x50000140`／
`0x50000160`。迁移只把已验证的汇编流程改写为 C 平台模块和最小启动汇编，
没有改变寄存器顺序、fence 位置、判定逻辑或 UART 文本。E6 的七个 TOR PMP
区域、`satp=0` 与 `sfence.vma` 顺序，以及 CSR `0xbc0` 的写通／uncached-load
策略，同样从 GEMM launcher 的 `e6_dma_platform_init()`／阶段切换代码移入
平台模块：SPM／DMA 取值继续来自 `bosc_ame_dma.h`，DDR staging 窗口由样例
`link.ld` 导出的 `__nh_e6_ddr_window_start`／`__nh_e6_ddr_window_end` 提供
（GEMM 为 `[0x80100000, 0x80300000)`，不缩小为数组实际占用范围）。
