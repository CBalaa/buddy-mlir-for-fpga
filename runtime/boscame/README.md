# 公共平台资源

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