# Hello World：NR 平台上的 NH 单核程序

纯 C 裸机样例，面向 `RAV0.5_FPGA_ALPHA_260908`。NH 直接运行 `main()`，
通过 UART 输出 `Hello, World!` 和验证结果，不启动 RA、不执行 AME。

## 构建与运行

从仓库根目录执行：

```bash
make -C examples/FPGA-BOSCAME/hello all check size
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/hello/build/hello.bin --fpga=5
```

或者进入样例目录：

```bash
cd examples/FPGA-BOSCAME/hello
make all check
../fpga_run.sh build/hello.bin --fpga=5
```

产物为 `build/hello.elf` 和未经 padding 的 `build/hello.bin`。
上板脚本自动上传到独立目录，开启串口后调用服务器 `make uv_run5` 完成补齐、
加载和运行，再把串口输出返回本地。无需另开 minicom。
默认接收 10 秒，可用 `--capture-seconds=60` 延长。
工具链、SSH 及服务器工作目录配置见[公共说明](../README.md)。

## 平台配置

| 项目 | 配置 |
| --- | --- |
| 链接／NH 入口 | `0x80000000` |
| DDR 加载目标 | `nanhu_xilinx.i_ddr`，从索引 0 开始 |
| UART 基地址 | `0x310b0000` |
| UART 状态／发送寄存器 | `+0x14`／`+0x20`，32 位 MMIO |
| UART 初始化 | 按 `freq / (baud × 16)` 四舍五入计算除数，8N1 |
| 配套服务器时钟／串口 | 5.5296 MHz／115200 baud |

UART 使用 [`../common/uart`](../common/uart)；通用 CRT 和 C 运行时使用
[`../common/runtime`](../common/runtime)。NR 的 `init_uart(freq, baud)`
根据运行时传入的输入时钟和波特率计算 UART 除数；未提供参数时回退到
5.5296 MHz／115200。[独立链接脚本](platform/hello_nr.ld)
提供地址布局、BSS 边界和 1 MiB 栈。

bare 运行时按职责拆分：`bare_runtime.c` 提供 `_init`、weak 钩子和终止 trap
入口，`bare_runtime_memory.c`、`bare_runtime_memref.c`、`bare_runtime_trace.c`、
`bare_runtime_ame.c`、`bare_runtime_w8a8.c` 分别提供堆与内存、memrefCopy、
trace、AME 和 W8A8 helper ABI，共享的编译宏默认值在
`bare_runtime_internal.h`。hello 链接这些模块，但用
`-DBARE_RUNTIME_ENABLE_AME=0` 关闭 AME，并由 `--gc-sections` 丢弃未使用的
函数和数据；只调用启动路径时它们不进入镜像。模块表见
[`../common/README.md`](../common/README.md)。

时钟和波特率也可以在构建时覆盖，例如：

```bash
make -C examples/FPGA-BOSCAME/hello clean all \\
  UART_FREQ_HZ=5529600u UART_BAUD=115200u
```

启动流程：清零 BSS/TBSS → `_init` → 初始化 UART → banner → `main()`
→ 完成提示 → `wfi`。程序只输出一次，不等待串口输入。

旧平台的 `0x86400000` 入口与 `0x10000000` UART 不适用于 NR；
只改变寄存器访问宽度并不足以完成平台适配。

## 预期输出

```text

========================================
  BUDDY-MLIR Hello World (plain C)
  NR / NH RISC-V @ 0x80000000
========================================

rt: call main
Hello, World!
computed: 40 + 2 = 0000002A
verify hello: PASS

=== Hello Done ===
```

`print_uart_int` 输出八位十六进制，`0000002A` 即十进制 42。
`40 + 2` 会被编译器折叠为常量，样例用于验证启动和输出路径。

## 验证范围

- `make check` 使用主机 UART 桩逐字节检查应用输出，不验证 FPGA MMIO。
- `make dump` 展示完整反汇编，供检查入口、启动和 UART 地址。
- 2026-09-16，迁移前的同一程序已在 FPGA5（B1/F1）实测：完整 256 字节
  UART 输出与主机预期逐字节一致，DDR 读回与补齐镜像一致。
- 历史验收的原始镜像为 1407 字节，SHA256：
  `68f08af8a38387d62f45d77f0fbc8b9e51b69d50a9812e74669f5de43fc91058`。
- 当前 patch6 默认镜像为 1569 字节，SHA256：
  `e294d924b00a5b70593d81f12d2bd15071fc342143a86384aae3a9d7c6c67f6d`；
  已在 FPGA4 以 5.5296 MHz／115200 baud 上板验证，UART 输出 256 字节、DDR
  读回一致，并出现 `verify hello: PASS`。
- 公共上板脚本的离线测试命令及日志位置见[公共说明](../README.md)。

手工加载时必须用新原始镜像重新生成补齐文件；仅重新加载旧的
`hello_padded.bin` 不会包含新编译的程序。
