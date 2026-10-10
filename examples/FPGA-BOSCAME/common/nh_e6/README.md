# NH/E6（RA SYSCTRL）双核平台

此目录是 BOSCAME SPM/DMA 样例的 NH/E6 平台运行环境：NH 从 `0x80000000`
启动，独占物理 UART 并持有最终判定；E6 从 `0x80010000` 启动，运行应用算法
并通过 SYSCTRL 发布完成状态。`gemm_64x64x64` 是当前唯一使用它的样例。

`nh_e6` 是平台名称，与 [`../nr`](../nr/README.md) 并列但不同：NR 用
`0x80010000` 作为 DDR 信号 mailbox，完成状态编码和链接布局也不同。两者不能
互相替代，同一个镜像也不允许同时链接两套启动、平台或日志实现。

## 文件职责

| 文件 | 类别 | 职责 |
| --- | --- | --- |
| `nh_e6_boot.S` | 启动 | NH `_start`（栈、`mtvec`）、E6 `e6_start`（独立栈、FP/vector/AME `mstatus`、`mtvec`、镜像 BSS 清零）、两个 4 字节对齐的 trap 入口与停机循环；E6 trap 入口捕获 `mcause`/`mepc`/`mtval` 后发布 `0xed`，全程不碰栈 |
| `nh_e6_platform.c` | 平台 | NH 侧：初始化 UART、打印 start、清零 DONE／MAILBOX／RESULT、经 SYSCTRL 编程并释放 E6、有界等待、进度报告、超时、通用 PASS/FAIL/TRAP/TIMEOUT 判定、trap CSR 与 DMA 失败上下文输出、调用应用报告、发布最终 mailbox；E6 侧：地址空间设置（七个 TOR PMP 区域、`satp=0`、`sfence.vma`）、CSR `0xbc0` 缓存策略、调用 `e6_main()`、发布 PASS/FAIL、等待 NH 结果并转发到 mailbox |
| `nh_e6_platform.h` | 平台 | 固定硬件地址与完成码、链接脚本必须导出的 DDR 窗口符号、平台服务与应用必须提供的三个钩子、平台诊断单元的声明 |
| `test_nh_e6_platform.c` | 测试 | host 测试：把 SYSCTRL、UART 与 CSR 写入换成内存假寄存器并直接 include 生产实现 |

应用侧只需一个 C 文件（GEMM 的 `gemm_64x64x64_launch.c`）：它提供
`e6_main()`、`nh_e6_app_tag()` 和 `nh_e6_app_report()`，通过
`nh_e6_platform.h` 使用平台服务。平台不知道矩阵尺寸或应用诊断字段；应用也
不直接写 SYSCTRL、不自己实现启动或停机，不写 PMP／`satp`／CSR `0xbc0`。

## E6 地址空间设置与缓存策略

`nh_e6_e6_main()` 在调用应用 `e6_main()` 之前先执行
`nh_e6_platform_init()`，因此应用的第一次 CSR `0xbc0` 切换和第一次 DMA/AME
访问都发生在 PMP／`satp` 就绪之后。顺序与迁移前 GEMM launcher 的
`e6_dma_platform_init()` 逐条一致，取值来源不同：

| 步 | CSR | 值／来源 |
| --- | --- | --- |
| 1 | `pmpaddr0` | `BOSC_AME_SPM0_BASE >> 2`（`bosc_ame_dma.h`） |
| 2 | `pmpaddr1` | `(BOSC_AME_SPM1_BASE + BOSC_AME_SPM_BANK_BYTES) >> 2` |
| 3 | `pmpaddr2` | `BOSC_AME_DMA0_BASE >> 2` |
| 4 | `pmpaddr3` | `(BOSC_AME_DMA1_BASE + BOSC_AME_DMA_CHANNEL_STRIDE) >> 2` |
| 5 | `pmpaddr4` | `__nh_e6_ddr_window_start >> 2`（链接脚本导出） |
| 6 | `pmpaddr5` | `__nh_e6_ddr_window_end >> 2`（链接脚本导出） |
| 7 | `pmpaddr6` | `0x003fffffffffffff`（其余地址空间） |
| 8 | `pmpcfg0` | `0x000f0f0f0f0f0f0f`（TOR、七段 R/W/X） |
| 9 | — | `fence iorw, iorw` |
| 10 | `satp` | `0`（bare；样例给 DMA/AME 传物理地址） |
| 11 | — | `sfence.vma zero, zero`（切换后必须刷 PTE 缓存） |

七个 TOR 段的覆盖范围：`[0, 0x20000000)`、两个 SPM bank
`[0x20000000, 0x20420000)`、`[0x20420000, 0x60000000)`、两个 DMA 通道
`[0x60000000, 0x60020000)`、含镜像与两核栈的 `[0x60020000, 0x80100000)`、
DDR staging 窗口 `[0x80100000, 0x80300000)`、其余地址空间。

**DDR 窗口由链接脚本提供。** 使用本平台的样例 `link.ld` 必须导出两个绝对
符号（`nh_e6_platform.c` 以 `extern char []` ＋ `&symbol` 取值）：

```ld
__nh_e6_ddr_window_start = 0x80100000;
__nh_e6_ddr_window_end = 0x80300000;
```

窗口按硬件约定保留，不按 `.dma_source`／`.dma_output` 的实际大小缩小；
`link.ld` 用 `ASSERT` 保证两个 staging 段落在窗口内，GEMM 的 `make audit`
（`pmpcheck`）还会在镜像上复查符号值与 PMP/satp 顺序。

**CSR `0xbc0` 缓存策略** 由平台统一提供。参考程序的 `shared_writes()`／
`shared_reads()` 都是先 `fence iorw, iorw` 再写 CSR，本模块保持同样顺序：

| 接口 | `0xbc0` | 用途 |
| --- | --- | --- |
| `nh_e6_cache_writes()` | `0x0f0f0f0f0f0f0f0f` | 写通 store、cached load；阶段边界恢复写策略 |
| `nh_e6_cache_uncached_loads()` | `0x0f0f000f000f000f` | 写通 store、uncached load；读取 DMA 刚写入的数据前使用 |

阶段切换仍是 `nh_e6_progress(marker)` → `nh_e6_cache_writes()`（GEMM 应用里
的 `e6_stage()`），阶段标记和访问顺序不变。与迁移前相比，写策略前多了一条
`fence iorw, iorw`（uncached-load 路径原本就有），与参考程序一致，只加强
顺序、不放松任何依赖。

## 固定硬件约定（已板测，不可重编号）

| 项 | 值 |
| --- | --- |
| NH 入口 | `0x80000000`（链接脚本 `ASSERT`） |
| E6 入口 | `0x80010000`（链接脚本 `ASSERT`，NH 把它写进 SYSCTRL PC） |
| E6 编程／释放 | SYSCTRL `+0x100` PC、`+0x060` control、`+0x050` release |
| E6 完成码 | `0x50000120`：`2` = PASS，`0xee` = FAIL，`0xed` = TRAP |
| NH 最终结果 | `0x50000140`：`0xff` = PASS，`0xee` = FAIL（TRAP／TIMEOUT 也是 `0xee`） |
| patch6 进度 mailbox | `0x50000160`：NH 释放前清零，E6 发布进度，终态值再转写一次供 VCS 完成协议 |
| 有界等待 | `NH_E6_WAIT_ITERATIONS`（默认与板测镜像一致的 `100000000`） |
| UART | `common/uart` 的 `0x310b0000` 驱动，只有 NH 访问 |
| E6 地址空间 | 七个 TOR PMP 区域、`satp=0`（bare）、切换后 `sfence.vma`；SPM／DMA 常量取自 `bosc_ame_dma.h` |
| DDR staging 窗口 | 样例 `link.ld` 导出的 `__nh_e6_ddr_window_start`／`__nh_e6_ddr_window_end`（GEMM：`[0x80100000, 0x80300000)`） |
| CSR `0xbc0` | 写通 store：`0x0f0f0f0f0f0f0f0f`；写通 store＋uncached load：`0x0f0f000f000f000f`，均由平台接口写入 |

访问顺序保持原 `launcher.S` 的写法：清零三个寄存器 → `fence iorw, iorw` →
写 PC／control=0／release=1／control=1 → `fence iorw, iorw`；轮询每轮先
`fence r, r` 再读 DONE 和 MAILBOX；每次发布后 `fence iorw, iorw`；DDR 诊断
单元写完 `fence rw, rw`。E6 侧的地址空间设置与缓存策略顺序见上一节。

## 失败路径

- **E6 trap**：`e6_trap_entry` 把 `mcause`/`mepc`/`mtval` 存进
  `e6_diag_mcause`/`e6_diag_mepc`/`e6_diag_mtval`，发布 `0xed`，停机。
- **DMA 失败**：`nh_e6_publish_dma_failure()` 记录 channel、reason、raw STATUS
  和 token 后 trap；NH 仅在判定为 TRAP 时打印 trap CSR 行，并在 DMA 记录非空
  时追加 DMA 上下文行。reason 取值见 `bosc_ame_dma.h`
  （1 通道非法、2 token 非法／缺少 launch、3 token 不匹配、4 bus_error、
  5 完成屏障后仍 busy）。
- **超时**：NH 打印 `TIMEOUT`，发布 `0xee`，仍然输出应用报告（可能全为初值）。
- **NH 自身 trap**：`nh_trap_entry` 跳到 `nh_e6_nh_trap()`，发布 `0xee`、
  打印 `NH TRAP`、转发 mailbox 后停机。

报告文本与迁移前的 `launcher.S` 逐字节一致（大小写、`\r\n`、字段顺序和
“`stats` 行与 `tr/acc size probe` 行之间没有换行”这一细节都保留）。

## BSS 与栈归属

- 镜像 `.bss`（含平台诊断单元和应用 `.bss.e6_diag`）由 **E6** 在
  `e6_start` 中清零，NH 不依赖其中任何初值；NH 流程只用栈和寄存器，平台
  模块也不在 `.bss` 保存跨核状态。
- 两核各用链接脚本给出的独立栈：`__nh_stack_top`、`__e6_stack_top`。
- E6 trap 入口不使用栈，因此栈损坏时仍能发布 `0xed` 和 CSR 上下文。

## 验证

无需硬件即可运行：

```sh
cc -std=c11 -Wall -Wextra -Werror \
   -I examples/FPGA-BOSCAME/common/runtime \
   -I examples/FPGA-BOSCAME/common/uart \
   -I examples/FPGA-BOSCAME/common/nh_e6 \
   examples/FPGA-BOSCAME/common/nh_e6/test_nh_e6_platform.c \
   -o /tmp/nh_e6_platform_test && /tmp/nh_e6_platform_test
# nh_e6_platform: all checks passed

python3 -m unittest discover examples/FPGA-BOSCAME/tools -p 'test_nh_e6_platform.py'
```

host 测试覆盖：PMP／`pmpcfg0` 的写入值与顺序、`fence`／`satp=0`／
`sfence.vma`，`nh_e6_e6_main()` 在调用应用 `e6_main()` 之前完成这些设置，
CSR `0xbc0` 的两个策略值与前置 fence，释放序列的寄存器写入与 fence 顺序、
进度只打印新标记、PASS/FAIL/E6 TRAP/TIMEOUT 判定与最终发布、trap CSR 行、
DMA 失败行的出现条件、每次结束都调用应用报告、E6 完成码与 mailbox 转发、
`init_uart(5529600, 115200)` 的除数与线路控制、`nh_e6_publish_dma_failure`
先发布再 trap。测试通过 `NH_E6_MMIO_READ32`／`WRITE32`、`NH_E6_CSR_WRITE`／
`NH_E6_SATP_WRITE_BARE`、`NH_E6_DDR_WINDOW_*` 与 `NH_E6_*_FENCE` 覆盖钩子
驱动假寄存器（与 `bosc_ame_dma.c` 相同的做法）；真实 MMIO 字符路径与真实
CSR 指令不在 host 上运行，由 RISC-V 反汇编确认。

它不能替代板上验证：AME 指令、PMP、CSR `0xbc0` 和 SYSCTRL 的真实时序只在
FPGA 上成立。

## 其他样例

`vit-base-patch16-224/linear_196x768x768_ame_dma_spm/` 仍保留自己的
`launcher.S`（同样是 NH/E6 协议，但诊断字段和消息不同），本次拆分没有改动
它；如需迁移，应复用本目录，并在其 `link.ld` 中导出 DDR staging 窗口符号
（`__nh_e6_ddr_window_start`／`__nh_e6_ddr_window_end`），再单独做板上复测。
