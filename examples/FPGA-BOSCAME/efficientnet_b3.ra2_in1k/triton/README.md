# EfficientNet-B3 Triton 算子

Licensed under the Apache License, Version 2.0. See [../LICENSE](../LICENSE).
SPDX-License-Identifier: Apache-2.0.

真实 `@triton.jit` 内核，经 `triton-riscv` 生成 linalg 后接入与上一层
`kernel.mlir` 相同的 Buddy NR lowering。复用各算子目录里已有的 `launch.c`。

```text
kernels.py 中的 @triton.jit
  → Triton ASTSource / CPUBackend.make_ttir
  → kernel.ttir
  → triton-shared-opt
  → kernel.linalg.mlir
  → Buddy bufferization、NR 降低
  → 原 launch.c + ABI adapter + 公共 NR runtime
```

## 准备

```bash
source ../../common/triton/triton-env.sh
"$TRITON_PYTHON" export.py --list
```

图上的 case 和分类器 W8A8 都在上一级，和本目录同级。W8A8 三份由
`../tools/generate_matmul.py` 和 `../tools/generate_quant.py` 写出。

## 构建和运行

```bash
make export CASE=silu_1x6x1x1
make check CASE=silu_1x6x1x1
make check GROUP=smoke JOBS=2
make run CASE=silu_1x6x1x1 FPGA=5
make run GROUP=smoke FPGA=5
```

分组：`smoke`、`conv`、`linear`、`aux`、`all`。
`cases.py` 搜索上一级的 case 目录。额外目录仍可通过 `EFFNET_CASE_ROOTS` 注入。
