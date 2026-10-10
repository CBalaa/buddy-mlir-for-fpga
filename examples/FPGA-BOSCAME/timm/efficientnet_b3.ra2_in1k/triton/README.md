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
source ../../../common/triton/triton-env.sh
"$TRITON_PYTHON" export.py --list
```

先在上一级执行 `python3 -B ../tools/generate_cases.py`，保证每个 case 目录里有
`kernel.mlir`、`metadata.json` 和 `launch.c`。

## 构建和运行

```bash
make export CASE=conv2d_stem_tiny
make check CASE=conv2d_stem_tiny
make check GROUP=smoke JOBS=2
make run CASE=conv2d_stem_tiny FPGA=5
make run GROUP=smoke FPGA=5
```

分组：`smoke`、`conv`、`linear`、`aux`、`all`。
整模形状 case 在已提交的 `../model-cases/`（`cases.py` 自动加入搜索根）；
额外目录仍可通过 `EFFNET_CASE_ROOTS` 注入。
