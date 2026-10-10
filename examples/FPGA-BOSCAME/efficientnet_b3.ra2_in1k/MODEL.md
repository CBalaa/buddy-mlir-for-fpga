# EfficientNet-B3 (`timm/efficientnet_b3.ra2_in1k`) 算子与形状

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
SPDX-License-Identifier: Apache-2.0.

本目录是 `timm/efficientnet_b3.ra2_in1k` 的适配：静态 `kernel.mlir`、
同一批算子的 Triton 前端、每个 case 的独立 C oracle，以及整模计算图。
整模与 `vit-base-patch16-224` 一样，走捕获图、按形状替换成 Triton 调用、
lowering、再链 NR 镜像。入口在 [model/README.md](model/README.md)。权重不入库。

- 权重：https://hf-mirror.com/timm/efficientnet_b3.ra2_in1k
- 输入：`1×3×288×288`，ImageNet-1k，1000 类
- 激活：SiLU；推理前把 Conv+BN 融成带 bias 的卷积
- Buddy 图张量布局：**NCHW**（`Conv2dOp._layout = NCHW_FCHW`）

## 结构参数（width=1.2, depth=1.4）

| Stage | 类型 | repeats | k | stride | e | C_out | 特征图 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| stem | Conv 3×3 | 1 | 3 | 2 | - | 40 | 144 |
| 0 | DS e=1 | 2 | 3 | 1 | 1 | 24 | 144 |
| 1 | IR | 3 | 3 | 2 | 6 | 32 | 72 |
| 2 | IR | 3 | 5 | 2 | 6 | 48 | 36 |
| 3 | IR | 5 | 3 | 2 | 6 | 96 | 18 |
| 4 | IR | 5 | 5 | 1 | 6 | 136 | 18 |
| 5 | IR | 6 | 5 | 2 | 6 | 232 | 9 |
| 6 | IR | 2 | 3 | 1 | 6 | 384 | 9 |
| head | Conv 1×1 + GAP + FC | 1 | 1 | 1 | - | 1536→1000 | 9→1 |

一个 MBConv：`1×1 expand`（e≠1）→ depthwise k×k → SE（GAP + Linear + SiLU +
Linear + Sigmoid + channel mul）→ `1×1 project` → 残差（stride=1 且通道相同）。

## 本目录覆盖的算子种类

| kind | linalg / Triton kernel | 说明 |
| --- | --- | --- |
| `conv2d_f32` | `conv2d_nchw` | im2col tile + `tl.dot`，FCHW 权重 + bias |
| `dwconv2d_f32` | `dwconv2d_nchw` | 按通道空间卷积，走 RA，不进 AME |
| `linear_f32` | `linear` | classifier / SE 的 `A[M,K] * W[N,K]^T + b` |
| `matmul_i8` | `matmul(INTEGER)` | W8A8 AME：`i8×i8→i32`（无 bias） |
| `per_token_quantization` / `dequantization` | `quantize` / `dequantize` | W8A8 激活量化与反量化 |
| `silu` / `sigmoid` / `add` / `mul` | 逐元素 | SE 与残差 |
| `gap` | 全局平均池化 | SE squeeze 与 head |
| `se_scale` | 通道广播乘法 | SE excite |

`quantize_*` / `matmul_i8` / `dequantize_*` 是分类器和 SE linear 的 W8A8
微基准，和 FP32 算子放在同一套 case 目录里。生成入口：`make generate`。

## 静态样例

`tools/generate_cases.py`、`tools/generate_matmul.py`、`tools/generate_quant.py`
写出代表形状。每个目录含 `kernel.mlir`、`launch.c`（独立 C oracle）、
`metadata.json` 和 `makefile`。图上的真实形状闭集在 `model-cases/`（96 个），
已随仓库提交。
