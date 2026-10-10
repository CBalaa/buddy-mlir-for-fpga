# EfficientNet-B3：FPGA 适配

已实现 9 类算子、99 个静态 case。每个目录都是 `1×3×288×288` 图里会出现的形状，带 `kernel.mlir`、metadata 和独立 C oracle，可做 Host 检查和 NR ELF/BIN 构建。同一批 case 有 Triton 前端。形状说明见 [MODEL.md](MODEL.md)。

96 个 FP32 形状和分类器的三个 W8A8（`quantize_1x1536`、`matmul_1x1000x1536`、`dequantize_1x1000`）都在本目录，和 `tools/`、`triton/` 同级。没有缩小尺寸的样例。

| 算子 | case 数 | 说明 |
|---|---:|---|
| add | 7 | 残差相加 |
| silu | 20 | MBConv / SE 激活 |
| gap | 13 | SE squeeze 与 head |
| linear_f32 | 1 | classifier `1×1000×1536` |
| conv2d_f32 | 41 | stem、expand、project |
| dwconv2d_f32 | 14 | depthwise |
| matmul_i8 | 1 | 同一条 classifier 的 W8A8 `i8×i8→i32` |
| per_token_quantization | 1 | `quantize_1x1536` |
| dequantization | 1 | `dequantize_1x1000` |

整模与 ViT-Base 那次提交同一条链路，脚本在 [model](model/README.md)：Buddy 捕获 `1×3×288×288` 的计算图，按形状把图节点替换成上面的 Triton 调用，再 lowering 成 NR 镜像上板。默认 FP32。没有 tokenizer、KV 和 attention。

Host：当前 smoke 是图上的 `silu_1x6x1x1`、`conv2d_1x1_6x24_k1_s1`、`linear_1x1000x1536`，以及分类器的三个 W8A8 case。缩小尺寸那一批删掉之后，旧的 11 个 smoke Host 日志不再对应当前目录。这 99 个 case 没有新的 Host 归档。

板上：没有逐 case 的 UART 入库。本机有一次整模前向跑完并打出 argmax，随后 top-logit 逐位比较失败（`verify efficientnet_b3.ra2_in1k: FAIL`）。该 UART 留在 `examples/FPGA-BOSCAME/build/fpga-runs/`，属于构建产物，不入库。

权重、checkpoint 和 `build/` 不入库。没有改 Qwen3 或公共编译/运行工具。

## 实现

- 图上形状：本目录下的 case，每个含 `kernel.mlir`、`launch.c`、`metadata.json`、`makefile`
- 分类器 W8A8：`tools/generate_matmul.py`、`tools/generate_quant.py`。`make generate` 只刷新这三份，并给 `case-index.json` 重建索引，不会再写出缩小尺寸的目录
- Triton：[`triton/`](triton/README.md)，复用各 case 的 `launch.c`
- 公共 NR runtime 和上板脚本仍用 `examples/FPGA-BOSCAME/common` 与 `fpga_run.sh`
- 整模脚本在 [`tools/`](tools/)，说明在 [`model/README.md`](model/README.md)。入口是 `bash tools/run_full.sh`

## 整模

在仓库根目录，先 `source` 本目录的 `env.sh`。算子 Triton 镜像和计算图是两条产物，输出目录分开。

```bash
EFF=examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k
source "$EFF/env.sh"
# 只编整模镜像，默认写到 model/build/fp32-full/
bash "$EFF/tools/run_full.sh"
# 编完立刻上板
RUN_FPGA=1 FPGA=5 CAPTURE_SECONDS=1800 bash "$EFF/tools/run_full.sh"
```

已有 `model/build/fp32-full/prepared` 时只上板：

```bash
"$EFF/tools/run_model.sh" "$EFF/model/build/fp32-full/prepared" \
  --fpga=5 --capture-seconds=1800 --startup-timeout=900
```

成功时 UART 里要有 `verify efficientnet_b3.ra2_in1k: PASS` 和 `[nr] RA returned: PASS`。分步脚本都在 `tools/`，说明见 [model/README.md](model/README.md)。

## Host / 上板

在本目录：

```bash
source ./env.sh
make -C silu_1x6x1x1 check
make check
cd triton && make check GROUP=smoke JOBS=4
```

`make` 和 `make check` 会走过本目录每一个 case。单个 case 上板：`make -C silu_1x6x1x1 run FPGA=5`。成功时 UART 里要有 `verify <case>: PASS` 和 `[nr] RA returned: PASS`。组跑默认是 smoke：`make run FPGA=5`。`GROUP=all` 会把 99 个 case 链进同一张 suite。

记录一次完整 suite 时，把 `run-ACTUAL` 换成 `fpga_run.sh` 打印的目录：

```bash
python3 tools/record_run.py build/suite-smoke/manifest.json \
  ../build/fpga-runs/run-ACTUAL --output validation/smoke
```

本目录目前没有这样的归档。
