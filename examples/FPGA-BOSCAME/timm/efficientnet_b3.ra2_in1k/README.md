# EfficientNet-B3：FPGA 适配

已实现 13 类算子、143 个静态 case 目录：代表形状和 `model-cases/` 里的图上真实形状都带 `kernel.mlir`、metadata 和独立 C oracle，可做 Host 检查和 NR ELF/BIN 构建。同一批 case 有 Triton 前端。形状说明见 [MODEL.md](MODEL.md)。

143 个目录里，96 个是输入 `1×3×288×288` 的图上形状，47 个是代表形状和 W8A8 微基准。其中 6 个名字在两处都有一份。

| 算子 | case 数 | 说明 |
|---|---:|---|
| add | 9 | 残差相加 |
| silu | 23 | MBConv / SE 激活 |
| sigmoid | 2 | SE excite |
| mul | 1 | 逐元素乘法 |
| se_scale | 2 | SE 通道广播 |
| gap | 16 | SE squeeze 与 head |
| linear_f32 | 5 | classifier / SE，FP32 |
| conv2d_f32 | 46 | stem、expand、project |
| dwconv2d_f32 | 17 | depthwise |
| matmul_f32 | 6 | FP32 matmul 微基准 |
| matmul_i8 | 6 | W8A8 `i8×i8→i32` |
| per_token_quantization | 5 | 激活量化 |
| dequantization | 5 | 反量化 |

整模与 ViT-Base 那次提交同一条链路，脚本在 [model](model/README.md)：Buddy 捕获 `1×3×288×288` 的计算图，按形状把图节点替换成上面的 Triton 调用，再 lowering 成 NR 镜像上板。默认 FP32。没有 tokenizer、KV 和 attention。

Host：本机 Triton smoke 的 11 个 case（`case-index.json` 的 smoke 列表）检查通过。该日志在被忽略的 `triton/build/` 里，不入库。其余 case 没有归档的 Host 结果，不能记成 143 个全部通过。

板上：没有逐 case 的 UART 入库。本机有一次整模前向跑完并打出 argmax，随后 top-logit 逐位比较失败（`verify efficientnet_b3.ra2_in1k: FAIL`）。该 UART 留在 `examples/FPGA-BOSCAME/build/fpga-runs/`，属于构建产物，不入库。

权重、checkpoint 和 `build/` 不入库。没有改 Qwen3 或公共编译/运行工具。

## 实现

- 代表形状：`tools/generate_cases.py`、`tools/generate_matmul.py`、`tools/generate_quant.py`
- 图上形状：已提交的 [model-cases](model-cases/README.md)
- Triton：[`triton/`](triton/README.md)，复用各 case 的 `launch.c`
- 公共 NR runtime 和上板脚本仍用 `examples/FPGA-BOSCAME/common` 与 `fpga_run.sh`
- 整模：[`model/`](model/README.md)，`make -C model help`

## 整模

在仓库根目录，先 `source` 本目录的 `env.sh`。算子 Triton 镜像和计算图是两条产物，输出目录分开。

```bash
EFF=examples/FPGA-BOSCAME/timm/efficientnet_b3.ra2_in1k
source "$EFF/env.sh"
# 只编整模镜像，默认写到 model/build/fp32-full/
bash "$EFF/model/tools/run_full.sh"
# 编完立刻上板
RUN_FPGA=1 FPGA=5 CAPTURE_SECONDS=1800 bash "$EFF/model/tools/run_full.sh"
```

已有 `model/build/fp32-full/prepared` 时只上板：

```bash
"$EFF/model/tools/run_model.sh" "$EFF/model/build/fp32-full/prepared" \
  --fpga=5 --capture-seconds=1800 --startup-timeout=900
```

成功时 UART 里要有 `verify efficientnet_b3.ra2_in1k: PASS` 和 `[nr] RA returned: PASS`。分步目标（`checkpoint`、`import`、`replace`、`lower`、`image`、`run`）见 `make -C model help`。

## Host / 上板

在本目录：

```bash
source ./env.sh
make -C add_1x256 check
make check
cd triton && make check GROUP=smoke JOBS=4
```

单个 case 上板：`make -C add_1x256 run FPGA=5`。成功时 UART 里要有 `verify <case>: PASS` 和 `[nr] RA returned: PASS`。组跑：`make run GROUP=smoke FPGA=5`。

记录一次完整 suite 时，把 `run-ACTUAL` 换成 `fpga_run.sh` 打印的目录：

```bash
python3 tools/record_run.py build/suite-smoke/manifest.json \
  ../../build/fpga-runs/run-ACTUAL --output validation/smoke
```

本目录目前没有这样的归档。
