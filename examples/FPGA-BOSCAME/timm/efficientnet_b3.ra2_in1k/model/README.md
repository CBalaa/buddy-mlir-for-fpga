# EfficientNet-B3：Buddy 计算图 + Triton 静态库 → NR FPGA

Licensed under the Apache License, Version 2.0. See [../LICENSE](../LICENSE).
SPDX-License-Identifier: Apache-2.0.

默认部署为 **完整 EfficientNet-B3（`timm/efficientnet_b3.ra2_in1k`）、batch=1、
输入 `1×3×288×288`、ImageNet-1k 1000 类、FP32**。板上用与
`host_reference.py --board-input` 相同的确定性 NCHW 填充，比较 argmax 与
top-logit 位型。

结构与算子形状见 [../MODEL.md](../MODEL.md)。公共 NR runtime / 上板脚本见
`../../../common/nr` 与 `../../../fpga_run.sh`。

整模阶段与 `vit-base-patch16-224` 相同：Buddy 捕获真实计算图，按形状把图节点替换成已有 Triton 调用，再 lowering 成 NR 镜像上板。**FP32** 为默认路径。分类器和 SE linear 另有 W8A8 工具（`quant_reference` / `build_nr_w8a8_*` / `triton_call_replace --w8a8`）。视觉模型没有 tokenizer、KV 和 attention：

```text
fetch → import → inventory → emit cases → Triton host+NR
  → replace [--w8a8] → host reference → (optional host EE)
  → lower → archive kernels → image → prepare → FPGA → archive run
```

W8A8 图替换示例（需根目录 `make generate` 且 Triton 已编好对应 cases）：

```bash
make -C model replace W8A8=1 TRITON_BUILD=../triton/build
# 随后用 build_nr_w8a8_segment.py / build_nr_w8a8_image.py 打包权重镜像
```

## 快速复跑已有镜像

本地若已有 `build/fp32-full/prepared`（或旧路径 `build/full/prepared`；
`deployment.json` 里 `execution: prepared only` 表示尚未归档板上结果）：

```bash
REPO="$(git rev-parse --show-toplevel)"
MODEL="$REPO/examples/FPGA-BOSCAME/timm/efficientnet_b3.ra2_in1k/model"
EFF="$REPO/examples/FPGA-BOSCAME/timm/efficientnet_b3.ra2_in1k"
source "$EFF/env.sh"
"$MODEL/tools/run_model.sh" "$MODEL/build/fp32-full/prepared" \
  --fpga=5 --capture-seconds=1800 --startup-timeout=900
```

成功时 UART 需同时出现：

```text
[model] argmax=... logit_bits=...
verify efficientnet_b3.ra2_in1k: PASS
[nr] RA returned: PASS
```

归档（把 `RUN` 换成 fpga_run 打印的目录）：

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/archive_model_run.py" \
  --run "$REPO/examples/FPGA-BOSCAME/build/fpga-runs/run-XXXXXXXX" \
  --prepared "$MODEL/build/fp32-full/prepared" \
  --reference "$MODEL/build/fp32-full/fp32-reference/host-reference.json" \
  --output "$MODEL/validation/board/fp32-full-$(date +%Y%m%d)"
```

## 从源码重建

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
EFF="$REPO/examples/FPGA-BOSCAME/timm/efficientnet_b3.ra2_in1k"
source "$EFF/env.sh"

# 截断 1 个 stage，先确认替换与镜像（快）→ model/build/stage0/
RUN_FPGA=0 bash "$EFF/model/tools/run_stage0.sh"

# 整模镜像（默认 CONFIG=fp32-full → model/build/fp32-full/）
bash "$EFF/model/tools/run_full.sh"

# 与 Qwen 相同的 build/<config>：例如板型前缀
# CONFIG=ame-v05/fp32-full bash "$EFF/model/tools/run_full.sh"
# 或 OUT=... 直接指定绝对路径

# 编完立刻上板
RUN_FPGA=1 FPGA=5 CAPTURE_SECONDS=1800 bash "$EFF/model/tools/run_full.sh"
```

或分步：

```bash
cd "$EFF/model"
make checkpoint
make import inventory
# 然后按 README 父目录第 3 节编 Triton / replace / lower / image
make -C .. -C triton check GROUP=smoke   # 算子侧
make full
```

## 主机 EE（上板前可选）

替换后的图在主机上通过 Buddy ExecutionEngine 跑一遍，核对 logits：

```bash
# 需要已有 replacement + triton host kernels + fp32-reference
make -C "$EFF/model" host-bridge \
  TRITON_BUILD="$EFF/model/build/fp32-full/triton-build" \
  REPLACE_DIR="$EFF/model/build/fp32-full/replacement"
make -C "$EFF/model" host-run \
  TRITON_BUILD="$EFF/model/build/fp32-full/triton-build" \
  REPLACE_DIR="$EFF/model/build/fp32-full/replacement" \
  REF_DIR="$EFF/model/build/fp32-full/fp32-reference" \
  CHECKPOINT="$EFF/model/assets/checkpoint"
```

## 与 Qwen model/ 的对应关系

| Qwen | EfficientNet | 说明 |
| --- | --- | --- |
| `build/ame-v05/model-28l-cap128` | `build/<CONFIG>`（默认 `fp32-full`） | 同构 `OUT`/`CONFIG`；可用 `CONFIG=ame-v05/fp32-full` |
| `import_model.py` | 同名 | timm + Conv+BN fuse，NCHW |
| `triton_call_replace.py` | 同名 | conv / dwconv / linear / silu / gap / se / add |
| `lower_model_nr.py` | 同名 | 图 → LLVM IR + `weights.bin` |
| `build_model_image.py` | 同名 | 链 NR 镜像，校验 argmax |
| `prepare_model_run.py` / `run_model.sh` | 同名 | DDR 段 + `fpga_run.sh` |
| `build_host_bridge.sh` / `run_graph_host.py` | 同名 | 上板前主机 EE |
| `archive_model_run.py` | 同名（分类版） | UART argmax / PASS 归档 |
| `quant_reference.py`, `build_weight_image.py`, W8A8 segments | 同名 | 部署工具链；整图 W8A8 板级待 FP32 通过后启用 |
| `qwen_profile_*` / `qwen_intermediate_*` | `effnet_profile_*` / `effnet_intermediate_*` | 本例自有运行时符号 |
| tokenizer / KV / attention | stub + README | LLM 专用，视觉 N/A |

## 产物目录

```text
model/build/fp32-full/     # run_full.sh 默认（CONFIG=fp32-full）
  import/ raw-ops.json
  triton-build/            # Triton NR + host
  replacement/             # adapters + replacement report
  fp32-reference/          # host-reference.json + logits.npy
  nr-forward/              # forward.ll + weights.bin
  model-lib/               # libeffnet_triton.a
  image/                   # effnet_model.elf/.bin
  prepared/                # image.bin + weights.bin + ddr-load.plan
model/build/stage0/        # run_stage0.sh 默认
model/build/<config>/      # CONFIG=ame-v05/fp32-full 等
../model-cases/            # 提交的整模静态 case 全集（run_* 默认使用）
model/validation/board/    # archive_model_run.py 归档（勿提交大二进制）
```

`build/`、`assets/`、板上原始 UART 大文件默认 gitignore。旧目录名 `build/full` 仍可用：
`CONFIG=full` 或 `OUT=$MODEL/build/full`。