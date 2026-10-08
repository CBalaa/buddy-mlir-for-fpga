# EfficientNet-B3：linalg 算子到 NR FPGA

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE)
(same text as the repository-root `LICENSE`).
SPDX-License-Identifier: Apache-2.0.

Checkpoint weights are **not** in this tree; fetch them locally and keep
them out of pull requests.

对照 `../qwen3-0.6b`，适配 [`timm/efficientnet_b3.ra2_in1k`](https://hf-mirror.com/timm/efficientnet_b3.ra2_in1k)。
结构见 [MODEL.md](MODEL.md)。与 Qwen 一样有三条路径：

```text
1. 每个算子目录的 kernel.mlir + launch.c，经 common.mk 降到 NR 并上板
2. 同一批算子的 Triton @triton.jit 实现，接入同一套 Buddy NR 后端
3. Buddy DynamoCompiler 捕获整图，替换为已编译的 Triton 调用后再上板
```

公共 NR runtime、UART、上板脚本、Triton 安装都复用 `../common` 和
`../fpga_run.sh`。`support.h` / `common.mk` /
`tools/{host_main.c,vectorize_nr.py,build_suite.py}` 与 Qwen 样例同一套 NR 配方。

## 0. 环境

```bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"   # 若已在仓库根，直接用 $PWD
EFF="$REPO/examples/FPGA-BOSCAME/efficientnet_b3.ra2_in1k"

# conda activate boscame
source "$EFF/env.sh"
```

`env.sh` 把 `BUDDY_PYTHON` / `TRITON_PYTHON` 都指到 `boscame` 3.12，并把
`PYTHONPATH` 接到本仓库 `build-migrate/python_packages`（不要用旁边 sibling
Buddy 的包）。图导入还需要同一解释器里的 `torch`、`timm`、`safetensors`。

## 1. linalg 算子（与 Qwen 相同入口）

```bash
python3 -B "$EFF/tools/generate_cases.py"   # 写出 kernel.mlir + launch.c

cd "$EFF"
make -C add_1x256 check
make -C conv2d_stem_tiny all check
make -j4 all
make check
make suite GROUP=smoke
make run GROUP=smoke FPGA=5 CAPTURE_SECONDS=3600
```

组名：`aux`（逐元素 / GAP / SE）、`conv`、`linear`、`smoke`、`all`。
必须看到 `verify <case>: PASS` 和 `[nr] RA returned: PASS`。

## 1b. Triton 前端

```bash
cd "$EFF/triton"
"$TRITON_PYTHON" export.py --list
make check CASE=conv2d_stem_tiny
make check GROUP=smoke JOBS=2
make run CASE=conv2d_stem_tiny FPGA=5 CAPTURE_SECONDS=900
```

## 2. 捕获计算图

```bash
MODEL="$EFF/model"
"$BUDDY_PYTHON" -B "$MODEL/tools/fetch_checkpoint.py" --output "$MODEL/assets/checkpoint"

# 先截断 1 个 stage，确认直方图
"$BUDDY_PYTHON" -B "$MODEL/tools/import_model.py" \
  --checkpoint "$MODEL/assets/checkpoint" \
  --output "$MODEL/build/import-stage0" --stages 1

# 整模
"$BUDDY_PYTHON" -B "$MODEL/tools/import_model.py" \
  --checkpoint "$MODEL/assets/checkpoint" \
  --output "$MODEL/build/import-full"
```

看 `import-report.json` 的 `histogram`。替换只能按真实 dataflow 匹配
`Conv2dOp` / `Mul(Sigmoid)` / `AdaptiveAvgPool2d` / `AddMM` 等。

## 3. Triton 编译、替换节点，lowering

整模真实形状 case 已提交在 `$EFF/model-cases/`（约 96 个）。默认直接用，不必
每次 emit。图变更后刷新：

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/emit_model_cases.py" \
  --ops "$MODEL/build/fp32-full/import/raw-ops.json" \
  --output "$EFF/model-cases"
# 或 REEMIT=1 bash model/tools/run_full.sh
```

```bash
"$TRITON_PYTHON" -B "$EFF/triton/build.py" --jobs 2
"$TRITON_PYTHON" -B "$EFF/triton/build.py" --jobs 2 --host

"$BUDDY_PYTHON" -B "$MODEL/tools/triton_call_replace.py" \
  --checkpoint "$MODEL/assets/checkpoint" \
  --triton-build "$EFF/triton/build" \
  --output "$MODEL/build/replacement" --stages 1

"$BUDDY_PYTHON" -B "$MODEL/tools/host_reference.py" \
  --checkpoint "$MODEL/assets/checkpoint" \
  --output "$MODEL/build/fp32-reference" --stages 1

"$BUDDY_PYTHON" -B "$MODEL/tools/lower_model_nr.py" \
  --checkpoint "$MODEL/assets/checkpoint" \
  --triton-build "$EFF/triton/build" \
  --output "$MODEL/build/nr-stage0" --stages 1
```

`triton-call-replacement.json` 里 `uncovered` 必须先清掉，再把 `--stages 1`
换成整模。

## 4. 整模上板

整模流水线与 Qwen `model/` 同构（FP32 视觉版，无 tokenizer/KV）。详见
[model/README.md](model/README.md)。

一条命令走完：导入图、按真实形状编 Triton、替换、lowering、链 NR 镜像、打 DDR 段。

```bash
cd "$EFF"
source ./env.sh
# 只编镜像
bash model/tools/run_full.sh
# 编完立刻上板（默认 FPGA=5）
RUN_FPGA=1 FPGA=5 CAPTURE_SECONDS=1800 bash model/tools/run_full.sh
```

已有 `model/build/fp32-full/prepared`（或旧路径 `model/build/full/prepared`）时只上板：

```bash
"$EFF/model/tools/run_model.sh" "$EFF/model/build/fp32-full/prepared" \
  --fpga=5 --capture-seconds=1800 --startup-timeout=900
```

成功时 UART 里要有 `verify efficientnet_b3.ra2_in1k: PASS` 和
`[nr] RA returned: PASS`。权重在独立 DDR 段 `weights.bin`，输入在板上按与
`host_reference.py --board-input` 相同的公式生成。

板上结果归档：

```bash
"$BUDDY_PYTHON" -B "$EFF/model/tools/archive_model_run.py" \
  --run "$REPO/examples/FPGA-BOSCAME/build/fpga-runs/run-XXXXXXXX" \
  --prepared "$EFF/model/build/fp32-full/prepared" \
  --reference "$EFF/model/build/fp32-full/fp32-reference/host-reference.json" \
  --output "$EFF/model/validation/board/fp32-full-run"
```

截断 1 个 stage 的冒烟：`bash model/tools/run_stage0.sh`。
整模默认 `CONFIG=fp32-full`；也可 `CONFIG=ame-v05/fp32-full`（与 Qwen
`build/<config>` 同构）。上板前可选主机 EE：`make -C model host-run`
（见 model/README.md）。
## 目录

```text
efficientnet_b3.ra2_in1k/
├── LICENSE                     # Apache-2.0，指向仓库根 LICENSE
├── tools/generate_cases.py     # kernel.mlir + metadata.json + launch.c
├── triton/{kernels,cases,export,build}.py
├── conv2d_*/ silu_*/ ...       # smoke/代表形状（generate_cases.py）
├── model-cases/                # 整模静态全集（emit_model_cases.py，可提交）
└── model/                      # 整模：import → replace → image → archive
    ├── README.md
    └── tools/                  # 与 Qwen model/tools 同构的 FP32 视觉版
```

生成出的 case `makefile` 使用相对路径（`include ../common.mk` 或 override ROOT），可以进 PR。
`model/build/`、`triton/build/`、`model/assets/` 不要提交。

## 许可证

源文件带 SPDX `Apache-2.0` 头，与 buddy-mlir 其余部分一致。上游
`timm/efficientnet_b3.ra2_in1k` 权重另行授权，不随本目录分发。
