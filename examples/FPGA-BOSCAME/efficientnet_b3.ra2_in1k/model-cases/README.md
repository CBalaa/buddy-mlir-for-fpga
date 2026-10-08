# model-cases — EfficientNet-B3 整模静态 case 全集

由 `model/tools/emit_model_cases.py` 从整模 `raw-ops.json`（input 288）一次生成，
覆盖图替换所需的全部真实形状（约 96 个）。`run_full.sh` / `run_stage0.sh` 默认
直接使用本目录，不再每次 emit。

刷新（图或 emit 逻辑变更后）：

```bash
# 需要已有 import 产物
python3 -B model/tools/emit_model_cases.py \
  --ops model/build/fp32-full/import/raw-ops.json \
  --output model-cases
# 或
REEMIT=1 bash model/tools/run_full.sh
```

每个子目录含 `kernel.mlir`、`launch.c`、`metadata.json`、相对路径 `makefile`。
`triton/cases.py` 会自动把本目录加入搜索根。
