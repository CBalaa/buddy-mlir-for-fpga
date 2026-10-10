# model-cases — EfficientNet-B3 图上静态 case

96 个真实形状，对应输入 `1×3×288×288` 的 EfficientNet-B3 图。每个子目录含 `kernel.mlir`、`launch.c`（独立 C oracle）、`metadata.json` 和 `makefile`。

`triton/cases.py` 会把本目录加入搜索根。代表形状和 W8A8 微基准在上一级目录，不在这里。
