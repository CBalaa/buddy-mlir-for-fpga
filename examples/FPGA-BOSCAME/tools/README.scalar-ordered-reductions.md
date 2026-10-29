# FPGA 有序 f32 规约的临时标量绕过

2026-10-28：正式 LayerNorm 标量路径已通过恢复后的 FPGA4 验收。
原始最小样例中，两个 1..16 的顺序求和应为 272，实际第二次调用可返回 136；
这是可精确表示的整数和，不能解释成普通浮点舍入误差。RTL 根因仍未确定。

## 实现

FPGA Makefile 默认向 lower-bosc-ame 传入
scalar-f32-ordered-reductions=true：在 MLIR SSA 层保留原标量 accumulator，
用 vector.store 保存输入向量，随后用顺序循环逐 lane 执行 flw / fadd.s。
seed 和结果始终是标量，不经过 vfmv.s.f / vfmv.f.s；所有 lane（含补零）仍参与加法。
函数入口只分配一次共享的有界 scratch；16 lane 时为64字节，不随循环次数增长。
快照访问降低为 volatile LLVM store/load，防止优化重新产生向量提取或规约。
其余向量运算保留。该实现沿用现有非 strict MLIR/LLVM 浮点契约，
不额外承诺动态浮点环境语义。

支持无 fastmath、显式 f32 seed、固定一维 f32 ADD 和顺序 func/SCF 控制流。
此类规约若是 scalable、masked、无 seed 或位于并发/未知 region，会明确报错。
其他类型、规约种类及带 fastmath 的形式保留原 lowering；最终 ISA 仍须通过 NR 审计。
编译 pass 的选项默认 false，现有非 FPGA 调用不改变；FPGA 构建策略默认开启。

正式构建使用 --reject-vector-ordered-reductions，拒绝残留 vfredosum.vs；
完整 ELF 审计再按机器码检查一次。不要仅依赖旧汇编展开：其简单求和可以通过，
但完整 LayerNorm 已出现超时/数值错误；去掉 vfredosum 本身不足以保证板端正确。
Python transform 的旧默认展开只保留作诊断及 ISA 语义回归，不是正式绕过入口。

## 使用与复现

正常执行样例的 make 即可，无需 clean。仅诊断原指令时：

~~~bash
make ALLOW_VECTOR_ORDERED_REDUCTIONS=1
# 恢复正式标量路径
make ALLOW_VECTOR_ORDERED_REDUCTIONS=0
~~~

开关统一控制 SSA lowering、汇编策略和 ELF 审计，策略变化会使 LLVM 阶段起重新构建。
单独调用编译器使用：

~~~bash
buddy-opt kernel.boscame.mlir --lower-bosc-ame="scalar-f32-ordered-reductions=true" -o kernel.llvm.mlir
~~~

主仓库根目录中按现有环境构建、上板：

~~~bash
conda activate boscame
export BUDDY_ROOT="$PWD/buddy-mlir"
export BUDDY_MLIR_BINARY_DIR="$BUDDY_ROOT/build/bin"
export LLVM_BIN="$BUDDY_ROOT/llvm/build/bin"
make -C examples/BOSCAMEFPGA/layernorm_197x768 check simcheck
python examples/BOSCAMEFPGA/run_bin.py examples/BOSCAMEFPGA/layernorm_197x768/build/layernorm_197x768/layernorm_197x768.bin
~~~

上板始终使用现有 run_bin.py 和调度队列；复用健康 SSH 隧道。
120秒为执行期限，排队另算；保留任务 ID、镜像 SHA256、UART 和原任务结果。

## 验证结果

| 正式镜像 | 校验 | cycle |
| --- | --- | --- |
| 主 LayerNorm 197×768，2轮 | 302592/302592 逐位相同，errors=0、guards=0 | 9851570 / 9847844 |
| ViT LayerNorm 197×768 | 151296 个输出，errors=0，最大绝对误差2^-22 | 10777140 |
| ViT LayerNorm 1×768 | 768 个输出，errors=0，最大绝对误差2^-22 | 55565 |

主镜像 SHA256 为58fd9d39faa5124c8b7348a79345144460899651ea7c8233ae4f953dd05de04c。
以上 cycle 按实际计数报告，服务未提供实测频率，不根据 UART 的硬编码频率标签换算时间。
已有镜像在板端执行完整 oracle 比较，未声明 DDR output，因此不声称下载过输出文件。

新 SSA 回归直接执行生成的主机 LLVM，覆盖159组顺序 f32 语义、补零、正负零、
NaN/Inf、抵消、非零 seed、17 lane、重复循环和内存 guards；并检查最终 RVV 无
vfredosum 和 seed/result 的 vfmv 中转。旧汇编解释器6400组合仅证明其 ISA 语义，
不能代替硬件验收。历史边界镜像88项没有完成，不计为通过。
原始故障复现见[standalone README](../../../reference/rvv_reduction_probe/standalone/README.md)，
任务日志和历史失败见[验证报告](../../../reference/rvv_reduction_probe/scalar-workaround/README.md)。
