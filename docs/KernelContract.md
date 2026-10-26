# Kernel Contract v1：bufferized Linalg + 中性事实

本层是 **triton-riscv（或其他前端）与 buddy-mlir 之间的唯一 IR 契约**。
它回答“程序读写什么、谁依赖谁、哪些值来自外部、mask 的有效区域是什么”，
**不回答**“放到哪个 bank、用哪个 DMA channel、切几个 packet”——后者是
`bosc-ame-plan` 的职责。

- 示例：[kernel.contract.mlir](../tests/KernelContract/Inputs/kernel.contract.mlir)（196x768x768 kernel，`buddy-opt` 可解析）
- 命名空间：`contract.*`（discardable attributes + verifier；不引入独立 dialect）
- 版本：`contract.version = 1`

---

## 1. 设计原则

| # | 原则 | 推论 |
| --- | --- | --- |
| P1 | **Plan-sufficient，不是 plan-carrying** | 契约里出现的每个事实都必须能被 planner 用来做决策；契约里绝不出现 target 决策或 target 常量 |
| P2 | **SSA 优先** | shape/stride/root/offset 已由 memref 类型、`memref.reinterpret_cast`/`subview` 链表达，契约属性不重复记录；属性只补 SSA 表达不了的语义（mask 种类、other、origin、任务图、scope、loop 携带关系） |
| P3 | **按 region/scope 组织图，允许不完整** | 计算图以 scope 为单位组织并各自声明完备性；不要求枚举动态实例；unmodeled 部分用 `complete=false` + limitation 显式降级 |
| P4 | **两端可验证** | 前端 emit 后跑 verifier；planner 入口再跑一遍。契约是 ABI，不能靠“生成者自觉” |
| P5 | **中性命名** | 不使用 `bosc_ame.*`；`bosc_ame` 只出现在 buddy-mlir 内部的 planned/final IR |

---

## 2. 承载规则（属性挂在哪里）

MLIR 不能给任意 SSA 值挂属性，因此采用以下固定规则：

| 事实 | 承载位置 | 说明 |
| --- | --- | --- |
| module 级事实 | `builtin.module` 属性 | version/producer/ABI |
| 函数级事实 | `func.func` 属性 | grid、compute graph 汇总 |
| 函数参数事实 | function/block 参数属性 | program 轴/角色、外部数据 value id |
| **一个 op 写出的内容版本** | 该 op 的 `contract.task.output_values`（循环用 `scf.for` 的 `contract.value_ids`） | 版本由任务图定义，**与 SSA 结果无关**（copy/fill/matmul 没有 SSA 结果，见 §3.4） |
| **一个 op 的内存事件** | 该 op 的属性 | `contract.access`（load/store 边界） |
| **一个 op 的任务节点** | 该 op 的属性 | `contract.task`（含所属 scope） |
| **一个 op 的计算语义** | 该 op 的属性 | `contract.compute`、`contract.fill` |
| **循环携带关系/循环体 scope** | `scf.for` 的属性 | `contract.loop_carried_slots` + `contract.cf` + `contract.scope` |
| **分支互斥/arm scope** | `scf.if` 的属性 | `contract.cf` + `contract.scopes` |

**逻辑引用**：使用 `int64` value id（定义见 §3.4），verifier 做闭包/双向一致
检查。**已定**：长期使用整数 ID，不引入 SSA 引用（§9）。

---

## 2.1 快速索引

| 属性 | 载体 | 一句话 |
| --- | --- | --- |
| `contract.version` | module | schema 版本 |
| `contract.producer` | module | 前端标识 |
| `contract.abi` | module（driver 注入） | 调用方对齐/不 alias 承诺 |
| `contract.compute_graph` | function | 图汇总：根 scope、聚合完备性、scope 数 |
| `contract.grid` | function | 已知 program grid |
| `contract.program` | 函数参数 | program_id / num_programs 轴 |
| `contract.value_id` | memref 函数参数 | 外部数据在逻辑值图中的入口 id |
| `contract.buffer` | alloc/alloca | tile 角色/逻辑形状 |
| `contract.access` | load/store 边界 op | 边界、mask、other、origin |
| `contract.task` | compute/dataflow op | 任务图节点（含 scope） |
| `contract.compute` | dot/elementwise/reduce | 计算语义 |
| `contract.fill` | linalg.fill | padding 填充语义 |
| `contract.provenance` | op | origin/lineage 事实 |
| `contract.program_axes` | op | 地址/任务随 program 轴变化 |
| `contract.value_ids` / `contract.value_users` | `scf.for` / 定义版本的 task | 循环 iter/result 版本；反向用户（可选） |
| `contract.loop_carried_slots` | scf.for | 迭代携带值关系 |
| `contract.cf` | scf.if / scf.for | 统一分支/循环事实 |
| `contract.scopes` | scf.if | 两个 arm 的 scope 记录（含完备性） |
| `contract.scope` | scf.for | 循环体 scope 记录（含完备性） |

---

## 3. 属性总表

### 3.1 module 级

#### `contract.version`（i32，必需）
契约 schema 版本。当前为 `1`。planner 只接受认识的版本；未知大版本报错，
未知小字段忽略。

#### `contract.producer`（string，必需）
产出契约的前端标识（如 `"triton-riscv"`），仅用于诊断/报告。

#### `contract.abi`（dict，可选；由 pipeline driver 注入）

```mlir
contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
```

| 字段 | 语义 |
| --- | --- |
| `args_aligned_bytes` | 公开入口 memref 的 aligned pointer 按该正整数字节数对齐；用于 64 B 访问时，该数必须是 64 的正整数倍 |
| `args_disjoint` | 不同公开入口 memref 指向的 DDR 区间互不重叠 |

该 ABI 仅适用于本模块内可证明没有符号使用的入口函数；public/private
可见性不能替代入口证明。被 `func.call` 调用、通过 `func.constant` 取址或
存在其他符号引用的 helper 不继承入口参数的对齐或不重叠承诺，调用者可
将同一 actual buffer 或未对齐的 subview 传给多个形参。符号使用无法完整
分析时也不得借用此 ABI；私有 alloc 与其他 root 的独立性仍由存储身份证明。

对齐承诺表示地址可被该数整除，不是“任意更大的数字都更强”：例如 128
可推出 64 B 对齐，96 不能。上述示例的 64 B 承诺仍须结合实际 offset、
stride、tile 起点和尾块检查；不能据此跳过每次访问的地址合法性证明。

没有该属性时 planner **不得**使用 direct DDR 路径，也不得基于“参数互不
alias”做任何合并。

`args_aligned_bytes` 与 `args_disjoint` 不提供 allocation extent。对示例中的
unranked DDR 参数，调用方须保证入口描述符支持 kernel 的合法访问；前端须
按源 shape、grid 和 mask 构造有效 view。verifier 可检查实际搬运不超出该
view，不能仅凭上述 ABI 属性独立证明 view 落在真实外部分配内。若以后要
验证 allocation bounds，须另行引入可核验的 extent 事实。

---

### 3.2 function 级

#### `contract.compute_graph`（dict，必需）

```mlir
contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
                          complete = true, scope_count = 5 : i64,
                          source_lineage_complete = true}
```

| 字段 | 类型 | 语义 | verifier |
| --- | --- | --- | --- |
| `version` | i32 | 图 schema 版本，1 | 必须存在且 = 1 |
| `root_scope` | i64 | 函数体 scope 的 id（固定 0） | 必须 = 0 |
| `complete` | bool | **聚合完备性**：所有 scope complete 且 scope 树完整描述 | 必须等于“所有 scope complete”的 AND |
| `scope_count` | i64 | scope 总数（含 root） | 等于 1 + 实际 scope 记录数 |
| `source_lineage_complete` | bool | provenance 是否完整 | 只影响报告与 REG 融合合法性判断 |
| `limitation` | string，可选 | 不完整的原因 | 任一 scope `complete=false` 时建议填写 |

**完备性以 scope 为单位**（见 §3.10），函数级 `complete` 只是聚合，
planner 决策时看**相关任务所在 scope** 的 `complete`。

图完备性与来源完备性独立：没有 source origin 的标准 Linalg 也可有完整
内存依赖图；完整图不能推出 `source_lineage_complete=true`。反之，planner
删除物理 load 后，只要其 source 事件事实仍完整保留，来源完备性不应因此
降级。已由标准 IR 证明没有 load 来源的常量/fill operand 不算未知来源。

#### `contract.grid`（array<i32: 3>，可选）
已知 program grid 大小；`-1` 表示该轴未知。来源是 kernel 启动配置。

```mlir
func.func @...(...) attributes {contract.grid = array<i32: 7, 6, 1>, ...}
```

---

### 3.3 函数参数

#### `contract.program`（dict，arg 属性）

```mlir
%arg6: i32 {contract.program = {role = "program_id", axis = 0 : i32}}
```

| role | 语义 |
| --- | --- |
| `"num_programs"` | 该轴的程序总数（运行时值） |
| `"program_id"` | 该轴上当前程序 id（运行时值） |

#### `contract.value_id`（i64，memref 函数参数）
标记该外部 memref 的初始内容在逻辑值图中的 value id（图入口）。只有带
`contract.value_id` 的参数可以被任务作为 `input_values` 引用；普通标量参数
不产生逻辑值。

```mlir
%arg0: memref<*xi8> {contract.value_id = 0 : i64}
%arg1: memref<*xi8> {contract.value_id = 1 : i64}
%arg2: memref<*xi32> {contract.value_id = 2 : i64}
```

---

### 3.4 值/缓冲区事实

#### `contract.buffer`（dict，`memref.alloc`/`memref.alloca` 属性）

```mlir
%c_buf = memref.alloc() alignment = 64
  {contract.buffer = {role = "tile", kind = "accumulator",
                      logical_shape = array<i64: 32, 128>}} : memref<32x128xi32>
```

| 字段 | 取值 | 语义 |
| --- | --- | --- |
| `role` | `"tile"` \| `"scratch"` | tile = 矩阵/向量运算的数据块；scratch = 私有临时 |
| `kind` | `"accumulator"` \| `"input-a"` \| `"input-b"` \| `"temporary"` \| `"padding"` | 数据角色 |
| `logical_shape` | array<i64> | 逻辑 tile 形状；memref 类型可扁平（由 view 提供布局） |

**不包含** memory space/bank/offset/容量——这些是 planner 输出。
rank-0 reduction init/result 使用 `logical_shape = array<i64>`（空形状），
承载一个标量元素；v1 支持 rank 0–2，不把 rank-0 buffer 当作零字节 buffer。

#### 内容版本：`contract.task.output_values` / `contract.value_ids`

**核心概念**：契约里的“逻辑值”（value id）是**存储内容的一个版本**，不是 SSA
结果。buffer 形态的 `memref.copy`/`linalg.fill`/`linalg.matmul` 没有 SSA
结果，但它们写存储；每次写产生一个新版本。这样“逻辑值 / 承载它的
buffer root / 最终物理窗口”三者分离：value id 只描述内容版本，root 由
view chain 决定（§3.11），物理窗口是 planner 输出（`plan.buffer`）。

三类版本：

| 载体 | 语义 |
| --- | --- |
| **私有 root**（`memref.alloc`/`memref.alloca` 及其 view） | 版本 = 该 root 的整块内容快照；部分写继承旧版本未覆盖区域（隐含继承，不需要额外属性） |
| **外部输入**（memref 函数参数） | 版本 = 该参数在 program 入口处被访问区域的内容（`contract.value_id`）；只读，不允许被重新定义 |
| **外部输出**（store / 直接写外部 root） | **不产生版本**；写由 `contract.access {kind="store"}` 表达，store 的 `output_values` 必须为空 |

定义点只有三个：

1. memref 函数参数：`contract.value_id`（外部入口版本）；
2. **任务**：`contract.task.output_values`（该任务写出的私有 root 的新版本；
   顺序 = 被写 root 的 operand 顺序）；
3. 已完整建模的单 slot `scf.for`：
   `contract.value_ids = [iter_arg_value_id, result_value_id]`
   （循环携带版本与循环退出后的结果版本，状态转移见 §3.9）。

**规则**（verifier V21–V26）：

- 每个 value id 在全函数的**静态 IR** 中恰好定义一次、非负；所有显式引用
  必须可解析。循环体的同一静态定义描述每轮各自的动态内容，不表示不同
  迭代共享同一份不可变快照；跨迭代关系只由 loop slot 摘要表达；
- 写私有 root 的任务定义一个新版本；`input_values` 包含**实际内存读取**的
  版本，以及**部分写继承未覆盖内容**所需的旧版本（§3.11）。二者之外的
  写后写顺序只用 `predecessors` 表达，不把被完整覆盖的旧内容伪装成输入；
- v1 首次写刚 alloc 的 root 必须完整初始化 root，没有旧版本；alloc 本身
  不定义版本。部分边界 load 必须先 fill 整个 root，再 copy 有效 subview；
- 终端外部写（store）不定义版本，`output_values` 为空；
- 允许存在未被读取的版本（只写不读的临时区），verifier 在报告里列出。

**示例（196 kernel）**：0/1/2 是三个外部输入的入口版本；t0 写出 C buffer
版本 3；then 循环的 iter/result 是 7/8（`scf.for.value_ids`），t3 每轮写出
版本 6；store 写外部 C、不产生版本。全函数共 17 个版本。

#### `contract.value_users`（array<array<i64>>，可选）
每个版本被哪些任务读取或继承的反向依赖边。可由 verifier 从
`contract.task.input_values` **加上 loop 的语义读取/继承关系**重建，因此可选；
出现时必须一致。slot consumer 列表还可能包含完整 kill 所需的边界 WAW
排序任务（§3.9），不能把这些任务直接加入该版本的反向用户集合。
它不区分真实内存读取与部分写继承；WAR/原地安全必须按 §3.11 的实际访问
集合派生，不能把所有反向依赖都视为覆盖区间上的读者。示例省略。

---

### 3.5 内存边界事实

#### `contract.access`（dict，load/store 边界 op 属性）

```mlir
// 满块 load
memref.copy %c_view, %c_buf {contract.access = {
  kind = "load", boundary = "external_ddr", origin_id = 0 : i64,
  mask = {kind = "none"}, other = 0 : i32}}

// tail：只前 4 行有效
memref.copy %c_src4, %c_dst4 {contract.access = {
  kind = "load", boundary = "external_ddr", origin_id = 0 : i64,
  mask = {kind = "rectangular", valid_shape = array<i64: 4, 128>},
  other = 0 : i32}}

// store（tail）
memref.copy %res4, %c_src4 {contract.access = {
  kind = "store", boundary = "external_ddr", origin_id = 4 : i64,
  mask = {kind = "rectangular", valid_shape = array<i64: 4, 128>}}}
```

| 字段 | 语义 |
| --- | --- |
| `kind` | `"load"` \| `"store"` |
| `boundary` | 全函数外部可见的 DDR 边界（未来可扩展 `"spm_view"`） |
| `origin_id` | 可选的前端 source event id（如 `tt.load`/`tt.store` 来源编号）；无 source 事件时不编造 ID |
| `mask.kind` | `"none"` / `"rectangular"`（静态）/ `"opaque"` |
| `mask.valid_shape` | 静态有效形状 |
| `mask.dynamic` | v2 预留；**v1 由 verifier 拒绝**（§4.3/V16） |
| `other` | mask 排除 lane 的填充值（load 专属；unmasked 可省略） |

（动态 mask 的定义与 v2 规则见 §4；本 kernel 全部是 none/static。）

---

### 3.6 任务图事实

#### `contract.task`（dict，每个 compute/dataflow 节点 op 属性）

```mlir
linalg.matmul {contract.task = {
  id = 3 : i64, kind = "dot", scope = 3 : i64, stage = 1 : i64,
  predecessors = array<i64: 0, 1, 2>,
  input_values = array<i64: 4, 5, 7>, output_values = array<i64: 6>,
  barrier_after = true, rematerializable = false, origin_id = 3 : i64}, ...}
```

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `id` | i64 | 函数内唯一任务 ID；必须小于后继 ID（拓扑序） |
| `kind` | string | `"memory-load"`、`"memory-store"`、`"scalar-load"`、`"fill"`、`"dot"`、`"elementwise"`、`"reduce"`、`"copy"` |
| `scope` | i64 | 任务所属的最内层 scope id（§3.10） |
| `stage` | i64 | 同步阶段：`max(前驱.stage + 前驱.barrier_after)` |
| `predecessors` | array<i64> | 直接前驱任务 ID（同 scope 或跨 scope 的合法边，见 §3.10/V19） |
| `input_values` | array<i64> | 实际读取或部分写继承的内容版本 ID；顺序与去重规则见 §3.11 |
| `output_values` | array<i64> | 写出的私有 root 的新内容版本（store/外部写必须为空） |
| `barrier_after` | bool | 任务完成后是否存在同步点 |
| `rematerializable` | bool | 纯计算、可安全重算 |
| `origin_id` | i64，可选 | 来源事件 id |

`stage/barrier_after` 是前端语义事实（不是调度决策），planner 只用它们判断
fuse 合法性与窗口复用边界。
`scalar-load` 默认 `barrier_after=false`、`rematerializable=false`；它是保留
读取时刻的内存事件，不生成独立 DMA，也不能跨可能覆盖源 root 的写者重算。

---

### 3.7 计算语义事实

#### `contract.compute`（dict，compute op 属性）

```mlir
linalg.matmul {contract.compute = {kind = "dot", signed = true,
                                   lhs = "i8", rhs = "i8", acc = "i32"}, ...}
```

只声明语义（“signed i8×i8→i32 的 dot”），是否走 AME 是 planner 结合 target
profile 的决策。`cast = #linalg.type_fn<cast_signed>` 保留为 linalg 语义属性。

兼容入口需要保留原有串行 reference matmul 行为时，显式声明执行顺序：

```mlir
linalg.matmul {contract.compute = {kind = "matmul", order = "mnk"}, ...}
```

该中性事实要求 M、N、K 顺序的串行循环，每个 K 步按 operand 0、operand 1、
旧 accumulator 的顺序读取，执行原始纯标量 arith payload 后立即写回。
它约束可观察的读取/写入顺序及原有浮点 flags，不表达 target engine。
`order` 只允许附在 `linalg.matmul`，且必须与 `kind="matmul"` 配套；v1
仅接受 `"mnk"`。未声明执行顺序的普通 matmul 继续执行 V25。

如果这个 reference op 的 A/B 与 output 无法证明不相交，它作为同步但
未建模的内存边界保留：scope 保持 incomplete，省略 `contract.task` 和
私有 output version，不伪造 snapshot/noalias 证明；失效所有可能受写入
影响的 root，未受影响的 root 保留。它不会使私有地址逃逸，后续完整、
不读旧值且来源已知的写仍可恢复局部版本。不得为含有这种边界的循环
声明 complete slot。已证明独立时可使用通常的 task/版本表达，仍不可
rematerialize。planner 必须选择机械 reference recipe，不能将这个语义
约束改成 AME 或 RVV 并行执行，也不能利用 incomplete 证明转发/窗口复用。

#### `contract.fill`（dict，`linalg.fill` 属性）

`padding=true` 表示这次 fill 是给 masked load 的 padding 区域做初始化；
planner 必须保证 fill 在对应 prefix DMA 之前（顺序由 scope 内任务顺序 +
`predecessors` 保证）。

---

### 3.8 provenance 事实

#### `contract.provenance`（dict，op 属性）

```mlir
linalg.matmul ... {contract.provenance = {
  origin_id = 3 : i64, source_load_origin_ids = array<i64: 0, 1, 2>,
  a_origin_id = 1 : i64, b_origin_id = 2 : i64,
  a_source_load_origin_ids = array<i64: 1>,
  b_source_load_origin_ids = array<i64: 2>,
  accumulator_source_load_origin_ids = array<i64: 0>,
  a_source_lineage_complete = true, b_source_lineage_complete = true,
  accumulator_source_lineage_complete = true}}
```

用途：报告、rematerialization 判断、跨 arm 同一逻辑值识别（两个 arm 的 load
共享 `origin_id`）。**不参与 alias 判断**。

`source_load_origin_ids` 以及各 operand 的 `*_source_load_origin_ids` 记录
源访问/初始化链识别的 source load 事件，包括仍存活及已被合并/消除的
事件；它们不是数值依赖的传递闭包。尤其 loop-carried accumulator 的
operand 来源沿 slot 的 init 追踪：本例是初始 C load `[0]`，不把前轮 A/B
的算术贡献展平到 accumulator 列表中；后者由 task 图和循环 slot 表达。
dot 的汇总列表仍保留 A/B/C 三组来源 `[0,1,2]`。lineage complete 表示这套
源访问追踪完整，不能作为数值独立或删除依赖的证明。
可选的 `source_store_origin_ids = array<i64: ...>` 同样保留相关 source store
事件。所有 origin id 必须非负；对应的
`*_lineage_complete=true` 声明存在时，其 source id 列表必须非空，例如
`source_store_lineage_complete=true` 对应 `source_store_origin_ids`。

常量/zero-fill operand 若经标准 IR 证明没有 source load，不声明要求非空
列表的 operand `*_source_lineage_complete=true`；函数级来源聚合仍可完整。
无法追踪的来源与这种已知空来源必须区分，不能给常量编造 load origin。

来源 ID 与任务 ID 是两个命名空间：planner 的 `task_map` 覆盖 rewrite 前的
全部任务 ID；provenance 保存 source 事件 ID。两者共同保留折叠关系，不能
要求 origin ID 出现在 `task_map.old_id` 中。planner 新建的 staging 是物理
事件，沿用可追踪的 operand 来源，或省略 origin 并记录
`contract.provenance={generated_by="planner-staging"}`；不能编造 source
origin。load/store 被融合后仍须保留其真实来源事件。

---

### 3.9 程序轴 / 循环 / 控制流事实

#### `contract.program_axes`（dict，op 属性）

```mlir
{contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>}}
```

表示该访问/任务的地址是否随 program id 轴变化。planner 用它做报告与
跨 program 复用分析（当前不建模）。

#### `contract.loop_carried_slots`（array<dict>，`scf.for` 属性）

```mlir
scf.for ... { } {contract.loop_carried_slots = [{
  slot = 0 : i64, init_value_id = 3 : i64, iter_arg_value_id = 7 : i64,
  yielded_value_id = 6 : i64, result_value_id = 8 : i64,
  init_producer_compute_ids = array<i64: 0>,
  iter_arg_consumer_compute_ids = array<i64: 3>,
  backedge_producer_compute_ids = array<i64: 3>,
  exit_consumer_compute_ids = array<i64: 4>}]}
```

把 `scf.for` 的 buffer 携带值映射回逻辑值图：init/iter_arg/yielded/result 四个
value id 以及相关任务 ID。`init_value_id` 必须由循环外的任务产生；
`iter_arg/result` id 由该循环的 `contract.value_ids` 定义。
**v1 限制**：本套 complete loop 摘要仅支持一个 buffer-carried slot，
`slot=0`；`contract.loop_carried_slots` 必须恰有一项，`contract.value_ids`
必须恰为对应的 `[iter,result]` 两个 ID。循环必须具有**正静态 trip count**、
正 step、串行执行且没有 early exit；trip count 必须与实际 bounds/step
一致，计算时检查整数溢出。零次循环的 result 应来自 init，但 v1 不实现
这条分支摘要，不能假装依赖没有执行的 backedge。多 slot、零次/动态次数
或其他未建模 loop 形态必须将相关 scope 标记 `complete=false`、填写
limitation 并诊断，不得只保留第一个 slot 后声称完整。

**同 root 与 latest-version 状态机**（verifier 必须从标准 IR 重建）：

1. 从 init 版本的真实定义推导唯一私有 storage root R；init 必须是进入
   loop 前 R 的最新版本。iter/result 继承此 root 绑定，不能另外指定 root；
   yielded 的真实 writer 也必须写同一个 R，不能拿 A/B 等其他 buffer 的
   body 版本冒充 C 的回边。
2. 在循环体入口，把 R 的当前版本绑定为 iter，而不是继续使用 init。
   正常任务按 §3.11 的真实读取、部分写继承及写入依次推进当前版本。
   因此首个 partial write 必须继承 iter；若前面已有 R 的 writer，则继承
   那个 writer 的最新版本，不能绕过它去引用 init、iter 或更早版本。
3. body 正常退出点，yielded 必须**恰为 R 的最后一个 writer 产生的最新
   版本**，且 writer/版本属于该 body；仅仅“由 body 中某任务产生”不够。
   退出 loop 后，把 R 的当前版本绑定为 result；后续实际读取或 partial
   write 必须引用 result，直到下一个 writer 推进它。
4. 动态语义为 `iter[0]=init`、`iter[k+1]=yielded[k]`、
   `result=yielded[trip_count-1]`。回边不作为违反拓扑序的
   `task.predecessors` 数字边加入；由 slot + serial loop 事实承载。

producer/consumer 列表必须由上述 root/版本状态机及实际 task 关系精确
派生、按 ID 升序去重；它们不是生成者可以任意省略的提示。init producer
与 yielded/backedge producer 必须与对应版本的真实定义一致。
`iter_arg_consumer_compute_ids` 包含循环体内对 iter 的实际读取/继承任务，
以及首次完整覆盖同一 root、因而需要从 init 建立入口 WAW 顺序的任务；
`exit_consumer_compute_ids` 同样包含父 scope 中对 result 的实际读取/继承
任务，以及退出后的首次完整覆盖同一 root 所需的 WAW 排序任务。
后一类任务仍不读取旧内容，不得为了匹配列表而在其 `input_values` 或
语义 `value_users` 中伪加 iter/result。边界列表只把真实存在的内存顺序
显式带过 scope；后续同 root writer 通过其 scope 内 predecessor 链排序。
列表中的任务、版本、root 绑定及 RAW/继承/WAW 原因均须在 verifier 中
交叉核验。

**不完整 loop 的 schema**：没有完整 slot 状态机的边界不得填造假的
`value_ids`/slot 来“补齐”图；应省略该未建模摘要，在 scope limitation 中
指出未建模的 root/边界及原因。已保留的任何任务、定义、引用和子摘要仍须
满足类型、唯一性、闭包及其声明的语义；不得用未知旧 ID 代替未知边界。
不完整摘要不能证明跨迭代融合、转发、重排或窗口复用。

#### `contract.cf`（dict，`scf.if`/`scf.for` 属性）

```mlir
scf.if ... { } else { } {contract.cf = {
  kind = "uniform_if", condition_axes = array<i64: 0>,
  arms_exclusive = true, arm_logical_tile = array<i64: 32, 128>}}

scf.for ... { } {contract.cf = {
  kind = "uniform_for", trip_count = 6 : i64, uniform = true, serial = true}}
```

| 字段 | 语义 |
| --- | --- |
| `kind` | `"uniform_if"`（条件只依赖 program id，分支互斥）或 `"uniform_for"`（固定 trip count、串行迭代） |
| `arms_exclusive` | 两 arm 对同一 program 互斥 → planner 可共享 SPM 窗口 |
| `arm_logical_tile` | 两个 arm 是同一逻辑 tile 的不同变体 |
| `condition_axes` | 条件依赖的 program 轴 |
| `trip_count/uniform/serial` | v1 complete loop 为正静态次数、各 program 相同、串行且无 early exit；次数与实际 bounds/step 一致 |

---

### 3.10 scope 事实（图的分区完备性）

计算图以 **scope** 为单位组织。scope 是一棵挂在函数体（root scope 0）下的
region 树：

```text
scope 0 (function)
├── scope 1 (if-arm then)      ← contract.scopes 的第 1 项（挂在 scf.if 上）
│   └── scope 3 (for-body)     ← contract.scope（挂在 scf.for 上）
└── scope 2 (if-arm else)
    └── scope 4 (for-body)
```

#### `contract.scopes`（array<dict>，`scf.if` 属性）

```mlir
scf.if ... { } else { } {
  contract.cf = {kind = "uniform_if", ...},
  contract.scopes = [{id = 1 : i64, kind = "if-arm", arm = "then",
                      parent = 0 : i64, arms_exclusive = true,
                      condition_axes = array<i64: 0>, complete = true},
                     {id = 2 : i64, kind = "if-arm", arm = "else",
                      parent = 0 : i64, arms_exclusive = true,
                      condition_axes = array<i64: 0>, complete = true}]}
```

| 字段 | 语义 |
| --- | --- |
| `id` | scope id（函数内唯一，非 0） |
| `kind` | `"if-arm"` |
| `arm` | `"then"` / `"else"` |
| `parent` | 父 scope id |
| `arms_exclusive` | 与同级的另一个 arm 互斥（必为 true） |
| `condition_axes` | 条件依赖的 program 轴 |
| `complete` | 该 arm 的数据流描述是否完备 |

#### `contract.scope`（dict，`scf.for` 属性）

```mlir
scf.for ... { } {
  contract.cf = {kind = "uniform_for", trip_count = 6 : i64,
                 uniform = true, serial = true},
  contract.scope = {id = 3 : i64, kind = "for-body", parent = 1 : i64,
                    complete = true},
  contract.value_ids = array<i64: 7, 8>,
  contract.loop_carried_slots = [...]}
```

| 字段 | 语义 |
| --- | --- |
| `id` | scope id |
| `kind` | `"for-body"` |
| `parent` | 父 scope id |
| `complete` | 循环体的数据流描述是否完备（循环本身用 `contract.cf` 的 uniform/trip_count 摘要） |

#### 每个 scope 的完备性定义（verifier 规则 V18）

- **叶子 scope**（无子 region）：`complete=true` ⇔ 该 scope 内所有任务的
  id/kind/scope/stage/predecessors/input_values/output_values 齐全，
  所有值引用可解析（V7），predecessors 属于同一 scope 或 V19 允许的
  loop init/exit 边，且没有 call/
  未知副作用。
- **if 所在 scope**：`complete=true` ⇔ 所有 arm scope complete + 两 arm
  之间没有私有内容版本的数据流边；共享仅限外部入口 memref 和普通标量
  SSA（不引入跨 scope 的标量内存任务依赖），私有 root 在各 arm 内独立
  定义 + `arms_exclusive` 成立。v1 不建模
  父私有 root 跨入 arm 的读取/写入或私有内容结果合并；相关 scope 必须
  incomplete，省略不可靠图断言，不能伪造 V19 之外的边。
- **for 所在 scope**：`complete=true` ⇔ 循环体 scope complete +
  `trip_count` 为正静态值、正 step 且 `uniform/serial`、无 early exit +
  v1 唯一 buffer-carried slot 按 §3.9 完整登记 + 循环内没有未建模 effect。
- **函数级聚合**：`compute_graph.complete = AND(所有 scope.complete)`。

**跨 scope 的合法边**（V19）：

1. 同 scope 内的 predecessor；
2. 父 scope init producer → 子循环体 slot 的 iter consumer 的 **loop init**
   边，包括实际读取/继承或同 root 首次完整 kill 所需的入口 WAW；
3. 子循环体 backedge producer → 父 scope slot 的 exit consumer 的
   **loop exit** 边，包括实际读取/继承或同 root 首次完整 kill 所需的出口
   WAW；
4. 兄弟 if-arm 之间**不允许**任何 predecessor/值边（互斥）。

未建模的 arm 触及父私有 root 后，该 root 在合流处的内容版本视为未知；
父 scope 后续读取不能沿用分支前的旧版本，须省略受影响断言并继续记录
limitation，直到可证明的完整初始化建立新版本。外部写 effect 则把所有
可能 arm 的写区域并集传播到父 scope：分支内 terminal store 后再读同一区域
仍是 V24 不支持的外部写后读，不能因 arm 互斥而遗漏。
完整初始化仅恢复该 root 后续的局部版本推理，不会把未建模边界所在 scope
重新变成 complete，也不能据此跨该边界复用或重排。

同样的失效规则适用于不完整 loop 和未知 effect：仅对已证明不受影响的
root 保留当前版本；可能被写入、escape 或影响范围无法证明的 root 变为
unknown，不能继续引用边界前的旧版本。恢复须由不读取未知旧内容的完整
覆盖建立新版本，例如完整 fill/copy 或纯写 generic；source/标量依赖仍须
已知且合法，恢复点没有可能冲突的未完成访问。matmul/reduce 读取旧 init，
partial fill/copy 继承未覆盖区域，
均不能单独恢复 unknown。无论后来是否恢复，未建模 scope 仍 incomplete。

**循环摘要**（V20）：对 complete 的 v1 单 slot loop，`iter_arg_value_id` 与 `result_value_id` 必须出现在该
循环的 `contract.value_ids` 中（顺序：iter、result）；`init_value_id` 必须由
父 scope 的任务产生；`yielded_value_id` 必须由循环体内任务产生；
`exit_consumer_compute_ids` 必须在父 scope；`backedge/iter_arg` 消费者必须在
循环体内。四版本同 root、init/body-exit/parent-exit 的 latest 关系与列表
精确性按 §3.9 重建验证，不能只检查 ID 存在及 producer 所在 scope。

本 kernel 的 scope 划分与完备性：

| scope | kind | parent | 任务 | complete |
| --- | --- | --- | --- | --- |
| 0 | function | — | （仅含 if） | true（聚合） |
| 1 | if-arm then | 0 | 0, 4 | true |
| 2 | if-arm else | 0 | 5, 6, 11 | true |
| 3 | for-body | 1 | 1, 2, 3 | true |
| 4 | for-body | 2 | 7, 8, 9, 10 | true |

> 旧实现的函数级 `complete=false`（“有控制流就不完整”）被本模型取代：
> 控制流由 scope 树 + uniform/loop 摘要显式描述，因此本 kernel 的图是
> **complete=true**。incomplete 来自 call、未知 effect、无法解析的访问或
> v1 未建模的控制流等明确边界，不能用来掩盖已声明 schema 的错误。

---

### 3.11 存储根、区间与读写派生（内容版本的基础）

**存储根（storage root）**：从 memref 值沿 view chain 向上，直到
`memref.alloc/alloca`、函数参数或 call 结果；中途的 `reinterpret_cast`/
`subview`/`cast`/`collapse_shape`/`expand_shape`/`transpose` 只改变
(view) 区间，不改变 root。公开入口之间由 `contract.abi.args_disjoint`
保证不 alias；这份 ABI 仅适用于声明过的函数参数，不能推及任意 call
结果等其他 root。同一 root 的不同 view 视为**可能 alias**（除非能静态
证明区间不相交）。

**区间**：每个 view 有 offset/sizes/strides；版本读写的字节区间 =
operand 的 view 区间 × 元素字节。动态 offset（program id 相关）不展开：
同一 root 下的区间若不能证明不相交，按“可能重叠”保守处理。
非连续 view 必须按实际访问集合判断，不能仅凭包围区间相同就认定逐元素
等价；rank-0 view 的访问集合是一个元素。
所有比较统一为相对于同一 root 的**字节地址集合**，不同元素类型的
element offset/stride 不能直接作比较。地址函数必须按完整 SSA view chain
的实际语义组合，包含 `memref.view` 的 byte-shift/base 调整；两个结果
MemRefType 的 offset/strides 相同不足以证明地址相同。不能重复叠加
`reinterpret_cast` 已替换或 rebased 的 descriptor offset。任意环节的
base/offset/stride 无法证明时，按可能重叠处理。

“完整覆盖 root”也必须证明实际写地址集合恰好覆盖 root 的全部内容，且
没有越界或重复地址掩盖空洞；相同元素数、相同包围区间或相同形状都不够。
不能证明完整覆盖的 view 按 partial write 处理，必须有合法旧版本继承；
若是首次初始化，则需要前置完整 fill/copy 或明确诊断。

**读写派生表**（verifier V22 按此表从 IR operand 派生，不需要额外属性）：

| op 形态 | read set（`input_values` 顺序） | write（新版本） |
| --- | --- | --- |
| `memref.copy %src, %dst` | `%src` 的版本 | `%dst` 私有 root 的新版本；`%dst` 为外部 root → store，无版本 |
| `linalg.fill ins(%s) outs(%o)` | 无（`%s` 是标量 SSA；其生产任务按下述标量规则建 predecessor） | `%o` 私有 root 的新版本 |
| `linalg.matmul ins(%a,%b) outs(%c)` | `%a`、`%b` 版本 + `%c` 旧版本 | `%c` 私有 root 的新版本；`%c` 为外部 root → direct accumulator，写由 store/route 表达 |
| `linalg.generic ins(...) outs(...)` | 仅列 payload 实际使用的 block argument 对应 memref 版本；outs block argument 未用时不读旧 output | 每个 outs 私有 root 一个新版本（本规范限制单输出） |
| `linalg.reduce ins(%x) outs(%init)` | `%x` 版本 + `%init` 旧版本 | `%init`（rank-0 root）新版本 |
| `memref.load %src` | **load 发生时** `%src` 的版本（读该元素） | 无；`kind="scalar-load"`，`output_values=[]`，SSA 标量结果不是内容版本 |
| `memref.store %v, %dst` | `%v` 是标量 SSA，不是内容版本；按标量规则建 predecessor | v1 仅外部 store，无版本；私有 rank-0 store 规范化为 fill，其他私有 scalar store 暂不支持 |
| `scf.for` | — | `contract.value_ids = [iter, result]`（由 loop slot 约束） |

以上是**语义读取集合**。另外，写私有 root 的 view 未覆盖整个 root 时，
新快照必须继承旧快照的未覆盖内容，因此把旧版本追加到 `input_values`
（若已是语义输入则不重复）。完整覆盖且不读旧 output 的 fill/copy/generic
不引用旧版本；仍必须以 WAW predecessor 保留同 root 前一次写的顺序。
`input_values` 按实际读取 operand 顺序收集，再追加部分写继承版本，首次出现
保留、去重。没有对应版本的真实读取或部分写继承均报错；不得读取未初始化内存。
继承版本必须是**该 write 发生之前同 root 的当前版本**；不能在循环体内
总是继承 loop init，或在中间已有 writer 时仍继承 iter/其他旧快照。版本
与 root 的绑定及 loop 入口/退出切换按 §3.9 执行。

**masked load 的版本链**：先 `linalg.fill(other, logical_buffer)` 写版本 v1，
再 `memref.copy(valid_subview, buffer_subview)` 读取外部入口版本并继承 v1，
写 root 的新版本 v2；consumer 读 v2。tail 的 padding 由 v1 继承，
无需新属性。完整块 copy 覆盖整个 root 时不需要这条继承输入。

**标量/索引链与读取时刻**：arith/index/math 等纯标量 SSA 运算不建任务、
不定义内容版本；`memref.load` 是独立的轻量 `scalar-load` 任务，读取版本
在该 load 位置解析。消费标量的 buffer task 沿 SSA use-def 递归到
`scalar-load`，把这些 load task 加入自己的 `predecessors`；常量及普通
标量参数终止追溯。reduce → scalar-load → div/rsqrt → fill 因而拥有完整
依赖链，而不是把 fill 误认为在读取 reduction buffer。load 之后原 buffer
可以在无其他内存读者时覆盖；SSA 标量仍保留 load 时的值。RAW/latest 检查
在 load 点执行，WAR 边也保护 load 本身。planner/emit 必须保留这些事件边及
标量 SSA 支配关系，不得只连 reduce→fill 而把 load 移到覆盖之后。
v1 只支持同 scope 内的这类标量链；跨 scope 的标量内存依赖未建摘要时，
对应 scope 必须 incomplete，不能作为重排证明。

**单任务原地规则**：旧版本在本次写覆盖区间上不得还有未来的**内存读者**
（`scalar-load` 算读者，已完成 load 的后续纯标量计算不算）。完整 kill
不要求读旧值；真正读取旧 destination 的 matmul/reduce/elementwise 必须
引用旧版本。除此之外还须满足以下 op 级条件，不能仅看 identity maps：

| op | 同一任务内的别名条件 |
| --- | --- |
| pointwise generic | 对每个与 output 重叠的实际读取 operand，逐元素地址函数必须与 output 相同；其他输入须不相交。要求单 arith、纯并行、identity indexing；仅有相同 indexing map 不够 |
| matmul | output 只与自身 accumulator 读取重叠；A/B 与 output 不相交；保持 linalg 累加语义 |
| reduce | output 只与自身 init 读取重叠；reduction input 与 output 不相交；rank-0 init/result 合法 |
| fill | 无 tensor/memref 输入，scalar SSA 已计算；可完整覆盖或按部分写继承 |
| copy | source/destination 不相交，或逐元素地址完全相同（可消除的 self-copy）；部分重叠及未知重叠拒绝 |

例如 `ins(root[0:768]) outs(root[1:769])` 即使两张 indexing map 都是 identity
也不合法。无法证明上述条件时，前端另开 root 并按需要复制旧 accumulator；
verifier 拒绝不安全的原地契约，不能由 emitter 猜测遍历顺序修复。
§3.7 显式 reference order 的 may-alias matmul 不声明 snapshot task，按
未建模边界保留；它不放宽 AME、普通 matmul 或 reduce 的原地规则。

**外部写序**：写外部 root（store/direct store）必须晚于该 region 的所有
先前读者（V24）；v1 外部入口版本只用于写前读取，不支持写后再次读取该
外部 region。direct store 的额外合法性条件（覆盖性/别名/最后观察点）见
03 §4.7。
该检查比较所有可能相交的外部 view，不能仅按 storage root 的 SSA 相等性
分组；不同函数参数只有在 ABI 或地址证明成立时才视为不相交。

---

## 4. mask 分析的输出契约（重要）

前端的 mask 分析必须把任意 predicate 归约成下面四类之一，planner 只处理
这四类：

| 类别 | 属性写法 | planner 行为 |
| --- | --- | --- |
| 无 mask | `mask = {kind = "none"}` | 整 tile 可任意 DMA/直连 |
| 静态矩形/前缀 | `mask = {kind = "rectangular", valid_shape = [...]}` | 逐行 prefix DMA；剩余区域由前置 fill 提供 |
| 动态矩形/前缀（**v2**） | `mask = {kind = "rectangular", dynamic = true}` + IR 结构提供 extent SSA | v1：verifier 直接报错（见 §4.3）；v2：clamp+64B 整除可证 → guarded prefix DMA，否则 opaque |
| 不可描述 | `mask = {kind = "opaque"}` | planner 必须选择 vector copy/逐元素；不允许 DMA 直搬 |

### 4.0 本 kernel 没有动态 mask

本 kernel 的边界维度（M=196、N=768、K=768）都是编译期常量，前端用
`scf.if (pid0 < 6)` 把 program 空间特化成“满块 / tail”两个 arm，tail 的
有效行数因此是**静态的 4**。所以示例只有 `mask = none` 与
`mask = {kind="rectangular", valid_shape=[4,128]}`，**没有** `dynamic = true`。
动态 mask 是契约给其它 kernel / 其它前端保留的能力，不是本 kernel 的特性。

### 4.1 “动态 mask 的 extent”是什么（v2 预留）

mask 的有效区域不是编译期常量时，用一个 **SSA 值**表示“有效长度”，称为
extent。典型来源：运行时边界（`valid_rows = clamp(N - row_base, 0, tile_rows)`）、
运行时偏移取整（`(limit & -16)` 再 clamp）。

契约属性不能保存 SSA，所以约定：**extent 必须作为边界 IR 的 operand 出现**
（动态 size 的 subview、scf.for 上界、guard 值），planner 从这里读取。

```mlir
%ext = ...                       // SSA：clamp 后的有效元素数
%src = memref.subview %source[0] [%ext] [1]
memref.copy %src, %dst {contract.access = {kind = "load", boundary = "external_ddr",
    mask = {kind = "rectangular", dynamic = true}}}
```

planner 的证明义务（全满足才允许 guarded prefix DMA）：0 ≤ extent ≤
static_max；extent 是 DMA 包元素的整数倍；定义支配消费点；target 支持
guarded prefix。任一不满足 → opaque 或报错。注意动态 extent 只描述
“有效长度”，不描述交错/数据相关的 lane 选择。

### 4.2 什么时候会出现动态 mask（即使 tile/张量形状静态）

关键是区分 **tile 形状**（静态）与 **valid 区域长度**（可以是运行时 SSA）。
常见来源：

1. **问题维度是运行时值**（LLM 变长 seq_len、KV 长度、可变 batch 边界）：
   前端无法把 tail 判成固定行数，只能保留运行时 extent；`if/else` 仍可分
   满块/尾块，但尾块内 valid 行数是 SSA。
2. **前端选择不特化**：即使可证明边界，也可能为控制分支/代码尺寸保留单一
   masked path → extent 是运行时可算值。
3. **其它前端 / 手写 buffer IR**：buddy-mlir 是多前端后端；现有 planner 的
   `analyzeDynamicPrefixCopy` 就是为这类输入准备的（rank-1、offset 0、
   unit-stride、SSA extent、静态最大值 buffer，并证明 clamp+整除）。
4. **无法静态分 arm 的 predicate**：不是“坐标 < 上界”的简单形式时，只能
   dynamic（可证 clamp）或 opaque。

### 4.3 v1 接受范围与前端义务（已定）

v1 只接受 `none` / `rectangular-static` / `opaque`；出现
`dynamic = true` 时 **verifier 直接报错**（V16），planner 不实现
guarded prefix（留 v2）。

对输入的要求**不只是“矩阵 shape 静态”**：

| 条件 | 说明 |
| --- | --- |
| C1：问题维度编译期常量 | M/N/K、seq_len 等必须来自编译期常量或带整除承诺的 ABI；**shape 静态是必要条件** |
| C2：mask 可静态分区 | 仍依赖 program id 的 mask 必须由前端特化成静态 arm（如 `pid0<6` 分成满块/tail 4 行）；不特化的话即使 M 静态也会得到运行时 valid 长度而被拒绝 |
| C3：其余 mask → opaque | 数据相关/交错 mask 只能标 opaque；opaque 不保证编译成功 |

**反例（C2 不可省）**：M=196 是常量，但 `valid_rows = clamp(196 - pid0*32,0,32)`
依赖运行时 pid0；只有特化成 arm 后才变成静态 `valid_shape=[4,128]`。

**反例（C1 不是字面 shape 静态）**：若运行时维度被 ABI 承诺为 tile 整数倍，
mask 可删成 none，即使形状是运行时值也不需要 dynamic。

**后果**：现有 `dynamic-prefix-*.mlir` 在 v1 下会被 verifier 拒绝；能力转 v2
（`analyzeDynamicPrefixCopy` 的证明逻辑保留为参考）。前端遇到无法静态化的
mask 必须报错并指出具体访问。

---

## 5. 什么绝对不能放进契约

| 禁止项 | 原因 |
| --- | --- |
| `bank`/`offset`/`capacity`/`memory_space` | target 决策 |
| packet 列表/`channel`/`token`/`wait` 策略 | DMA 物理细节 |
| direct/staged/route 决策 | planner 职责 |
| `bm/bn/bk`、`lower_to_ame`、`enable_*`、`placement` | 目标配置/计划选项 |
| `bosc_ame.*`、`plan.*` 命名 | 与后端演进解耦 |
| target 常量 | 换 target 即失效 |
| SSA 值的重复描述（shape/stride/offset） | 已由 memref 类型与 view chain 表达 |
| 调度顺序/同步指令 | 计划与 emit 职责 |

---

## 6. Verifier 规则

两端各执行一次；违反即编译错误（诊断指向具体 op）。`complete=false`
表示存在明确未建模的 IR/effect/控制流边界，不表示现有属性可以违反 schema。
**所有已声明**的任务、定义、引用、root 绑定和子摘要始终必须满足字段类型、
ID 唯一性、闭包及其已建模语义；不得用不存在的旧 value id、重复定义或
虚构依赖绕过校验。未建模部分应省略不可靠的图断言，并在对应 scope 的
limitation 中定位边界及原因。不完整摘要不得用作融合、转发、复用、重排
或跨迭代优化证明；独立的操作级路线合法性仍可按其完备的访问事实验证。

| # | 规则 |
| --- | --- |
| V1 | `contract.version` 存在且受支持；未知字段忽略 |
| V2 | `contract.abi` 若存在：字段类型合法；`args_aligned_bytes` 为正整数，用于64 B访问须为64的正整数倍（96不满足）；无该属性时 planner 记录“无 ABI 承诺” |
| V3 | 内存接口 memref-only：无 tensor、无 unrealized_conversion_cast、无 SPM memory space、无 pointer 类型；program id 等普通标量参数允许 |
| V4 | 每个 `contract.access`：kind/boundary 合法；mask.kind ∈ {none,rectangular,opaque}；rectangular 时 valid_shape rank 与访问 view rank 一致且各维 >0；other 若存在必须 kind=load 且元素类型匹配（masked load 必须记录） |
| V5 | 每个 `contract.buffer`：载体是 alloc/alloca；role/kind 枚举合法；logical_shape rank ∈ [0,2] 且与底层 buffer/view 字节数一致；rank-0 空形状承载一个元素 |
| V6 | `contract.task.id` 全函数唯一、非负、**严格递增**；`compute_graph.scope_count` 与实际 scope 数一致 |
| V7 | value 定义点与使用闭包：定义点仅函数参数 `contract.value_id` / 任务 `output_values` / 循环 `value_ids=[iter,result]`；每个 id **恰好定义一次**；每个 `input_values`/`init_value_id` 引用可解析；`yielded/result` 与循环内产生值/循环 value_ids 一致 |
| V8 | `contract.value_users` 若存在：与从 `task.input_values` + loop语义读取/继承边重建的消费者集合完全一致；slot consumer中的纯WAW任务不构成版本读取 |
| V9 | `task.predecessors` 中每个 id 可解析且 < 当前任务 id；跨 scope 边必须匹配 §3.10/V19 的三种合法形态 |
| V10 | `contract.cf` 只出现在 `scf.if`/`scf.for`；`arms_exclusive` 仅 `uniform_if` 合法；`condition_axes` 引用的轴存在 |
| V11 | `contract.loop_carried_slots`：四个 value id 与任务 id 全部可解析；init 生产者在父 scope；iter/backedge 消费者在循环体内；exit consumer 在父 scope |
| V12 | `contract.provenance`：全部origin/source id非负，包括可选`source_store_origin_ids`；其 `*_lineage_complete=true` 时对应source id列表非空，标准IR证明的无load来源按§3.8表达；函数 `compute_graph.source_lineage_complete` 必须等于从实际内存边界和保留来源事实重新计算的 aggregate；task_map覆盖旧任务ID，provenance保留source事件ID，不混用命名空间；不因物理事件消除丢失来源 |
| V13 | （v2）动态 mask：extent SSA 可在 IR 结构中定位；否则失败 |
| V14 | 禁止项递归检查：module/function/body/payload/所有嵌套 region、参数属性及嵌套属性/type 中均不得夹带 `bosc_ame.*`/`plan.*`、SPM memory space 或 target 决策；不能只检查 task op；incomplete 不豁免 |
| V15 | 不完整降级：明确未建模边界令相关 scope `complete=false` 并填定位原因；已声明部分仍强制 schema/唯一性/闭包，未知旧 ID 不合法。complete=true 时遗漏实际任务/effect 必须报错；不可靠摘要不能作为优化证明 |
| V16 | **v1 拒绝动态 mask**：任何 `mask.dynamic = true` → 报错“v1 只支持 none/static/opaque” |
| V17 | scope 结构：scope id 唯一非 0；parent 引用存在且构成树（root=0）；`scope_count` = 1 + 记录数；`if-arm` 恰好 then/else 两个且 `arms_exclusive=true` |
| V18 | 每 scope `complete` 按其 kind 的定义（§3.10）成立；函数级 `complete` = AND(所有 scope) |
| V19 | 跨 scope 边合法：同 scope / 父→子 loop init / 子→父 loop exit；兄弟 arm 之间**不得**有 predecessor 或值边 |
| V20 | v1 complete 循环摘要仅一个 `slot=0`、正静态 trip/正 step/serial/无 early exit；`value_ids=[iter,result]` 与唯一 slot 一致；init/iter/yielded/result 同一 private root，init 为入口最新版本、yielded 为 body 退出最新版本、result 为父 scope 退出后的当前版本；所有 producer/consumer 列表精确派生。多 slot/零次/其他未建模 loop 必须 incomplete 并诊断，不伪造 slot |
| V21 | **版本定义完整性**：定义点仅 §3.4 的三类；每个 id 在静态 IR 恰好定义一次；每个任务的 `output_values` 与它写的私有 root 一一对应（顺序 = operand 顺序）；store/外部写的 `output_values` 必须为空；完整单 slot loop 的 `value_ids` 恰好 `[iter,result]`，未建模 loop 不伪造摘要 |
| V22 | **读写与继承派生**：按 §3.11 实际使用的 operand/block argument 派生语义读取，再追加部分写继承旧版本；`input_values` 与所得有序去重集合一致。完整 kill 仅建 WAW，不伪读旧版本；首次私有写须完整初始化；函数参数入口版本只读 |
| V23 | **RAW / latest-version**：每个实际内存读取或部分写继承引用其事件点该 root 的最新版本；producer 必须为该事件的前驱或合法 loop/入口来源；不得以本任务尚未写出的版本满足自己的读取。`scalar-load` 在 load 点校验，下游纯标量 SSA 使用不再次读取旧 root |
| V24 | **WAR/WAW 与外部写序**：同一私有 root 的写者由 predecessor 关系全序（loop slot 允许迭代间顺序）；覆盖区间上的旧版本内存读者须先于写者；无法证明不相交即按重叠处理。外部同 region 的入口读取先于终端写，写后读不属 v1；scalar-load→writer 的 WAR 和 scalar-load→标量消费者的 SSA 依赖均须保留 |
| V25 | **原地安全**：按 §3.11 op 级条件验证 pointwise/matmul/reduce/fill/copy；覆盖区间旧版本无未来内存读者；读取旧 destination 时必须有旧版本输入，纯 kill 不要求。identity indexing 还须逐元素地址等价或不相交；部分重叠、未知别名及不支持 op 不得声明安全 snapshot task，否则报错要求前端另开 buffer；§3.7 显式 reference-ordered may-alias matmul 仅以 incomplete 无 task 的边界保留 |
| V26 | **scope/loop 版本约束**：loop slot 的 init 版本来自父 scope、iter/backedge 在循环体、result 在父 scope；互斥 if-arm 之间不得共享私有 root 版本；跨 arm 只允许外部输入/输出事实。未建模的父私有root跨if访问/合流后版本为unknown，不得沿用旧版；arm外部写effect须传播到父scope |

---

## 7. 与现有 `bosc_ame.*` 属性的迁移映射

| 现有属性（bufferized.mlir） | 契约属性 |
| --- | --- |
| `bosc_ame.program_axis/program_role`（arg） | `contract.program {role, axis}` |
| `bosc_ame.compute_graph_version` | `contract.compute_graph.version` |
| `bosc_ame.compute_graph_complete/limitation/node_count` | 改为 per-scope：`contract.compute_graph.complete/limitation/scope_count` + `contract.scope(s)` |
| `bosc_ame.source_lineage_complete` | `contract.compute_graph.source_lineage_complete` |
| `gpu.known_grid_size` | `contract.grid` |
| `bosc_ame.compute_id/kind/stage/predecessors/input_values/output_values/barrier_after/rematerializable/compute_origin_id` | `contract.task {id, kind, scope, stage, predecessors, input_values, output_values, barrier_after, rematerializable, origin_id}` |
| `bosc_ame.invariant_program_axes/varying_program_axes` | `contract.program_axes` |
| `bosc_ame.logical_value_ids/logical_value_users` | **不直接搬**：旧 id 是 bufferization 遗留的 SSA/内容混合编号。新前端按 §3.4 重新分配“内容版本”，定义在 `contract.task.output_values` / `loop.value_ids`，用户边由 `contract.value_users` 重建 |
| `bosc_ame.memory` | `contract.access` |
| `bosc_ame.padding` | `contract.fill {padding=true}` |
| `bosc_ame.lower_to_ame` | `contract.compute {kind="dot", signed=...}` |
| `bosc_ame.origin_id/source_load_origin_ids/a_origin_id/...` | `contract.provenance` |
| `bosc_ame.logical_shape`（alloc） | `contract.buffer.logical_shape` |
| `bosc_ame.loop_carried_slots` | `contract.loop_carried_slots` + 循环 `contract.value_ids` |
| `bosc_ame.bm/bn/bk/target/enable_*/placement/expected_ame_matmuls/dma.abi` | **删除**（target profile/driver；ABI → `contract.abi`） |
| `cast`（linalg 语义属性） | 保留 |

**注意**：旧实现的 `compute_graph_complete=false`（有控制流即不完整）不再直接
迁移；新前端必须把控制流描述成 scope 树（arm/loop），并给每个 scope 判定
完备性。旧逻辑值 ID 空间（含 bufferization 遗留的 content-version id）不得
直接搬进契约；新前端自己 bufferize，必须给每个逻辑值一个唯一生产点。

---

## 8. 版本与兼容

- 契约是**跨项目 ABI**：任何字段语义变化必须提升 `contract.version`。
- 同一大版本内：新增可选字段允许；删除/改义字段禁止。
- planner 必须忽略未知的 `contract.*` 字段（前向兼容），并在 plan report 里
  打印“发现未知契约字段”。
- 前端 verifier 与 planner verifier 共享同一份规则实现（长期做成共享库）。
- 若某前端无法提供任务图，允许只提供 `contract.access/buffer/compute` +
  `contract.program` + `contract.grid`，并把所有 scope 标
  `complete=false`、`scope_count` 按实际树填写；planner 必须能仅靠边界事实
  做操作级规划。

---

## 9. Schema 形式：已定决策

**已定：契约长期使用 discardable attributes + 独立 verifier，不引入独立
dialect。** 理由：跨仓库依赖最小；属性已可解析可联调；正确性由共享 verifier
（V1–V26）+ conformance 测试保证。当前 `plan/tools` 的 Python 检查只是
文档/schema fixture 验证，不是该跨项目编译器 verifier 的实现。

**已定：逻辑值引用使用整数 ID，不引入 SSA 引用。** 由 V7/V8 强制闭合与
双向一致；本示例 17 个值直接可读。

**已定：图按 scope 组织（per-region graph + uniform-loop 摘要）**，见 §3.10。

## 10. 契约负例验收

除三个正例 fixture 外，独立 verifier 的 conformance 用例至少覆盖：

- loop 的 yielded 指向 A/B 等其他 root、init 不是父 scope 最新版本、
  yielded 不是 body 最后版本、首个 partial write 继承 init 或绕过前一
  writer，以及退出消费者绕过 result；均须拒绝。
- 零 trip、非正 step、多 slot、动态 trip 或 early exit 的 loop 不得声称
  v1 complete；合法 incomplete 不伪造 slot/value id，同时仍拒绝未知
  引用、重复定义和错误字段类型。
- 仅以 root/type/shape 相同证明重叠 view 原地等价，尤其遗漏
  `memref.view` 的 byte shift；必须拒绝未证明的部分重叠。
- 在 module/function、嵌套 region、参数属性、嵌套 attribute/type 隐藏
  `bosc_ame.*`、`plan.*` 或 SPM memory space；不论 complete 与否均拒绝。
- 完整 kill 出现在 loop 的首个/退出后首个 root writer 时，允许以 slot
  consumer 表达 WAW，但不得伪加旧版本 `input_values/value_users`；漏掉
  必需跨 scope WAW 边或把其他 root 的任务填进列表均须拒绝。
- ABI 对齐为 96 B 时不得据此批准 64 B direct 访问；负 origin、错误或空的
  complete lineage source 列表、task_map 遗失被折叠的 load/store 任务，
  以及 provenance 丢失其 source origin 均须拒绝。
- 父私有 root 在 arm 内被读写或私有内容结果合并时，不得伪称 v1 complete；
  合流后沿用旧内容版本必须拒绝。arm 内外部 store 后的父 scope 同区域
  read 仍须拒绝。未触及的其他 root 不因该降级丢失已有证明。
- 不完整 loop/未知 effect 可能写入的 root 同样失效；完整纯写可恢复局部
  版本推理，partial write、读取旧 init 或未知 source 不可恢复；恢复后
  scope 仍 incomplete，不得据此跨未知边界转发或复用。
- 无 origin 的完整标准 Linalg 图允许图 complete、来源 incomplete；常量
  operand 的已知空来源和未知来源分别测试。物理 load 转发后完整 source
  事实仍在时，不得无故降级；origin ID 与 old task ID 不同也须合法。

## Function-scoped mathematical accuracy

`contract.math` attaches directly to `func.func`. It is an explicit caller
promise, not a property inferred from a target or an operation. Its absence
never authorizes approximation, FP fusion, reassociation or a larger error
budget. Source evaluation order remains binding. For example:

```mlir
func.func @kernel() attributes {
  contract.math = {
    exp = {domain = "all-f32", special_values = "ieee",
           max_abs_error = 1.0e-6 : f64, max_rel_error = 1.0e-5 : f64}
  }
} { return }
```

The outer dictionary is nonempty and accepts only `erf` and `exp`. Each value
is a closed dictionary containing exactly the four fields shown above.
`domain` must equal `all-f32`; `special_values` must equal `ieee`. Both error
budgets must be finite, nonnegative f64 attributes. Nested, operation-scoped,
unknown-field and wrong-type accuracy declarations fail verification (V26).

For a finite mathematical reference result `r`, an output `y` must satisfy
`abs(y - r) <= max_abs_error + max_rel_error * abs(r)`. `all-f32` includes
nonfinite inputs. `ieee` requires mathematical special values, signed zero
where defined, gradual underflow and round-to-nearest, ties-to-even. NaN
payloads and floating-point exception flags are outside this value contract.
A target implementation must fit both the caller's explicit authorization and
its established error evidence. BOSCAME recipes, coefficients, evaluation
order and evidence belong to Buddy and are not part of this neutral schema.

## Implementation ownership and version boundary

This document is the normative schema. Buddy implements it in
`midend/include/Analysis` and `midend/lib/Analysis`; triton-riscv owns its strict
producer implementation in `include/triton-shared/Analysis` and `lib/Analysis`.
The repositories exchange standard MLIR text and a handoff manifest. They do
not share a library, source package or SDK. Conformance fixtures run in each
repository's ordinary test suite. The frontend build identity records the
schema version and hashes of its actual local implementation files.

The producer checks any existing `contract.version` and every existing
`contract.compute_graph.version` before normalization or graph reconstruction.
An absent module version permits first annotation. An existing version must
be an integer with value 1; an existing graph dictionary must contain such a
version. Unknown versions are never rewritten to v1. Integer attribute widths
accepted by v1 remain accepted.

Strict producers accept only standard buffer IR, logical memory space zero
and neutral facts. `plan.*`, target dialects/types and unrealized conversion
casts are rejected. A consumer may use the separate planned-IR adapter to
rebuild graph metadata after planning; it admits existing `plan.*`, retains
logical memory space zero and skips producer normalization.

Source policies recognize synthetic transfers through a callback shared by
annotation and verification. A synthetic transfer carrying an `origin_id`
does not turn incomplete retained source lineage into complete lineage.
Buddy alone interprets its `planner-staging` provenance spelling. Verified
incomplete graphs retain limitations; verification is not planner acceptance.
