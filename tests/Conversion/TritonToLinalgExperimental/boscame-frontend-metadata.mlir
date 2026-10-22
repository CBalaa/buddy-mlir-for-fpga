// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias bm=32 bn=64 bk=256' -o %t.linalg
// RUN: FileCheck %s --check-prefix=TENSOR --implicit-check-not=bosc_ame.spm.view --implicit-check-not=bosc_ame.dma.start < %t.linalg
// RUN: buddy-opt %t.linalg --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse -o %t.buffer
// RUN: FileCheck %s --check-prefix=BUFFER --implicit-check-not=bufferization. --implicit-check-not=tensor< --implicit-check-not=bosc_ame.spm.view < %t.buffer
// RUN: buddy-opt %S/Inputs/matmul-i8.ttir --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.matmul
// RUN: FileCheck %s --check-prefix=MARK --implicit-check-not=bosc_ame.tile.mma --implicit-check-not=bosc_ame.dma.start < %t.matmul
// RUN: buddy-opt %t.matmul --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse | FileCheck %s --check-prefix=MARK-BUFFER
// RUN: buddy-opt %S/Inputs/layernorm.ttir --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.layernorm-facts.mlir 2> %t.layernorm-facts
// RUN: FileCheck %s --check-prefix=FACTS < %t.layernorm-facts
// RUN: FileCheck %s --check-prefix=MEMORY < %t.layernorm-facts
// RUN: FileCheck %s --check-prefix=SOURCE-ADDR < %t.layernorm-facts
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias bm=32 bn=64 bk=256' -o %t.address-facts.mlir 2> %t.address-facts
// RUN: FileCheck %s --check-prefix=ADDRESS-EXPR < %t.address-facts
// RUN: diff %t.linalg %t.address-facts.mlir
// RUN: buddy-opt %S/Inputs/layernorm.ttir --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.layernorm-plain.mlir 2> %t.layernorm-plain.err
// RUN: test ! -s %t.layernorm-plain.err
// RUN: diff %t.layernorm-facts.mlir %t.layernorm-plain.mlir
// RUN: buddy-opt %t.layernorm-plain.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse -o %t.layernorm-buffer.mlir
// RUN: FileCheck %s --check-prefix=GRAPH-BUFFER < %t.layernorm-buffer.mlir
// RUN: buddy-opt %S/Inputs/layernorm.ttir --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse --linalg-to-boscame='enable-spm-dma=true enable-ame=true store-policy=spm-dma dma-abi=args_aligned64_nonalias plan-report=true' -o %t.layernorm-boscame.mlir 2> %t.layernorm-plan
// RUN: FileCheck %s --check-prefix=DATAFLOW < %t.layernorm-plan
// RUN: buddy-opt %S/Inputs/matmul-i8.ttir --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.matmul-facts.mlir 2> %t.matmul-facts
// RUN: FileCheck %s --check-prefix=MATMUL-FACTS < %t.matmul-facts
// RUN: diff %t.matmul %t.matmul-facts.mlir
// RUN: buddy-opt %S/Inputs/matmul-lineage.ttir --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.matmul-lineage-facts.mlir 2> %t.matmul-lineage-facts
// RUN: FileCheck %s --check-prefix=MATMUL-LINEAGE < %t.matmul-lineage-facts
// RUN: buddy-opt %S/Inputs/matmul-lineage.ttir --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.matmul-lineage-plain.mlir 2> %t.matmul-lineage-plain.err
// RUN: test ! -s %t.matmul-lineage-plain.err
// RUN: diff %t.matmul-lineage-facts.mlir %t.matmul-lineage-plain.mlir
// RUN: buddy-opt %t.matmul-lineage-plain.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse -o %t.matmul-lineage-buffer.mlir
// RUN: FileCheck %s --check-prefix=MATMUL-LINEAGE-BUFFER < %t.matmul-lineage-buffer.mlir
// RUN: buddy-opt %S/Inputs/conditional-address.ttir --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.conditional-address.mlir 2> %t.conditional-address
// RUN: FileCheck %s --check-prefix=ADDRESS-UNKNOWN < %t.conditional-address
// RUN: buddy-opt %S/Inputs/conditional-address.ttir --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' -o %t.conditional-address-plain.mlir 2> %t.conditional-address-plain.err
// RUN: test ! -s %t.conditional-address-plain.err
// RUN: diff %t.conditional-address.mlir %t.conditional-address-plain.mlir
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias bm=0' 2>&1 | FileCheck %s --check-prefix=BAD-TILE
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias target=unknown' 2>&1 | FileCheck %s --check-prefix=BAD-TARGET
// RUN: not buddy-opt %s --triton-to-linalg-experimental='dump-facts=true' 2>&1 | FileCheck %s --check-prefix=BAD-FACTS

// BOSCAME selection records configuration and the original logical memory
// events. Standard bufferization is a separate stage; only the later backend
// selects staging storage. Both copies keep the exact authorized prefix.
module {
  tt.func public @masked_copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %limit = arith.constant dense<16> : tensor<32xi32>
    %mask = arith.cmpi slt, %r, %limit : tensor<32xi32>
    %other = arith.constant dense<2.0> : tensor<32xf32>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %sp = tt.addptr %s, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %dp = tt.addptr %d, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %v = tt.load %sp, %mask, %other : tensor<32x!tt.ptr<f32>>
    tt.store %dp, %v, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// TENSOR: bosc_ame.bk = 256 : i64
// TENSOR-SAME: bosc_ame.bm = 32 : i64
// TENSOR-SAME: bosc_ame.bn = 64 : i64
// TENSOR-SAME: bosc_ame.enable_ame = true
// TENSOR-SAME: bosc_ame.enable_spm_dma = true
// TENSOR-SAME: bosc_ame.expected_ame_matmuls = 0 : i64
// TENSOR-SAME: bosc_ame.store_policy = "direct-ddr"
// TENSOR-LABEL: func.func @masked_copy
// TENSOR-SAME: bosc_ame.program_axis = 0 : i32
// TENSOR-SAME: bosc_ame.program_role = "num_programs"
// TENSOR-SAME: bosc_ame.program_axis = 1 : i32
// TENSOR-SAME: bosc_ame.program_role = "num_programs"
// TENSOR-SAME: bosc_ame.program_axis = 2 : i32
// TENSOR-SAME: bosc_ame.program_role = "num_programs"
// TENSOR-SAME: bosc_ame.program_axis = 0 : i32
// TENSOR-SAME: bosc_ame.program_role = "program_id"
// TENSOR-SAME: bosc_ame.program_axis = 1 : i32
// TENSOR-SAME: bosc_ame.program_role = "program_id"
// TENSOR-SAME: bosc_ame.program_axis = 2 : i32
// TENSOR-SAME: bosc_ame.program_role = "program_id"
// TENSOR: memref.alloc() {{.*}}alignment = 64
// TENSOR: linalg.fill
// TENSOR-SAME: bosc_ame.padding
// TENSOR: memref.copy
// TENSOR-SAME: bosc_ame.memory = {{.*}}kind = "load"{{.*}}origin_id = 0 : i64{{.*}}valid_shape = array<i64: 16>
// TENSOR: memref.copy
// TENSOR-SAME: bosc_ame.memory = {{.*}}destination = "external_ddr"{{.*}}kind = "store"{{.*}}origin_id = 1 : i64{{.*}}valid_shape = array<i64: 16>
// BUFFER-LABEL: func.func @masked_copy
// BUFFER: linalg.fill
// BUFFER: memref.copy
// BUFFER-SAME: bosc_ame.memory = {{.*}}kind = "load"{{.*}}origin_id = 0 : i64
// BUFFER-SAME: : memref<16xf32
// BUFFER: memref.copy
// BUFFER-SAME: bosc_ame.memory = {{.*}}destination = "external_ddr"{{.*}}kind = "store"{{.*}}origin_id = 1 : i64
// BUFFER-SAME: : memref<16xf32

// MARK: linalg.matmul
// MARK-SAME: bosc_ame.lower_to_ame
// MARK-SAME: bosc_ame.origin_id
// MARK-BUFFER: bosc_ame.compute_graph_complete = false
// MARK-BUFFER-SAME: bosc_ame.compute_graph_node_count = 11 : i64
// MARK-BUFFER: linalg.matmul
// MARK-BUFFER-SAME: bosc_ame.a_origin_id = 0 : i64
// MARK-BUFFER-SAME: bosc_ame.b_origin_id = 1 : i64
// MARK-BUFFER-SAME: bosc_ame.lower_to_ame
// MARK-BUFFER-SAME: bosc_ame.origin_id
// MARK-BUFFER-SAME: ins({{.*}} : memref<
// MARK-BUFFER-NOT: bosc_ame.accumulator_origin_id
// MARK-BUFFER: memref.copy {{.*}}bosc_ame.compute_origin_id = 3 : i64{{.*}}bosc_ame.memory = {{.*}}origin_id = 3 : i64
// MARK-BUFFER: memref.copy {{.*}}bosc_ame.compute_origin_id = 4 : i64{{.*}}bosc_ame.memory = {{.*}}origin_id = 4 : i64

// FACTS: ========== BOSCAME FACTS REPORT ==========
// FACTS-NEXT: SourceFacts:
// FACTS: program-query kind=program_id axis=0
// FACTS: event id=0 kind=load result=tensor<1024xf32> address=%6:tensor<1024x!tt.ptr<f32>> address-expression=addptr(base=%5:tensor<1024x!tt.ptr<f32>> splat-source=%4:!tt.ptr<f32> offset=%1:tensor<1024xi32>) access-mask=arith.cmpi other=dense<0.000000e+00>
// FACTS: event id=1 kind=load result=tensor<1024xf32> address=%9:tensor<1024x!tt.ptr<f32>> address-expression=addptr(base=%8:tensor<1024x!tt.ptr<f32>> splat-source=%arg1:!tt.ptr<f32> offset=%1:tensor<1024xi32>) access-mask=arith.cmpi other=dense<1.000000e+00>
// FACTS: event id=2 kind=load result=tensor<1024xf32> address=%12:tensor<1024x!tt.ptr<f32>> address-expression=addptr(base=%11:tensor<1024x!tt.ptr<f32>> splat-source=%arg2:!tt.ptr<f32> offset=%1:tensor<1024xi32>) access-mask=arith.cmpi other=dense<0.000000e+00>
// FACTS: event id=3 kind=store
// FACTS: LinalgFacts:
// FACTS: program-argument role="num_programs" axis=0 : i32 type=i32
// FACTS: compute op#{{[0-9]+}} linalg.generic results=[tensor<1024xi1>] body=[arith.cmpi] inputs=[{{.*}}] outputs=[{{.*}}] maps=[{{.*}}] iterators=[{{.*}}parallel{{.*}}] result-users=[
// FACTS: memory-op ir-op#{{[0-9]+}} kind=fill role=padding inputs=[{{.*}}] outputs=[{{.*}}memref<1024xf32>]
// FACTS: memory-event ir-op#[[LOAD_COPY:[0-9]+]] op=memref.copy {{.*}}kind = "load"{{.*}}origin_id = 0 : i64{{.*}}other = 0.000000e+00 : f32{{.*}}valid_shape = array<i64: 768>
// FACTS-NEXT: memory-op ir-op#[[LOAD_COPY]] kind=copy role=load origin-id=0 operands=[ir-op#{{[0-9]+}}:memref.subview.result#0,ir-op#{{[0-9]+}}:memref.subview.result#0]
// FACTS: memory-event {{.*}}kind = "load"{{.*}}origin_id = 1 : i64{{.*}}other = 1.000000e+00 : f32{{.*}}valid_shape = array<i64: 768>
// FACTS: memory-event {{.*}}kind = "load"{{.*}}origin_id = 2 : i64{{.*}}other = 0.000000e+00 : f32{{.*}}valid_shape = array<i64: 768>
// FACTS: memory-op ir-op#{{[0-9]+}} kind=fill role=logical inputs=[{{.*}}:f32] outputs=[ir-op#{{[0-9]+}}:tensor.empty.result#0:tensor<f32>]
// FACTS: compute op#{{[0-9]+}} linalg.reduce dimensions=array<i64: 0> inputs=[{{.*}}] outputs=[{{.*}}] result-users=[result#0 users=[ir-op#{{[0-9]+}}:tensor.extract]] combiner=[arith.addf] input-chain={{.*}}arith.select
// FACTS: compute op#{{[0-9]+}} linalg.generic results=[tensor<1024xf32>] body=[arith.addf] {{.*}} result-users=[result#0 users=[ir-op#{{[0-9]+}}:bufferization.to_buffer]]
// FACTS: memory-event ir-op#[[STORE_COPY:[0-9]+]] op=memref.copy {{.*}}destination = "external_ddr"{{.*}}kind = "store"{{.*}}origin_id = 3 : i64{{.*}}valid_shape = array<i64: 768>
// FACTS-NEXT: memory-op ir-op#[[STORE_COPY]] kind=copy role=store origin-id=3 operands=[ir-op#{{[0-9]+}}:memref.subview.result#0,ir-op#{{[0-9]+}}:memref.subview.result#0]
// FACTS: ir-node[{{[0-9]+}}]: op=linalg.reduce {{.*}} kind=reduction {{.*}} reduction-dimensions=[0] reduction-structure={complete=true input-args=[{{[0-9]+}}] init-args=[{{[0-9]+}}] init-values=[{{[0-9]+}}] yields=[{{[0-9]+}}] reducer-body-ops=[{{[0-9]+}}] output-facts=[{output-index=0 combiner-op={{[0-9]+}} neutral=0.000000e+00 : f32 seed-neutral=matches}]}
// FACTS: ComputeStageGraph:
// FACTS: function triton_layernorm_197x768 complete=true
// FACTS: node id={{[0-9]+}} kind=memory-load stage={{[0-9]+}} predecessors=array<i64: {{[0-9, ]*}}>{{.*}} varying-axes=array<i64: 0>{{.*}} origin-id=0
// FACTS: node id={{[0-9]+}} kind=reduction stage={{[0-9]+}} predecessors=array<i64: {{[0-9, ]+}}>{{.*}} barrier-after=true
// FACTS: node id={{[0-9]+}} kind=elementwise stage={{[0-9]+}} predecessors=array<i64: {{[0-9, ]+}}>{{.*}} zero-tail-prefix=768
// FACTS: ==========================================
// GRAPH-BUFFER: bosc_ame.compute_graph_complete = true
// GRAPH-BUFFER: bosc_ame.compute_graph_node_count = {{[0-9]+}} : i64
// GRAPH-BUFFER-SAME: bosc_ame.source_lineage_complete = true
// GRAPH-BUFFER: linalg.reduce {{.*}}bosc_ame.compute_id = {{[0-9]+}} : i64{{.*}}bosc_ame.compute_kind = "reduction"{{.*}}bosc_ame.compute_predecessors = array<i64:{{.*}}bosc_ame.logical_value_ids = array<i64:{{.*}}bosc_ame.source_load_origin_ids = array<i64: 0>
// GRAPH-BUFFER: linalg.generic {{.*}}bosc_ame.compute_id = {{[0-9]+}} : i64{{.*}}bosc_ame.zero_tail_select_prefix = 768 : i64
// GRAPH-BUFFER: bosc_ame.source_load_origin_ids = array<i64: 0, 1, 2>
// GRAPH-BUFFER: memref.copy {{.*}}bosc_ame.compute_kind = "memory-store"{{.*}}kind = "store"{{.*}}bosc_ame.source_load_origin_ids = array<i64: 0, 1, 2>
// DATAFLOW: BOSCAME plan report:
// DATAFLOW: compute-graph: status=complete tasks=22 dependencies=28 logical-values=29
// DATAFLOW: task[5]: op=linalg.generic kind=memory-load
// DATAFLOW: task[7]: op=linalg.generic kind=memory-load
// DATAFLOW: task[9]: op=linalg.generic kind=memory-load
// DATAFLOW: task[11]: op=linalg.reduce kind=reduction stage=1
// DATAFLOW: task[21]: op=linalg.generic kind=memory-store
// DATAFLOW: logical-value[11]: producer=5 consumers=[11,13] lifetime=known
// DATAFLOW: logical-value[36]: producer=20 consumers=[21] lifetime=known
// DATAFLOW: resources: transfers=28 dma-packets=33 read-bytes=65536 write-bytes=61440 totals-exact=true latency=uncalibrated
// MATMUL-FACTS: SourceFacts:
// MATMUL-FACTS: event id=2 kind=dot {{.*}}accumulator-source=block-argument
// MATMUL-FACTS: event id=3 kind=load result=tensor<16x16xi32>
// MATMUL-FACTS: event id=4 kind=store
// MATMUL-FACTS: compute op#{{[0-9]+}} linalg.matmul inputs=[{{.*}}] outputs=[{{.*}}] maps=[{{.*}}] result-users=[{{.*}}] origin_id=2 : i64 a_origin_id=0 : i64 b_origin_id=1 : i64 a_source_load_origin_ids=array<i64: 0> a_source_lineage_complete=true b_source_load_origin_ids=array<i64: 1> b_source_lineage_complete=true accumulator_source_load_origin_ids=array<i64: 0, 1> accumulator_source_lineage_complete=true
// MATMUL-FACTS: memory-event {{.*}}kind = "load"{{.*}}origin_id = 3 : i64
// MATMUL-FACTS: memory-event {{.*}}destination = "external_ddr"{{.*}}kind = "store"{{.*}}origin_id = 4 : i64
// MATMUL-FACTS-NOT: accumulator_origin_id

// MATMUL-LINEAGE: compute op#{{[0-9]+}} linalg.generic {{.*}} body=[arith.addi]
// MATMUL-LINEAGE: compute op#{{[0-9]+}} linalg.generic {{.*}} body=[arith.addi]
// MATMUL-LINEAGE-NOT: a_origin_id =
// MATMUL-LINEAGE-NOT: b_origin_id =
// MATMUL-LINEAGE: compute op#{{[0-9]+}} linalg.matmul {{.*}} a_source_load_origin_ids=array<i64: 0> a_source_lineage_complete=true b_source_load_origin_ids=array<i64: 1> b_source_lineage_complete=true
// MATMUL-LINEAGE: compute op#{{[0-9]+}} linalg.matmul {{.*}} a_source_load_origin_ids=array<i64> a_source_lineage_complete=false b_source_load_origin_ids=array<i64> b_source_lineage_complete=false
// MATMUL-LINEAGE-NOT: a_origin_id =
// MATMUL-LINEAGE: compute op#{{[0-9]+}} linalg.matmul {{.*}} a_source_load_origin_ids=array<i64> a_source_lineage_complete=false b_source_load_origin_ids=array<i64> b_source_lineage_complete=false
// MATMUL-LINEAGE-NOT: accumulator_origin_id
// MATMUL-LINEAGE-BUFFER: linalg.matmul {{.*}}bosc_ame.a_source_lineage_complete = true{{.*}}bosc_ame.a_source_load_origin_ids = array<i64: 0>{{.*}}bosc_ame.b_source_lineage_complete = true{{.*}}bosc_ame.b_source_load_origin_ids = array<i64: 1>

// The same mask/other on two loads keeps address identity independently
// observable; origin IDs join each source pointer operand to its copy facts.
// SOURCE-ADDR: event id=0 kind=load result=tensor<1024xf32> address=%6:tensor<1024x!tt.ptr<f32>> address-expression=addptr(base=%5:tensor<1024x!tt.ptr<f32>> splat-source=%4:!tt.ptr<f32> offset=%1:tensor<1024xi32>) access-mask=arith.cmpi other=dense<0.000000e+00>
// SOURCE-ADDR: event id=2 kind=load result=tensor<1024xf32> address=%12:tensor<1024x!tt.ptr<f32>> address-expression=addptr(base=%11:tensor<1024x!tt.ptr<f32>> splat-source=%arg2:!tt.ptr<f32> offset=%1:tensor<1024xi32>) access-mask=arith.cmpi other=dense<0.000000e+00>
// SOURCE-ADDR: memory-event ir-op#[[LOAD_A:[0-9]+]] op=memref.copy {{.*}}kind = "load"{{.*}}origin_id = 0 : i64
// SOURCE-ADDR-NEXT: memory-op ir-op#[[LOAD_A]] kind=copy role=load origin-id=0 operands=[ir-op#{{[0-9]+}}:memref.subview.result#0,ir-op#{{[0-9]+}}:memref.subview.result#0]
// SOURCE-ADDR: memory-event ir-op#[[LOAD_B:[0-9]+]] op=memref.copy {{.*}}kind = "load"{{.*}}origin_id = 2 : i64
// SOURCE-ADDR-NEXT: memory-op ir-op#[[LOAD_B]] kind=copy role=load origin-id=2 operands=[ir-op#{{[0-9]+}}:memref.subview.result#0,ir-op#{{[0-9]+}}:memref.subview.result#0]

// ADDRESS-EXPR: event id=0 kind=load result=tensor<32xf32> address=%4:tensor<32x!tt.ptr<f32>> address-expression=addptr(base=%2:tensor<32x!tt.ptr<f32>> splat-source=%arg0:!tt.ptr<f32> offset=%0:tensor<32xi32>)
// ADDRESS-EXPR: event id=1 kind=store address=%5:tensor<32x!tt.ptr<f32>> address-expression=addptr(base=%3:tensor<32x!tt.ptr<f32>> splat-source=%arg1:!tt.ptr<f32> offset=%0:tensor<32xi32>)
// ADDRESS-UNKNOWN: event id=0 kind=load result=f32 address=%0:!tt.ptr<f32> address-expression=unknown access-mask={{.*}} other=ssa({{%[a-zA-Z0-9]+}}:f32)

// BAD-TILE: BOSCAME tile sizes must be positive
// BAD-TARGET: unsupported BOSCAME target; expected nr-fpga
// BAD-FACTS: dump-facts requires boscame=true

// The shared SSA snapshot joins event direction and source IDs to the concrete
// copy/fill operands without asserting alias or memory-content equivalence.
// MEMORY-DAG: op=memref.copy {{.*}}memory-event={kind=copy role=load effect=read origin-id=0 inputs=[{{[0-9]+}}] outputs=[{{[0-9]+}}]}
// MEMORY-DAG: op=memref.copy {{.*}}memory-event={kind=copy role=store effect=write origin-id=3 inputs=[{{[0-9]+}}] outputs=[{{[0-9]+}}]}
// MEMORY-DAG: op=linalg.fill {{.*}}memory-event={kind=fill role=padding effect=write origin-id=<unknown> inputs=[{{[0-9]+}}] outputs=[{{[0-9]+}}]}
// MEMORY-DAG: op=linalg.fill {{.*}}memory-event={kind=fill role=logical effect=write origin-id=<unknown> inputs=[{{[0-9]+}}] outputs=[{{[0-9]+}}]}
