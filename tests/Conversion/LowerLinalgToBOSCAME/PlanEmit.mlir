// RUN: buddy-opt %S/../../../plan/examples/matmul128.planned.mlir --bosc-ame-plan-verify --bosc-ame-emit | FileCheck %s --check-prefix=MATMUL --implicit-check-not=linalg.matmul --implicit-check-not=bosc_ame.spm.view --implicit-check-not=bosc_ame.dma.start --implicit-check-not=plan.route
// RUN: buddy-opt %S/../../../plan/examples/kernel.planned.mlir --bosc-ame-emit | FileCheck %s --check-prefix=MIXED --implicit-check-not=linalg.matmul --implicit-check-not=plan.route
// RUN: buddy-opt %S/../../../plan/examples/layernorm.planned.mlir --bosc-ame-emit | FileCheck %s --check-prefix=NORM --implicit-check-not=linalg. --implicit-check-not=plan.vector
// RUN: buddy-opt %S/../../../plan/examples/matmul128.planned.mlir --bosc-ame-emit --lower-bosc-ame -o /dev/null
// RUN: buddy-opt %S/../../../plan/examples/kernel.planned.mlir --bosc-ame-emit --lower-bosc-ame -o /dev/null
// RUN: buddy-opt %S/../../../plan/examples/layernorm.planned.mlir --bosc-ame-emit --lower-bosc-ame -o /dev/null

// MATMUL-LABEL: func.func @matmul128
// MATMUL: bosc_ame.tile.load_acc
// MATMUL-SAME: bosc_ame.direct_ddr
// MATMUL-COUNT-2: bosc_ame.tile.mma
// MATMUL: bosc_ame.tile.store_acc
// MATMUL-SAME: contract.task

// MIXED-LABEL: func.func @triton_matmul_i8_196x768x768
// MIXED: bosc_ame.spm.view
// MIXED: bosc_ame.dma.start
// MIXED: bosc_ame.dma.wait
// MIXED: bosc_ame.tile.load_b
// MIXED-SAME: bosc_ame.direct_ddr
// MIXED: bosc_ame.tile.mma
// MIXED: bosc_ame.tile.store_acc {{.*}}contract.task

// NORM-LABEL: func.func @layernorm_197x768
// NORM: bosc_ame.spm.view
// NORM: memref.alloca
// NORM: vector.broadcast
// NORM: bosc_ame.dma.start
// NORM: bosc_ame.dma.wait
// NORM-SAME: contract.task
// NORM: vector.reduction <add>
// NORM: memref.store
// NORM-SAME: contract.task
// NORM: math.rsqrt
// NORM: arith.mulf
