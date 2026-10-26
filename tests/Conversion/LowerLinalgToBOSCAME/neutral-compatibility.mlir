// RUN: buddy-opt %S/Inputs/contracts/matmul128.contract.mlir --linalg-to-boscame='plan-report=true' 2>&1 | FileCheck %s
// RUN: buddy-opt %S/Inputs/contracts/matmul128.contract.mlir --lower-linalg-to-boscame='plan-report=true' 2>&1 | FileCheck %s
// RUN: buddy-opt %S/Inputs/contracts/matmul128.planned.mlir --linalg-to-boscame | FileCheck %s --check-prefix=EMIT
// RUN: buddy-opt %S/Inputs/contracts/matmul128.planned.mlir --lower-linalg-to-boscame | FileCheck %s --check-prefix=EMIT
// RUN: not buddy-opt %S/Inputs/contracts/matmul128.contract.mlir --linalg-to-boscame='target=unsupported' 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: not buddy-opt %S/Inputs/contracts/matmul128.contract.mlir --lower-linalg-to-boscame='bm=1' 2>&1 | FileCheck %s --check-prefix=GEOMETRY

// CHECK-DAG: plan.budget=
// CHECK-DAG: plan.route=
// CHECK: func.func @matmul128
// CHECK: bosc_ame.tile.mma
// CHECK: bosc_ame.tile.store_acc
// EMIT: func.func @matmul128
// EMIT: bosc_ame.tile.mma
// EMIT: bosc_ame.tile.store_acc
// TARGET: unsupported BOSCAME target; expected nr-fpga
// GEOMETRY: legacy bm/bn/bk overrides are obsolete; geometry is taken from buffer-form Linalg operands
