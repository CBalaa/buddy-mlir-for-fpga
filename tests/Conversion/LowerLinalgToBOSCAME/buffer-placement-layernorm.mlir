// RUN: not buddy-opt %S/Inputs/layernorm-197x768-bufferized.mlir --linalg-to-boscame -o /dev/null 2>&1 | FileCheck %s --check-prefix=LEGACY
// RUN: buddy-opt %S/../../../plan/examples/layernorm.contract.mlir --bosc-ame-plan='plan-report=true' --bosc-ame-emit -o %t 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: FileCheck %s --check-prefix=IR < %t
// RUN: buddy-opt %t --bosc-ame-spm-dma-check -o /dev/null

// The retained historical frontend artifact needs regeneration: its global
// geometry/target configuration is not a neutral contract. The neutral design
// reference uses six conservative 4 KiB windows; actual safe bufferization may
// share a dead broadcast root, which is checked by the frontend numerical test.
// LEGACY: obsolete geometry attribute bosc_ame.bm
// PLAN-COUNT-6: memref.alloc plan.buffer=
// PLAN: high_water = array<i64: 12288, 12288>
// IR-COUNT-6: bosc_ame.spm.view
// IR-COUNT-4: bosc_ame.dma.start
// IR-NOT: bosc_ame.dma.start
