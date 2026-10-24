// Legacy modules carrying the removed global policy receive a migration
// diagnostic before planning. Route selection is now automatic per access.
// RUN: not buddy-opt %s --linalg-to-boscame 2>&1 | FileCheck %s
// RUN: not buddy-opt %s --lower-linalg-to-boscame 2>&1 | FileCheck %s
// RUN: not buddy-opt %s --bosc-ame-plan 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: not buddy-opt %s --linalg-to-boscame='store-policy=spm-dma' 2>&1 | FileCheck %s --check-prefix=OPTION
// RUN: not buddy-opt %s --lower-linalg-to-boscame='store-policy=spm-dma' 2>&1 | FileCheck %s --check-prefix=OPTION
// RUN: not buddy-opt %s --bosc-ame-plan='store-policy=spm-dma' 2>&1 | FileCheck %s --check-prefix=OPTION
// RUN: not buddy-opt %s --linalg-to-boscame-store=direct-ddr 2>&1 | FileCheck %s --check-prefix=ALIAS

module attributes {bosc_ame.store_policy = "direct-ddr"} {
}

// CHECK: bosc_ame.store_policy is obsolete; remove it and rerun the automatic BOSCAME planner
// PLAN: bosc_ame.store_policy is obsolete; remove it and replan
// OPTION: no such option store-policy
// ALIAS: Unknown command line argument '--linalg-to-boscame-store=direct-ddr'
