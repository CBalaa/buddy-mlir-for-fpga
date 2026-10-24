// The migration diagnostic also applies to an already planned neutral module.
// It must run before the compatibility entry point dispatches to plan emission.
// RUN: not buddy-opt %s --linalg-to-boscame 2>&1 | FileCheck %s
// RUN: not buddy-opt %s --bosc-ame-plan-verify 2>&1 | FileCheck %s
// RUN: not buddy-opt %s --bosc-ame-emit 2>&1 | FileCheck %s

module attributes {
  contract.version = 1 : i32,
  contract.producer = "migration-test",
  plan.version = 1 : i32,
  plan.target_profile = "nr-fpga",
  bosc_ame.store_policy = "direct-ddr"
} {
}

// CHECK: bosc_ame.store_policy is obsolete; remove it and rerun the automatic BOSCAME planner
