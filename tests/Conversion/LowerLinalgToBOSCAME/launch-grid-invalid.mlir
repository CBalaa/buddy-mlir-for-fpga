// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

//--- nonpositive.mlir
module {
  func.func @bad_extent() attributes {gpu.known_grid_size = array<i32: 1, 0, 1>} {
    return
  }
}

//--- wrong-rank.mlir
module {
  func.func @bad_rank() attributes {gpu.known_grid_size = array<i32: 1, 2>} {
    return
  }
}

//--- wrong-type.mlir
module {
  func.func @bad_type() attributes {gpu.known_grid_size = array<i64: 1, 2, 3>} {
    return
  }
}
