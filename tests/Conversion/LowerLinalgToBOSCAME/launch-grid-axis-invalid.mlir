// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

module {
  func.func @bad_axes(
      %grid_x: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
      %grid_y: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
      %grid_z: i32 {bosc_ame.program_axis = 5 : i32, bosc_ame.program_role = "num_programs"}) {
    return
  }
}
