// RUN: %PYTHON %S/Inputs/check_legacy_grid.py %s %t buddy-opt

// Valid roles and known grid extents survive target-to-neutral adaptation.
// Missing/invalid axes are explicit migration errors, never guessed from names.
module {
  func.func @known(
      %grid_x: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
      %grid_y: i32 {bosc_ame.program_axis = 1 : i32, bosc_ame.program_role = "num_programs"},
      %grid_z: i32 {bosc_ame.program_axis = 2 : i32, bosc_ame.program_role = "num_programs"})
      attributes {gpu.known_grid_size = array<i32: 197, 1, 1>} {
    return
  }

  func.func @unknown(
      %grid_x: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
      %grid_y: i32 {bosc_ame.program_axis = 1 : i32, bosc_ame.program_role = "num_programs"},
      %grid_z: i32 {bosc_ame.program_axis = 2 : i32, bosc_ame.program_role = "num_programs"}) {
    return
  }

  func.func @partial_shuffled(
      %pid: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "program_id"},
      %grid_z: i32 {bosc_ame.program_axis = 2 : i32, bosc_ame.program_role = "num_programs"},
      %grid_x: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"})
      attributes {gpu.known_grid_size = array<i32: 5, 6, 7>} {
    return
  }

  func.func @contract_only()
      attributes {gpu.known_grid_size = array<i32: 8, 9, 10>} {
    return
  }

  func.func @missing_axis(
      %grid: i32 {bosc_ame.program_role = "num_programs"}) {
    return
  }

  func.func @missing_axis_with_contract(
      %grid: i32 {bosc_ame.program_role = "num_programs"})
      attributes {gpu.known_grid_size = array<i32: 11, 12, 13>} {
    return
  }

  // Names and positions do not establish launch roles.
  func.func @unannotated(%grid_x: i32, %grid_y: i32, %grid_z: i32) {
    return
  }

  func.func @program_id_only(
      %pid: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "program_id"}) {
    return
  }

  func.func @unrecognized_role(
      %value: i32 {bosc_ame.program_axis = 9 : i32, bosc_ame.program_role = "opaque"}) {
    return
  }

  func.func private @declaration(
      i32 {bosc_ame.program_axis = 1 : i32, bosc_ame.program_role = "num_programs"})
      attributes {gpu.known_grid_size = array<i32: 14, 15, 16>}
}
