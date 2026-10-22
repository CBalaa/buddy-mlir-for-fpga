// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t 2>%t.plan
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t.default
// RUN: cmp %t.default %t
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan

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

// PLAN: launch-grid: function=@known axis[0]=runtime(arg#0) extent=197 axis[1]=runtime(arg#1) extent=1 axis[2]=runtime(arg#2) extent=1
// PLAN: launch-grid: function=@unknown axis[0]=runtime(arg#0) extent=<unknown> axis[1]=runtime(arg#1) extent=<unknown> axis[2]=runtime(arg#2) extent=<unknown>
// PLAN: launch-grid: function=@partial_shuffled axis[0]=runtime(arg#2) extent=5 axis[2]=runtime(arg#1) extent=7
// PLAN: launch-grid: function=@contract_only known-grid=[8,9,10]
// PLAN: launch-grid: function=@missing_axis axis=<unknown>:runtime(arg#0) extent=<unknown>
// PLAN: launch-grid: function=@missing_axis_with_contract axis=<unknown>:runtime(arg#0) extent=<unknown> known-grid=[11,12,13]
// PLAN: launch-grid: function=@unannotated unavailable
// PLAN: launch-grid: function=@program_id_only unavailable
// PLAN: launch-grid: function=@unrecognized_role unavailable
// PLAN: launch-grid: function=@declaration axis[1]=runtime(arg#0) extent=15
