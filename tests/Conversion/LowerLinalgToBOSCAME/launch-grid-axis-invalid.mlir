// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t 2>&1 | FileCheck %s
// RUN: not buddy-opt %s --lower-linalg-to-boscame -o %t 2>&1 | FileCheck %s

module {
  func.func @bad_axes(
      %grid_x: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
      %grid_y: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
      %grid_z: i32 {bosc_ame.program_axis = 5 : i32, bosc_ame.program_role = "num_programs"}) {
    return
  }
}

// CHECK-COUNT-2: invalid BOSCAME num_programs axis; expected unique axes in [0,2]
