// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

module {
  memref.global "private" @left_global : memref<16xf32> = uninitialized
  memref.global "private" @right_global : memref<16xf32> = uninitialized

  func.func @distinct_arguments(%left: memref<16xf32>, %right: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0] : memref<16xf32>
    %rightValue = memref.load %right[%c0] : memref<16xf32>
    %leftGlobal = memref.get_global @left_global : memref<16xf32>
    %leftGlobalValue = memref.load %leftGlobal[%c0] : memref<16xf32>
    %rightGlobal = memref.get_global @right_global : memref<16xf32>
    %rightGlobalValue = memref.load %rightGlobal[%c0] : memref<16xf32>
    return
  }

  func.func @called_helper(%left: memref<16xf32>, %right: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0] : memref<16xf32>
    %rightValue = memref.load %right[%c0] : memref<16xf32>
    return
  }

  func.func @caller(%buffer: memref<16xf32>) {
    func.call @called_helper(%buffer, %buffer)
      : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }

  func.func @custom_space_arguments(%left: memref<16xf32, 1>,
                                    %right: memref<16xf32, 1>) {
    %c0 = arith.constant 0 : index
    %leftValue = memref.load %left[%c0] : memref<16xf32, 1>
    %rightValue = memref.load %right[%c0] : memref<16xf32, 1>
    return
  }
}
