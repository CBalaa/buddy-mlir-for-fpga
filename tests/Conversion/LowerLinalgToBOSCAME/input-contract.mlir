// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// The backend consumes an explicitly bufferized program.  It must reject
// incomplete bufferization without inserting a second implicit bufferization
// pipeline or accepting a cast as a data movement operation.

//--- tensor.mlir
module {
  func.func @tensor(%a: tensor<16xf32>) -> tensor<16xf32> {
    return %a : tensor<16xf32>
  }
}

//--- bridge.mlir
module {
  func.func @bridge(%a: memref<16xf32>) -> tensor<16xf32> {
    %tensor = bufferization.to_tensor %a restrict : memref<16xf32> to tensor<16xf32>
    return %tensor : tensor<16xf32>
  }
}

//--- unused-tensor.mlir
module {
  func.func @unused_tensor(%unused: tensor<16xf32>) {
    return
  }
}

//--- unrealized.mlir
module {
  func.func @unrealized(%a: memref<16xf32>) {
    %cast = builtin.unrealized_conversion_cast %a : memref<16xf32> to memref<16xf32>
    return
  }
}

//--- external-tensor.mlir
module {
  func.func private @external_tensor(tensor<16xf32>) -> tensor<16xf32>
}
