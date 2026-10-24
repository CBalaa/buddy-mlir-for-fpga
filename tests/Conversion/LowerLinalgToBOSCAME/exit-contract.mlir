// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// Unsupported scalar operations and unused declarations must fail even when
// there is no tensor SSA or staged Linalg operation to trigger the planner.

//--- triton.mlir
module {
  func.func @residual() -> i32 {
    %id = tt.get_program_id x : i32
    return %id : i32
  }
}

//--- structured.mlir
module {
  func.func @residual(%base: !tt.ptr<f32>) {
    %ptr = tts.make_tptr %base to sizes: [16], strides: [1], offsets: [0], shape: [16], order: [0] : !tt.ptr<f32> to !tt.ptr<f32>
    return
  }
}

//--- clone.mlir
module {
  func.func @residual(%base: memref<16xf32>) -> memref<16xf32> {
    %copy = bufferization.clone %base : memref<16xf32> to memref<16xf32>
    return %copy : memref<16xf32>
  }
}

//--- gather.mlir
module {
  func.func @gather(%base: memref<16xf32>, %indices: vector<16xi32>, %mask: vector<16xi1>, %pass: vector<16xf32>) -> vector<16xf32> {
    %c0 = arith.constant 0 : index
    %v = vector.gather %base[%c0][%indices], %mask, %pass : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    return %v : vector<16xf32>
  }
}

//--- scatter.mlir
module {
  func.func @scatter(%base: memref<16xf32>, %indices: vector<16xi32>, %mask: vector<16xi1>, %v: vector<16xf32>) {
    %c0 = arith.constant 0 : index
    vector.scatter %base[%c0][%indices], %mask, %v : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
    return
  }
}

//--- global.mlir
module {
  memref.global "private" @invalid : memref<16xf32, #bosc_ame.spm<0>> = uninitialized
}

//--- ordinary.mlir
module {
  func.func @ordinary(%out: memref<4xf32>) {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%out : memref<4xf32>)
    return
  }
}
