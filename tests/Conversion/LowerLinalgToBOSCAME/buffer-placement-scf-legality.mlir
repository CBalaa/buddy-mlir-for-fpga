// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// Yielded/loop-carried descriptors need an inter-region alias proof that the
// fixed private-root path deliberately does not claim. Calls and multi-block
// regions also retain staging. No partial placement is applied to these roots.

//--- escape.mlir
module {
  func.func @escaping_branch(%condition: i1, %out: memref<16xf32>) {
    %zero = arith.constant 0.0 : f32
    %escaped = scf.if %condition -> memref<16xf32> {
      %local = memref.alloc() alignment = 64 : memref<16xf32>
      linalg.fill ins(%zero : f32) outs(%local : memref<16xf32>)
      scf.yield %local : memref<16xf32>
    } else {
      scf.yield %out : memref<16xf32>
    }
    memref.copy %escaped, %out : memref<16xf32> to memref<16xf32>
    return
  }
}

//--- iter-arg.mlir
module {
  func.func @alias_through_iter_arg(%out: memref<16xf32>, %count: index) {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %local = memref.alloc() alignment = 64 : memref<16xf32>
    linalg.fill ins(%zero : f32) outs(%local : memref<16xf32>)
    %alias = scf.for %i = %c0 to %count step %c1
        iter_args(%current = %local) -> memref<16xf32> {
      scf.yield %current : memref<16xf32>
    }
    memref.copy %alias, %out : memref<16xf32> to memref<16xf32>
    return
  }
}

//--- call.mlir
module {
  func.func private @unknown(memref<16xf32>)
  func.func @escaping_call() {
    %zero = arith.constant 0.0 : f32
    %local = memref.alloc() alignment = 64 : memref<16xf32>
    linalg.fill ins(%zero : f32) outs(%local : memref<16xf32>)
    func.call @unknown(%local) : (memref<16xf32>) -> ()
    return
  }
}

//--- blocks.mlir
module {
  func.func @multiple_blocks(%condition: i1) {
    %zero = arith.constant 0.0 : f32
    %local = memref.alloc() alignment = 64 : memref<16xf32>
    scf.execute_region {
      cf.cond_br %condition, ^then, ^exit
    ^then:
      linalg.fill ins(%zero : f32) outs(%local : memref<16xf32>)
      cf.br ^exit
    ^exit:
      scf.yield
    }
    return
  }
}

//--- execute.mlir
module {
  func.func @single_block_region() {
    %zero = arith.constant 0.0 : f32
    scf.execute_region {
      %local = memref.alloc() alignment = 64 : memref<16xf32>
      linalg.fill ins(%zero : f32) outs(%local : memref<16xf32>)
      scf.yield
    }
    return
  }
}

//--- parallel.mlir
module {
  func.func @parallel_private_root() {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      %local = memref.alloc() alignment = 64 : memref<16xf32>
      linalg.fill ins(%zero : f32) outs(%local : memref<16xf32>)
      scf.reduce
    }
    return
  }
}
