// RUN: split-file %s %t
// RUN: not buddy-opt %t/spm-input.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=SPM-INPUT
// RUN: not buddy-opt %t/spm-result.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=SPM-RESULT
// RUN: not buddy-opt %t/parallel.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=FILL
// RUN: not buddy-opt %t/forall.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=FILL
// RUN: not buddy-opt %t/async.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=ASYNC
// RUN: not buddy-opt %t/parallel-call.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=FILL
// RUN: not buddy-opt %t/serial.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=FILL

// Preserve each original source. Physical SPM formals/results do not have
// neutral private-root ownership, async tokens are outside the contract,
// and external fills have no independent V1 recipe even in a serial loop.
// Concurrent private workspace and serial acceptance are covered by
// PlanWorkspaceScopes.py; interprocedural ownership by PlanWorkspaceCalls.py.
// SPM-INPUT: target memory must be rooted in a verified SPM window
// SPM-RESULT: kernel contract V14: target attributes are forbidden in the neutral contract
// ASYNC: compatibility input contains a non-neutral dialect or bridge operation
// FILL: unplanned external fill

//--- spm-input.mlir
module {
  func.func @spm_input(%input: memref<16xf32, #bosc_ame.spm<0>>,
                       %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32, #bosc_ame.spm<0>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }
}

//--- spm-result.mlir
module {
  func.func private @spm_result() -> memref<16xf32, #bosc_ame.spm<0>>
}

//--- parallel.mlir
module {
  func.func @parallel(%out: memref<2x16xf32>) {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      %row = memref.subview %out[%i, 0] [1, 16] [1, 1]
        : memref<2x16xf32> to memref<16xf32, strided<[1], offset: ?>>
      linalg.fill ins(%zero : f32)
        outs(%row : memref<16xf32, strided<[1], offset: ?>>)
      scf.reduce
    }
    return
  }
}

//--- forall.mlir
module {
  func.func @forall(%out: memref<2x16xf32>) {
    %zero = arith.constant 0.0 : f32
    scf.forall (%i) in (2) {
      %row = memref.subview %out[%i, 0] [1, 16] [1, 1]
        : memref<2x16xf32> to memref<16xf32, strided<[1], offset: ?>>
      linalg.fill ins(%zero : f32)
        outs(%row : memref<16xf32, strided<[1], offset: ?>>)
      scf.forall.in_parallel {
      }
    }
    return
  }
}

//--- async.mlir
module {
  func.func @asynchronous(%out: memref<16xf32>) {
    %zero = arith.constant 0.0 : f32
    %token = async.execute {
      linalg.fill ins(%zero : f32) outs(%out : memref<16xf32>)
      async.yield
    }
    async.await %token : !async.token
    return
  }
}

//--- serial.mlir
module {
  func.func @serial(%out: memref<16xf32>) {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      linalg.fill ins(%zero : f32) outs(%out : memref<16xf32>)
    }
    return
  }
}


//--- parallel-call.mlir
module {
  func.func private @staged(%out: memref<16xf32>) {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%out : memref<16xf32>)
    return
  }

  func.func private @wrapper(%out: memref<16xf32>) {
    func.call @staged(%out) : (memref<16xf32>) -> ()
    return
  }

  func.func @parallel_call() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      // Each source invocation owns distinct DDR storage. Their staged
      // transitive callee would nevertheless share the fixed SPM workspace.
      %local = memref.alloc() alignment = 64 : memref<16xf32>
      func.call @wrapper(%local) : (memref<16xf32>) -> ()
      memref.dealloc %local : memref<16xf32>
      scf.reduce
    }
    return
  }
}
