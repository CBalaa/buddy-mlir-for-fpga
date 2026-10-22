// RUN: buddy-opt %s --split-input-file --linalg-to-boscame='plan-report=true dma-abi=args_aligned64_nonalias' -o %t.ir 2>%t.plan
// RUN: buddy-opt %s --split-input-file --linalg-to-boscame='dma-abi=args_aligned64_nonalias' -o %t.default
// RUN: cmp %t.default %t.ir
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: buddy-opt %s --split-input-file --linalg-to-boscame='plan-report=true' -o %t.noabi.ir 2>%t.noabi.plan
// RUN: FileCheck %s --check-prefix=NOABI < %t.noabi.plan

// Data dependencies, overwrite ordering, and read-before-overwrite ordering
// are distinct hazards. Two reads have no memory ordering requirement.
module {
  func.func @raw(%root: memref<16xf32>, %value: f32) -> f32 {
    %c0 = arith.constant 0 : index
    memref.store %value, %root[%c0] : memref<16xf32>
    %read = memref.load %root[%c0] : memref<16xf32>
    return %read : f32
  }

  func.func @war(%root: memref<16xf32>, %value: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %read = memref.load %root[%c0] : memref<16xf32>
    memref.store %value, %root[%c0] : memref<16xf32>
    return %read : f32
  }

  func.func @waw(%root: memref<16xf32>, %first: f32, %second: f32) {
    %c0 = arith.constant 0 : index
    memref.store %first, %root[%c0] : memref<16xf32>
    memref.store %second, %root[%c0] : memref<16xf32>
    return
  }

  func.func @read_read(%root: memref<16xf32>) -> (f32, f32) {
    %c0 = arith.constant 0 : index
    %first = memref.load %root[%c0] : memref<16xf32>
    %second = memref.load %root[%c0] : memref<16xf32>
    return %first, %second : f32, f32
  }

  func.func @disjoint_static_views(%root: memref<32xf32>, %value: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %left = memref.subview %root[0] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1]>>
    %right = memref.subview %root[16] [16] [1]
      : memref<32xf32> to memref<16xf32, strided<[1], offset: 16>>
    memref.store %value, %left[%c0] : memref<16xf32, strided<[1]>>
    %read = memref.load %right[%c0]
      : memref<16xf32, strided<[1], offset: 16>>
    return %read : f32
  }
}

// PLAN: buffer-accesses: count=10 scope=post-planning
// PLAN: memory-dependencies: count=3 scope=lexical status=complete
// PLAN-NEXT: memory-dependency[0]: function=@raw accesses=[0,1] kind=RAW
// PLAN-NEXT: memory-dependency[1]: function=@war accesses=[2,3] kind=WAR
// PLAN-NEXT: memory-dependency[2]: function=@waw accesses=[4,5] kind=WAW
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// Single-block execute_region ops run exactly once, so their direct accesses
// have a lexical order within each region and across sibling wrappers.
module {
  func.func @execute_region_order(%root: memref<16xf32>, %value: f32) -> f32 {
    %c0 = arith.constant 0 : index
    memref.store %value, %root[%c0] : memref<16xf32>
    scf.execute_region {
      %inside = memref.load %root[%c0] : memref<16xf32>
      memref.store %inside, %root[%c0] : memref<16xf32>
      scf.yield
    }
    scf.execute_region {
      %between = memref.load %root[%c0] : memref<16xf32>
      scf.yield
    }
    %after = memref.load %root[%c0] : memref<16xf32>
    return %after : f32
  }
}

// PLAN: buffer-accesses: count=5 scope=post-planning
// PLAN: memory-dependencies: count=7 scope=lexical status=complete
// PLAN-NEXT: memory-dependency[0]: function=@execute_region_order accesses=[0,1] kind=RAW
// PLAN-NEXT: memory-dependency[1]: function=@execute_region_order accesses=[0,2] kind=WAW
// PLAN-NEXT: memory-dependency[2]: function=@execute_region_order accesses=[0,3] kind=RAW
// PLAN-NEXT: memory-dependency[3]: function=@execute_region_order accesses=[0,4] kind=RAW
// PLAN-NEXT: memory-dependency[4]: function=@execute_region_order accesses=[1,2] kind=WAR
// PLAN-NEXT: memory-dependency[5]: function=@execute_region_order accesses=[2,3] kind=RAW
// PLAN-NEXT: memory-dependency[6]: function=@execute_region_order accesses=[2,4] kind=RAW
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// An unresolved call is an opaque memory barrier. Its conservative edges
// order both sides without claiming to analyze the callee's internal effects.
module {
  func.func private @opaque()

  func.func @unknown_barrier(%root: memref<16xf32>, %value: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %read = memref.load %root[%c0] : memref<16xf32>
    func.call @opaque() : () -> ()
    memref.store %value, %root[%c0] : memref<16xf32>
    return %read : f32
  }
}

// PLAN: buffer-accesses: count=3 scope=post-planning
// PLAN: memory-dependencies: count=3 scope=lexical status=complete
// PLAN-DAG: memory-dependency[{{[0-9]+}}]: function=@unknown_barrier accesses=[0,1] kind=unknown-effect
// PLAN-DAG: memory-dependency[{{[0-9]+}}]: function=@unknown_barrier accesses=[0,2] kind=WAR
// PLAN-DAG: memory-dependency[{{[0-9]+}}]: function=@unknown_barrier accesses=[1,2] kind=unknown-effect
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// Walk order does not establish execution order across structured regions.
// A branch's access leaves coverage incomplete, without a fabricated edge.
module {
  func.func @nested_scf(%root: memref<16xf32>, %value: f32,
                        %condition: i1) -> f32 {
    %c0 = arith.constant 0 : index
    memref.store %value, %root[%c0] : memref<16xf32>
    %result = scf.if %condition -> (f32) {
      %read = memref.load %root[%c0] : memref<16xf32>
      scf.yield %read : f32
    } else {
      scf.yield %value : f32
    }
    return %result : f32
  }
}

// PLAN: buffer-accesses: count=2 scope=post-planning
// PLAN: memory-dependencies: count=0 scope=lexical status=incomplete
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// Storage allocation and release carry lifetime ordering as well as accesses.
module {
  func.func @allocation_lifetime(%value: f32) {
    %c0 = arith.constant 0 : index
    %buffer = memref.alloc() : memref<16xf32>
    memref.store %value, %buffer[%c0] : memref<16xf32>
    memref.dealloc %buffer : memref<16xf32>
    return
  }
}

// PLAN: buffer-accesses: count=3 scope=post-planning
// PLAN: memory-dependencies: count=3 scope=lexical status=complete
// PLAN-DAG: memory-dependency[{{[0-9]+}}]: function=@allocation_lifetime accesses=[0,1] kind=lifetime
// PLAN-DAG: memory-dependency[{{[0-9]+}}]: function=@allocation_lifetime accesses=[0,2] kind=lifetime
// PLAN-DAG: memory-dependency[{{[0-9]+}}]: function=@allocation_lifetime accesses=[1,2] kind=lifetime
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// Empty storage still has allocation/release ordering even though its byte
// interval is empty. Lifetime proofs follow storage identity, not payload.
module {
  func.func @empty_allocation_lifetime() {
    %buffer = memref.alloc() : memref<0xf32>
    memref.dealloc %buffer : memref<0xf32>
    return
  }
}

// PLAN: buffer-accesses: count=2 scope=post-planning
// PLAN: memory-dependencies: count=1 scope=lexical status=complete
// PLAN-NEXT: memory-dependency[0]: function=@empty_allocation_lifetime accesses=[0,1] kind=lifetime
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// Entry nonalias contracts exclude hazards across distinct entry arguments.
// A called helper may receive the same actual buffer for both formal roots;
// missing pairwise alias facts therefore cannot establish disjointness.
module {
  func.func @distinct_arguments(%left: memref<16xf32>,
                                 %right: memref<16xf32>, %value: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %read = memref.load %left[%c0] : memref<16xf32>
    memref.store %value, %right[%c0] : memref<16xf32>
    return %read : f32
  }

  func.func @called_helper(%left: memref<16xf32>,
                            %right: memref<16xf32>, %value: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %read = memref.load %left[%c0] : memref<16xf32>
    memref.store %value, %right[%c0] : memref<16xf32>
    return %read : f32
  }

  func.func @caller(%root: memref<16xf32>, %value: f32) -> f32 {
    %result = func.call @called_helper(%root, %root, %value)
      : (memref<16xf32>, memref<16xf32>, f32) -> f32
    return %result : f32
  }
}

// PLAN: memory-dependencies: count=1 scope=lexical status=complete
// PLAN-NEXT: memory-dependency[0]: function=@called_helper accesses=[2,3] kind=WAR
// PLAN-NOT: memory-dependency[
// PLAN: resources:
// NOABI: memory-dependency[{{[0-9]+}}]: function=@distinct_arguments accesses=[0,1] kind=WAR
// NOABI: memory-dependency[{{[0-9]+}}]: function=@called_helper accesses=[2,3] kind=WAR

// -----

// Pointer effects remain physical accesses even without a memref descriptor.
// The entry ABI cannot prove a raw pointer disjoint from a memref argument.
module {
  func.func @mixed_pointer(%pointer: !llvm.ptr, %root: memref<16xf32>,
                          %value: f32) {
    %c0 = arith.constant 0 : index
    %read = llvm.load %pointer : !llvm.ptr -> f32
    memref.store %read, %root[%c0] : memref<16xf32>
    llvm.store %value, %pointer : f32, !llvm.ptr
    return
  }
}

// PLAN: buffer-accesses: count=3 scope=post-planning
// PLAN: memory-dependencies: count=3 scope=lexical status=complete
// PLAN-NEXT: memory-dependency[0]: function=@mixed_pointer accesses=[0,1] kind=WAR
// PLAN-NEXT: memory-dependency[1]: function=@mixed_pointer accesses=[0,2] kind=WAR
// PLAN-NEXT: memory-dependency[2]: function=@mixed_pointer accesses=[1,2] kind=WAW
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// Nested effects inside an otherwise supported execute_region remain outside
// the lexical-order model.
module {
  func.func @nested_execute_region(%root: memref<16xf32>, %value: f32,
                                   %condition: i1) {
    %c0 = arith.constant 0 : index
    memref.store %value, %root[%c0] : memref<16xf32>
    scf.execute_region {
      scf.if %condition {
        %inside = memref.load %root[%c0] : memref<16xf32>
      }
      scf.yield
    }
    return
  }
}

// PLAN: buffer-accesses: count=2 scope=post-planning
// PLAN: memory-dependencies: count=0 scope=lexical status=incomplete
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// A multi-block function remains incomplete, but local order within one CFG
// block is still reported when it is independently established.
module {
  func.func @multiblock_function(%root: memref<16xf32>, %value: f32) {
    cf.br ^body
  ^body:
    %c0 = arith.constant 0 : index
    memref.store %value, %root[%c0] : memref<16xf32>
    memref.store %value, %root[%c0] : memref<16xf32>
    return
  }
}

// PLAN: buffer-accesses: count=2 scope=post-planning
// PLAN: memory-dependencies: count=1 scope=lexical status=incomplete
// PLAN-NEXT: memory-dependency[0]: function=@multiblock_function accesses=[0,1] kind=WAW
// PLAN-NOT: memory-dependency[
// PLAN: resources:

// -----

// A multi-block execute_region is not ordered by its textual walk, even
// though this particular branch reaches the load exactly once.
module {
  func.func @multiblock_execute_region(%root: memref<16xf32>, %value: f32) {
    %c0 = arith.constant 0 : index
    memref.store %value, %root[%c0] : memref<16xf32>
    scf.execute_region {
      cf.br ^read
    ^read:
      %inside = memref.load %root[%c0] : memref<16xf32>
      scf.yield
    }
    return
  }
}

// PLAN: buffer-accesses: count=2 scope=post-planning
// PLAN: memory-dependencies: count=0 scope=lexical status=incomplete
// PLAN-NOT: memory-dependency[
// PLAN: resources:
