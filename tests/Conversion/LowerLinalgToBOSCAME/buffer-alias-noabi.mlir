// RUN: buddy-opt %s --linalg-to-boscame='plan-report=true' -o %t.noabi.ir 2>%t.noabi.plan
// RUN: buddy-opt %s --linalg-to-boscame='plan-report=true dma-abi=args_aligned64_nonalias' -o %t.abi.ir 2>%t.abi.plan
// RUN: FileCheck %s --check-prefix=NO-ABI --implicit-check-not=proof=abi-nonalias-entry-arguments < %t.noabi.plan
// RUN: FileCheck %s --check-prefix=ABI < %t.abi.plan

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

// NO-ABI: buffer-alias-pairs: count=1 scope=known-roots-within-function unproven-pairs=unknown status=complete
// NO-ABI: buffer-alias-pair[{{[0-9]+}}]: function=@distinct_arguments accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=distinct-global-symbols
// ABI: buffer-alias-pairs: count=2 scope=known-roots-within-function unproven-pairs=unknown status=complete
// ABI-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@distinct_arguments accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=abi-nonalias-entry-arguments
// ABI-DAG: buffer-alias-pair[{{[0-9]+}}]: function=@distinct_arguments accesses=[{{[0-9]+}},{{[0-9]+}}] relation=disjoint proof=distinct-global-symbols
