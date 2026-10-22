// RUN: split-file %s %t
// RUN: not buddy-opt %t/nonpositive.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-EXTENT
// RUN: not buddy-opt %t/wrong-rank.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-RANK
// RUN: not buddy-opt %t/wrong-type.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE
// RUN: not buddy-opt %t/nonpositive.mlir --lower-linalg-to-boscame -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-EXTENT

// CHECK-EXTENT: invalid gpu.known_grid_size contract; expected three positive i32 extents
// CHECK-RANK: gpu.known_grid_size must contain exactly 3 elements
// CHECK-TYPE: gpu.known_grid_size must be a dense i32 array

//--- nonpositive.mlir
module {
  func.func @bad_extent() attributes {gpu.known_grid_size = array<i32: 1, 0, 1>} {
    return
  }
}

//--- wrong-rank.mlir
module {
  func.func @bad_rank() attributes {gpu.known_grid_size = array<i32: 1, 2>} {
    return
  }
}

//--- wrong-type.mlir
module {
  func.func @bad_type() attributes {gpu.known_grid_size = array<i64: 1, 2, 3>} {
    return
  }
}
