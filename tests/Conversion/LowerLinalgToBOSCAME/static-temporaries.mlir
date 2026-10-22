// RUN: buddy-opt %s --linalg-to-boscame | FileCheck %s
// RUN: buddy-opt %s --linalg-to-boscame --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM

// Static padded physical storage must become an identity stack allocation
// with the exact descriptor, including its nonzero element offset, restored.
// This also covers buffers larger than the generic promotion pass's 1 KiB
// default threshold and removes deallocation of the promoted temporary.
// CHECK-LABEL: func.func @temporary
// CHECK: memref.alloca() alignment = 64 : memref<16511xf32>
// CHECK: memref.reinterpret_cast {{.*}} offset: [64], sizes: [128, 64], strides: [129, 1]
// CHECK-NOT: memref.dealloc
// CHECK-NOT: memref.alloc(
// LLVM-NOT: @malloc
// LLVM-NOT: @aligned_alloc
// LLVM-NOT: @free
// LLVM: alloca
module {
  func.func @temporary(%out: memref<128x64xf32>) {
    %local = memref.alloc() alignment = 64 : memref<128x64xf32, strided<[129, 1], offset: 64>>
    memref.copy %local, %out : memref<128x64xf32, strided<[129, 1], offset: 64>> to memref<128x64xf32>
    memref.dealloc %local : memref<128x64xf32, strided<[129, 1], offset: 64>>
    return
  }
}
