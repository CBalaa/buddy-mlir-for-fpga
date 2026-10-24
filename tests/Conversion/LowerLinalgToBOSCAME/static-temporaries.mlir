// RUN: not buddy-opt %s --linalg-to-boscame -o /dev/null 2>&1 | FileCheck %s

// The original temporary is read without initialization. Content validation
// rejects that read before placement, so no stack-promotion result is assumed.
module {
  func.func @temporary(%out: memref<128x64xf32>) {
    %local = memref.alloc() alignment = 64 : memref<128x64xf32, strided<[129, 1], offset: 64>>
    memref.copy %local, %out : memref<128x64xf32, strided<[129, 1], offset: 64>> to memref<128x64xf32>
    memref.dealloc %local : memref<128x64xf32, strided<[129, 1], offset: 64>>
    return
  }
}

// CHECK: kernel contract V22/V23: read or partial inheritance has no initialized content version
