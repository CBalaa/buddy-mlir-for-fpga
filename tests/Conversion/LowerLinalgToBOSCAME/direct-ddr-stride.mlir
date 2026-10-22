// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' --lower-bosc-ame | FileCheck %s

// The AME stride is measured in bytes.  A 1024-element i32 row pitch must be
// emitted as 4096, while C remains an external DDR accumulator operand.
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32, strided<[1024, 1]>>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32, strided<[1024, 1]>>)
    return
  }
}

// CHECK: "bosc_ame.intr.msettype"
// CHECK: %[[LOAD_STRIDE:[0-9]+]] = llvm.mlir.constant(4096 : i64)
// CHECK: "bosc_ame.intr.mlce32.m"{{.*}}%[[LOAD_STRIDE]]
// CHECK: %[[STORE_STRIDE:[0-9]+]] = llvm.mlir.constant(4096 : i64)
// CHECK: "bosc_ame.intr.msce32.m"{{.*}}%[[STORE_STRIDE]]
