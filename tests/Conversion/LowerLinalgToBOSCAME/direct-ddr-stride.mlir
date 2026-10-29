// RUN: buddy-opt %s --linalg-to-boscame='plan-version=1 enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' --lower-bosc-ame -o %t
// RUN: FileCheck %s < %t
// RUN: buddy-translate --buddy-to-llvmir %t | FileCheck %s --check-prefix=IR
// Explicit V1 storage-policy compatibility; V2 defaults are tested by the schedule/stream tests.

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

// CHECK: %[[LOAD_STRIDE:[0-9]+]] = llvm.mlir.constant(4096 : i64)
// CHECK: llvm.riscv.bosc.ordered.load.acc{{.*}}%[[LOAD_STRIDE]]
// CHECK: %[[STORE_STRIDE:[0-9]+]] = llvm.mlir.constant(4096 : i64)
// CHECK: llvm.riscv.bosc.ordered.store.acc{{.*}}%[[STORE_STRIDE]]

// Check the actual stride operand as well as the ordered accumulator chain.
// IR: %[[ACC:[[:alnum:]_]+]] = call <vscale x 32 x i32> @llvm.riscv.bosc.ordered.load.acc.nxv32i32(ptr {{.*}}, i64 4096, i64 64, i64 64, i64 65602)
// IR: %[[A:[[:alnum:]_]+]] = call <vscale x 128 x i8> @llvm.riscv.bosc.ordered.load.a.nxv128i8(ptr {{.*}}, i64 64, i64 64, i64 64, i64 65552)
// IR: %[[B:[[:alnum:]_]+]] = call <vscale x 128 x i8> @llvm.riscv.bosc.ordered.load.b.nxv128i8(ptr {{.*}}, i64 64, i64 64, i64 64, i64 65552)
// IR: %[[C:[[:alnum:]_]+]] = call <vscale x 32 x i32> @llvm.riscv.bosc.ordered.mma.nxv32i32.nxv128i8(<vscale x 32 x i32> %[[ACC]], <vscale x 128 x i8> %[[A]], <vscale x 128 x i8> %[[B]], i64 64, i64 64, i64 64, i64 65552)
// IR: call void @llvm.riscv.bosc.ordered.store.acc.nxv32i32(<vscale x 32 x i32> %[[C]], ptr {{.*}}, i64 4096, i64 64, i64 64, i64 65602)
