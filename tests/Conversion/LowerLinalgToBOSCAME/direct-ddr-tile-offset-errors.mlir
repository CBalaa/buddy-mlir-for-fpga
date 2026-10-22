// RUN: split-file %s %t
// RUN: not buddy-opt %t/bn15.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga bn=15' 2>&1 | FileCheck %s --check-prefix=BN
// RUN: not buddy-opt %t/dynamic.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga bn=16' 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: buddy-opt %t/bn15.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true store-policy=spm-dma dma-abi=args_aligned64_nonalias target=nr-fpga bn=15' | FileCheck %s --check-prefix=COMPAT

// The default direct-DDR policy must diagnose an AME C tile plan whose N
// step is not a 64-byte multiple. Otherwise the planner would silently stage
// C through SPM/DMA, or emit a later tile at an unaligned column offset.
//--- bn15.mlir
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// BN: direct-ddr AME C output cannot be tiled with 64-byte-aligned accumulator rows
// COMPAT: bosc_ame.dma.start

// A dynamic descriptor offset is valid only when its divisibility can be
// proved. An arbitrary i32 offset can put a later C tile at a non-cache-line
// address, so direct mode must reject it before emitting tile operations.
//--- dynamic.mlir
module {
  func.func @gemm(%a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %raw: memref<8192xi32>, %offset: index) {
    %c = memref.reinterpret_cast %raw to offset: [%offset], sizes: [64, 64],
      strides: [64, 1] : memref<8192xi32> to
      memref<64x64xi32, strided<[64, 1], offset: ?>>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32, strided<[64, 1], offset: ?>>)
    return
  }
}

// DYNAMIC: direct-ddr AME C output address must be provably 64-byte aligned
