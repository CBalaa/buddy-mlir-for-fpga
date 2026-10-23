// RUN: split-file %s %t
// RUN: buddy-opt %t/bn15.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga bn=15' 2>&1 | FileCheck %s --check-prefix=BN
// RUN: not buddy-opt %t/dynamic.mlir --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga bn=16' 2>&1 | FileCheck %s --check-prefix=DYNAMIC

// The automatic planner must diagnose an AME C tile plan whose N
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

// BN: bosc_ame.dma.start
// BN: bosc_ame.tile.load_acc {{.*}}#bosc_ame.spm<
// BN-NOT: bosc_ame.direct_ddr

// A dynamic descriptor offset is valid only when its divisibility can be
// proved. An arbitrary i32 offset can put a later C tile at a non-cache-line
// address, so both direct access and DMA staging reject this endpoint.
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

// DYNAMIC: BOSCAME DDR endpoint effective address cannot be proven 64 B aligned (descriptor offset in elements)
