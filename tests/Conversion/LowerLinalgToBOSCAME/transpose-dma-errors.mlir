// RUN: not buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @unaligned_transposed_b(%a: memref<64x64xi8>,
                                    %c: memref<64x64xi32>) {
    %b = memref.alloc() : memref<64x64xi8>
    %bt = memref.transpose %b (i, j) -> (j, i)
      : memref<64x64xi8> to
        memref<64x64xi8, affine_map<(d0, d1) -> (d0 + d1 * 64)>>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>,
                   bosc_ame.lower_to_ame}
      ins(%a, %bt : memref<64x64xi8>,
                    memref<64x64xi8, affine_map<(d0, d1) -> (d0 + d1 * 64)>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// CHECK: endpoint allocation must prove 64 B alignment
