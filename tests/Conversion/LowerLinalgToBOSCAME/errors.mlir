// RUN: not buddy-opt %s --lower-linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s

// A one-byte store cannot become a legal typed DMA packet by rounding its
// length up: doing so would write addresses outside the original operation.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @scalar(%a: memref<1xi8>, %b: memref<1xi8>, %c: memref<1xi8>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%a : memref<1xi8>) outs(%c : memref<1xi8>) {
    ^bb0(%x: i8, %y: i8):
      linalg.yield %x : i8
    }
    return
  }
}

// CHECK: every BOSCAME DMA packet must be a positive 64-byte range
