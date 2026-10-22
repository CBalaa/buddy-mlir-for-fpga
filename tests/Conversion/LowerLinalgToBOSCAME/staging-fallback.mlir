// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' | FileCheck %s

// Unsupported SPM vector element types stay in ordinary memory even when the
// typed DMA planner is enabled.
module {
  func.func @f16(%input: memref<32xf16>, %out: memref<32xf16>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<32xf16>) outs(%out : memref<32xf16>) {
    ^bb0(%x: f16, %unused: f16):
      linalg.yield %x : f16
    }
    return
  }

}

// CHECK-LABEL: func.func @f16
// CHECK-NOT: bosc_ame.dma.start
// CHECK: scf.for
