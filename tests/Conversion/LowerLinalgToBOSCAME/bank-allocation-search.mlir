// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.reported 2>%t.plan
// RUN: cmp %t.default %t.reported
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan

module {
  func.func @two_large_transfers(%src: memref<24576xf32>,
                                 %dst: memref<24576xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<24576xf32>) outs(%dst : memref<24576xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// PLAN: transfer[0]: consumer=linalg.generic residency=operation-local consumers=[linalg.generic] role=input-a shape=[24576] strides=[1] validShape=[24576] bank=0 offset=0 capacity=98304 effective-byte-offset-bytes=0 endpoint-alignment-proven=true packets=1 read=true write=false
// PLAN: transfer[1]: consumer=linalg.generic residency=operation-local consumers=[linalg.generic] role=output-c shape=[24576] strides=[1] validShape=[24576] bank=1 offset=0 capacity=98304 effective-byte-offset-bytes=0 endpoint-alignment-proven=true packets=1 read=false write=true
// PLAN: bank-allocation-search: scope=operation-local groups=1 exact-assignments=4 exact-rejected-assignments=2 bounded-groups=0 repair-groups=0 repair-search-nodes=0 repair-rejected-placements=0 objective=minimize-bank-high-water status=exact latency=uncalibrated
// PLAN: bank[0]: high-water=98304/131072 bytes
// PLAN: bank[1]: high-water=98304/131072 bytes
