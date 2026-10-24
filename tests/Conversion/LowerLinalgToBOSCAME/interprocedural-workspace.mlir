// RUN: not buddy-opt %s '--linalg-to-boscame=enable-spm-dma=true dma-abi=args_aligned64_nonalias' -o /dev/null 2>&1 | FileCheck %s

// The called helper's formals have no entry disjointness guarantee, so V1
// rejects the potentially overlapping copy before routing. Live caller/callee
// SPM clobbering is separately rejected by PlanWorkspaceCalls.py, including
// calls with no memref operands.
module {
  func.func @helper(%input: memref<16xf32>, %out: memref<16xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                     affine_map<(i) -> (i)>],
                    iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %unused: f32):
      linalg.yield %x : f32
    }
    return
  }

  func.func @caller(%input: memref<16xf32>, %temporary: memref<16xf32>,
                    %out: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %two = arith.constant dense<2.0> : vector<16xf32>
    %saved = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<16xf32, #bosc_ame.spm<0>>
    vector.store %two, %saved[%c0]
      : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    func.call @helper(%input, %temporary)
      : (memref<16xf32>, memref<16xf32>) -> ()
    %value = vector.load %saved[%c0]
      : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    vector.store %value, %out[%c0] : memref<16xf32>, vector<16xf32>
    return
  }
}

// CHECK: kernel contract V25: overlapping views must have identical element addresses; use a separate buffer
