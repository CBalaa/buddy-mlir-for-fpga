// RUN: not buddy-opt %s --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias ' 2>&1 | FileCheck %s

// A preexisting explicit SPM endpoint must retain the original execution
// scope check when its identity copy becomes a planned boundary transfer.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @parallel_boundary(%input: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %resident = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<16xf32, #bosc_ame.spm<0>>
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : memref<16xf32>)
        outs(%resident : memref<16xf32, #bosc_ame.spm<0>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      scf.reduce
    }
    return
  }
}

// CHECK: BOSCAME fixed SPM workspace requires serial execution inside func.func; unsupported ancestor scf.parallel
