// RUN: not buddy-opt %s '--linalg-to-boscame=buffer-placement=false enable-spm-dma=true dma-abi=args_aligned64_nonalias' -o /dev/null 2>&1 | FileCheck %s
// RUN: not buddy-opt %s '--linalg-to-boscame=buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias' -o /dev/null 2>&1 | FileCheck %s

// The original affine access is outside the V1 identity-indexed pointwise
// recipe. The physical-placement debug override cannot change that semantic
// boundary or restore the removed ordinary fallback compiler.
module {
  func.func @unsupported_vector_layout() {
    %one = arith.constant 1.0 : f32
    %a = memref.alloc() alignment = 64 : memref<32xf32>
    %b = memref.alloc() alignment = 64 : memref<16xf32>
    linalg.fill ins(%one : f32) outs(%a : memref<32xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i * 2)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a : memref<32xf32>) outs(%b : memref<16xf32>) {
    ^bb0(%value: f32, %out: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// CHECK: kernel contract V25: pointwise update requires parallel identity indexing
