// RUN: buddy-opt %s --linalg-to-boscame='buffer-placement=false enable-spm-dma=true dma-abi=args_aligned64_nonalias' -o /dev/null
// RUN: buddy-opt %s --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t 2>%t.plan
// RUN: FileCheck %s < %t.plan

// The ordinary path permits this strided affine input access. The SPM vector
// island requires identity maps and matching shapes, so both connected roots
// must retain their ordinary placement before any resident storage is changed.
// CHECK: buffer-placement: resident=0 ddr=2 applied=false
// CHECK-COUNT-2: kind=StagedFallback bytes={{128|64}}{{.*}}group=0 reason=computation-legality applied=false
// CHECK-SAME: SPM generic requires matching shapes, element byte widths and identity indexing maps

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
