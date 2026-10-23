// RUN: buddy-opt %s --linalg-to-boscame='buffer-placement=true enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias store-policy=spm-dma plan-report=true' -o %t 2>%t.plan
// RUN: FileCheck %s < %t.plan

// The unrelated root needs an entire bank. AME's selected staged A/B/C need
// 64 KiB + 64 KiB + 16 KiB, which fits the original two-bank workspace but
// cannot fit beside the resident root. AME roots remain outside the new
// placement model, while their staging still participates in its budget.
module {
  func.func @ame_staging_is_in_the_budget(
      %a: memref<64x1024xi8>,
      %b: memref<1024x64xi8, strided<[1, 1024]>>,
      %c: memref<64x64xi32>) {
    %zero = arith.constant 0.0 : f32
    %unrelated = memref.alloc() alignment = 64 : memref<32768xf32>
    linalg.fill ins(%zero : f32) outs(%unrelated : memref<32768xf32>)
    linalg.matmul {bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x1024xi8>,
        memref<1024x64xi8, strided<[1, 1024]>>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

// CHECK: buffer-placement: resident=0 ddr=1 applied=false
// CHECK: root[0]: kind=StagedFallback bytes=131072{{.*}}reason=missing-staging-capacity applied=false
// CHECK-SAME: detail=resident + explicit windows leave insufficient per-op staging capacity for linalg.matmul
