// RUN: buddy-opt %s --linalg-to-boscame='buffer-placement=true enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t 2>%t.plan
// RUN: FileCheck %s < %t.plan

// The unrelated root occupies an entire bank. The external C has a proven
// direct route, so only A/B need staging and their two 64 KiB windows fit in
// the other bank. Both selected routes participate in the placement budget.
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

// CHECK: buffer-placement: resident=1 ddr=0 applied=true
// CHECK: root[0]: kind=SpmResident bytes=131072 bank=0 offset=0{{.*}}reason=group-legality-and-capacity applied=true
// CHECK: transfer[0]:{{.*}}role=input-a{{.*}}bank=1 offset=0 capacity=65536
// CHECK: transfer[1]:{{.*}}role=input-b{{.*}}bank=1 offset=65536 capacity=65536
// CHECK: direct-C=direct-ddr (proven)
