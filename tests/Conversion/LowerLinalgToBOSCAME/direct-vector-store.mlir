// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga store-policy=direct-ddr' | FileCheck %s
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga store-policy=direct-ddr' --lower-bosc-ame | FileCheck %s --check-prefix=LLVM

// A frontend-preserved store event stages the producer into SPM, then writes
// the result directly to its external DDR destination with vector stores.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @copy(%src: memref<64xf32>, %dst: memref<64xf32>) {
    memref.copy %src, %dst {
      bosc_ame.memory = {kind = "store", destination = "external_ddr",
                         logical_shape = array<i64: 64>,
                         valid_shape = array<i64: 64>, masked = false}
    } : memref<64xf32> to memref<64xf32>
    return
  }
}

// CHECK: bosc_ame.spm.view
// CHECK-COUNT-1: bosc_ame.dma.start
// CHECK: vector.load {{.*}}#bosc_ame.spm
// CHECK: vector.store {{.*}}%arg1
// CHECK-NOT: bosc_ame.dma.start
// LLVM-COUNT-1: llvm.call @bosc_ame_dma_start
// LLVM: llvm.store {{.*}}vector<16xf32>
// LLVM-NOT: llvm.call @bosc_ame_dma_start
