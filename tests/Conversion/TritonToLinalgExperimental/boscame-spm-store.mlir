// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse --linalg-to-boscame='store-policy=spm-dma' --promote-buffers-to-stack | FileCheck %s
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse --linalg-to-boscame='store-policy=spm-dma' --promote-buffers-to-stack --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// CHECK-LABEL: func.func @spm_store
// CHECK-NOT: memref.alloc
// CHECK: vector.store {{.*}} : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
// CHECK: bosc_ame.dma.start {{.*}} : (memref<16xf32, strided<[1], offset: ?>, #bosc_ame.spm<1>>, memref<16xf32
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: memref.copy
// CHECK-NOT: bufferization.
// LLVM: store <16 x float>
// LLVM: call i64 @bosc_ame_dma_start
// LLVM: call void @bosc_ame_dma_wait

module {
  tt.func public @spm_store(%out: !tt.ptr<f32>) {
    %zero = arith.constant 0 : index
    %ones = arith.constant dense<1.0> : vector<16xf32>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>
    vector.store %ones, %spm[%zero] : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    %tensor = bufferization.to_tensor %spm restrict writable : memref<16xf32, #bosc_ame.spm<0>> to tensor<16xf32>
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %address = tt.addptr %ptr, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %address, %tensor : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
