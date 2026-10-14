// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' | FileCheck %s
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias structured-ldst-mode=tensor-first-vector-cpu' | FileCheck %s
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// CHECK: bosc_ame.dma.abi = "args_aligned64_nonalias"
// CHECK-LABEL: func.func @masked
// CHECK: arith.constant 3072 : index
// CHECK: bosc_ame.spm.view {{.*}} : memref<768xf32, #bosc_ame.spm<{{[01]}}>>
// CHECK: memref.alloca() {{.*}} : memref<1024xf32>
// CHECK: linalg.fill ins({{.*}} : f32) outs({{.*}} : memref<1024xf32>)
// CHECK: bosc_ame.dma.start {{.*}} : (memref<768xf32, strided<[1], offset: ?>>,
// CHECK: bosc_ame.dma.wait
// CHECK: vector.load {{.*}} : memref<768xf32, #bosc_ame.spm<{{[01]}}>>, vector<16xf32>
// CHECK: vector.store {{.*}} : memref<1024xf32>, vector<16xf32>
// CHECK: vector.load {{.*}} : memref<1024xf32>, vector<16xf32>
// CHECK: vector.store {{.*}} : memref<768xf32, #bosc_ame.spm<{{[01]}}>>, vector<16xf32>
// CHECK: bosc_ame.dma.start {{.*}} : (memref<768xf32, #bosc_ame.spm<{{[01]}}>>, memref<768xf32, strided<[1], offset: ?>>
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: bufferization.
// LLVM: call i64 @bosc_ame_dma_start
// LLVM: load <16 x float>
// LLVM: store <16 x float>
// LLVM: call void @bosc_ame_dma_wait

module {
  tt.func public @masked(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) attributes {test.custom = "preserved"} {
    %c768 = arith.constant 768 : i32
    %ones = arith.constant dense<1.0> : tensor<1024xf32>
    %limit = arith.constant dense<768> : tensor<1024xi32>
    %row = tt.get_program_id x : i32
    %offset = arith.muli %row, %c768 : i32
    %srow = tt.addptr %src, %offset : !tt.ptr<f32>, i32
    %drow = tt.addptr %dst, %offset : !tt.ptr<f32>, i32
    %r = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %mask = arith.cmpi slt, %r, %limit : tensor<1024xi32>
    %s = tt.splat %srow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %d = tt.splat %drow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %sp = tt.addptr %s, %r : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %dp = tt.addptr %d, %r : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %v = tt.load %sp, %mask, %ones : tensor<1024x!tt.ptr<f32>>
    tt.store %dp, %v, %mask : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
