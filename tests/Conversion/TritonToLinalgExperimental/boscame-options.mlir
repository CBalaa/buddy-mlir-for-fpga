// RUN: buddy-opt %s --triton-to-linalg-experimental -o %t.default
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=false' -o %t.false
// RUN: diff %t.default %t.false
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true' 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=unknown' 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse --linalg-to-boscame='store-policy=spm-dma' --promote-buffers-to-stack | FileCheck %s --check-prefix=UNMASKED --implicit-check-not=vector.load --implicit-check-not=vector.store
// MISSING: boscame=true requires dma-abi=args_aligned64_nonalias
// INVALID: invalid dma-abi: unknown
// UNMASKED-LABEL: func.func @copy
// UNMASKED: %[[LOAD_IN:.*]] = bosc_ame.dma.start
// UNMASKED-NEXT: bosc_ame.dma.wait %[[LOAD_IN]]
// UNMASKED: %[[LOAD_COPY:.*]] = bosc_ame.dma.start {{.*}} : (memref<16xf32, strided<[1], offset: ?>, #bosc_ame.spm<0>>, memref<16xf32, strided<[1], offset: ?>, #bosc_ame.spm<1>>, index)
// UNMASKED-NEXT: bosc_ame.dma.wait %[[LOAD_COPY]]
// UNMASKED: %[[LOAD_OUT:.*]] = bosc_ame.dma.start
// UNMASKED-NEXT: bosc_ame.dma.wait %[[LOAD_OUT]]
// UNMASKED: %[[STORE_IN:.*]] = bosc_ame.dma.start
// UNMASKED-NEXT: bosc_ame.dma.wait %[[STORE_IN]]
// UNMASKED: %[[STORE_COPY:.*]] = bosc_ame.dma.start {{.*}} : (memref<16xf32, strided<[1], offset: ?>, #bosc_ame.spm<0>>, memref<16xf32, strided<[1], offset: ?>, #bosc_ame.spm<1>>, index)
// UNMASKED-NEXT: bosc_ame.dma.wait %[[STORE_COPY]]
// UNMASKED: %[[STORE_OUT:.*]] = bosc_ame.dma.start
// UNMASKED-NEXT: bosc_ame.dma.wait %[[STORE_OUT]]
// UNMASKED-NOT: bosc_ame.dma.start
// UNMASKED-NOT: memref.alloc(
// UNMASKED-NOT: memref.copy
// UNMASKED-NOT: bufferization.

module {
  tt.func public @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %sp = tt.addptr %s, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %dp = tt.addptr %d, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %sp : tensor<16x!tt.ptr<f32>>
    tt.store %dp, %v : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
