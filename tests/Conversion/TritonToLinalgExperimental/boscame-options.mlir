// RUN: buddy-opt %s --triton-to-linalg-experimental -o %t.default
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=false' -o %t.false
// RUN: diff %t.default %t.false
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true' 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=unknown' 2>&1 | FileCheck %s --check-prefix=INVALID
// MISSING: boscame=true requires dma-abi=args_aligned64_nonalias
// INVALID: invalid dma-abi: unknown

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
