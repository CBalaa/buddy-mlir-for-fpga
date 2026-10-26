// RUN: not buddy-opt %s --kernel-contract-verify 2>&1 | FileCheck %s
// A frontend contract cannot contain an already placed backend SPM operand.
// Physical SPM inputs belong to the backend's buffer-form input adapter.
// CHECK: Dialect `tt' not found for custom op 'tt.func'

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
