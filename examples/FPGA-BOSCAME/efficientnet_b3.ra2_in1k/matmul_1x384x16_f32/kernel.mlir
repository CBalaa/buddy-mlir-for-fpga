// FP32 linear operator; Buddy transpose-B RVV lowering, no quantization.
// Physical [N,K] B permits contiguous K-vector loads without a copy.
module {
  func.func @kernel_matmul_1x384x16_f32(%a: memref<1x16xf32>,
      %b: memref<384x16xf32>,
      %c: memref<1x384xf32>) attributes {llvm.emit_c_interface} {
    linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>] ins(%a, %b : memref<1x16xf32>, memref<384x16xf32>)
      outs(%c : memref<1x384xf32>)
    return
  }
}
