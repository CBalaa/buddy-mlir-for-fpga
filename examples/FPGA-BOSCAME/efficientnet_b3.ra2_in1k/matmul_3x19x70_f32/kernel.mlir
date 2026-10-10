// FP32 linear operator; Buddy transpose-B RVV lowering, no quantization.
// Physical [N,K] B permits contiguous K-vector loads without a copy.
module {
  func.func @kernel_matmul_3x19x70_f32(%a: memref<3x70xf32>,
      %b: memref<19x70xf32>,
      %c: memref<3x19xf32>) attributes {llvm.emit_c_interface} {
    linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>] ins(%a, %b : memref<3x70xf32>, memref<19x70xf32>)
      outs(%c : memref<3x19xf32>)
    return
  }
}
