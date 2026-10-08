// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_1x16x32(%a: memref<1x32xi8>,
      %b: memref<32x16xi8, strided<[1, 32]>>,
      %c: memref<1x16xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<1x32xi8>, memref<32x16xi8, strided<[1, 32]>>)
      outs(%c : memref<1x16xi32>)
    return
  }
}
