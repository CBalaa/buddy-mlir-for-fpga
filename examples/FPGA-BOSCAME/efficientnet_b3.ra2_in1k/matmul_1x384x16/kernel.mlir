// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_1x384x16(%a: memref<1x16xi8>,
      %b: memref<16x384xi8, strided<[1, 16]>>,
      %c: memref<1x384xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<1x16xi8>, memref<16x384xi8, strided<[1, 16]>>)
      outs(%c : memref<1x384xi32>)
    return
  }
}
