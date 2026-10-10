// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {
  func.func @kernel_matmul_1x1000x1536(%a: memref<1x1536xi8>,
      %b: memref<1536x1000xi8, strided<[1, 1536]>>,
      %c: memref<1x1000xi32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%a, %b : memref<1x1536xi8>, memref<1536x1000xi8, strided<[1, 1536]>>)
      outs(%c : memref<1x1000xi32>)
    return
  }
}
