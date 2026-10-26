// One output tile with a transposed physical B layout and a live i32 seed.
// K is substituted by ordered-ame.py to cover full and padded reductions.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @gemm_program(
      %a: memref<64x1024xi8>,
      %b: memref<1024x128xi8, strided<[1, 1024]>>,
      %c: memref<64x128xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x1024xi8>,
                   memref<1024x128xi8, strided<[1, 1024]>>)
      outs(%c : memref<64x128xi32>)
    return
  }
}
