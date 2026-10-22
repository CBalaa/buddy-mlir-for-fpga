// Canonical buffer-form input for M=128, N=128, K=1024.
// Launch two programs, (pid_m, pid_n) = (0, 0), (1, 0).
// Each call receives A + pid_m*64*1024 and C + pid_m*64*128 elements;
// B is shared by both calls. All descriptor offsets below are zero.
//
// A is row-major. B is logically [K,N], physically [N,K]:
// B[k,n] addresses B_physical[n*1024+k], with no implicit transpose/pack.
// The caller provides 64-byte aligned, mutually disjoint A/B/C storage.
// C is an input/output accumulator, not a zero-initialized result buffer.
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @gemm_program(
      %a: memref<64x1024xi8>,
      %b: memref<1024x128xi8, strided<[1, 1024]>>,
      %c: memref<64x128xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>,
                   bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x1024xi8>,
                   memref<1024x128xi8, strided<[1, 1024]>>)
      outs(%c : memref<64x128xi32>)
    return
  }
}
