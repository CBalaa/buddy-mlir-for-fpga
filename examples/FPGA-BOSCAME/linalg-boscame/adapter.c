//===- adapter.c - Ranked slab descriptors to flattened LLVM ABI --------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "abi.h"

// This is @gemm_program in gemm.ll: three ranked rank-2 memrefs, each flattened
// as allocated, aligned, offset, size0, size1, stride0, stride1. The kernel
// does not request a _mlir_ciface wrapper and takes no runtime grid/program
// scalars.
extern void gemm_program(const int8_t *a_allocated, const int8_t *a_aligned,
                         int64_t a_offset, int64_t a_size0, int64_t a_size1,
                         int64_t a_stride0, int64_t a_stride1,
                         const int8_t *b_allocated, const int8_t *b_aligned,
                         int64_t b_offset, int64_t b_size0, int64_t b_size1,
                         int64_t b_stride0, int64_t b_stride1,
                         int32_t *c_allocated, int32_t *c_aligned,
                         int64_t c_offset, int64_t c_size0, int64_t c_size1,
                         int64_t c_stride0, int64_t c_stride1);

static BoscameGemmStatus check_ranges(const int8_t *a, size_t a_bytes,
                                      const int8_t *b, size_t b_bytes,
                                      int32_t *c, size_t c_bytes) {
  uintptr_t starts[3] = {(uintptr_t)a, (uintptr_t)b, (uintptr_t)c};
  size_t lengths[3] = {a_bytes, b_bytes, c_bytes};
  uintptr_t ends[3];
  for (size_t i = 0; i < 3; ++i) {
    if (starts[i] == 0 || lengths[i] > UINTPTR_MAX - starts[i])
      return BOSCAME_GEMM_INVALID_POINTER;
    if (starts[i] % BOSCAME_GEMM_ALIGNMENT != 0)
      return BOSCAME_GEMM_UNALIGNED;
    ends[i] = starts[i] + lengths[i];
  }
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = i + 1; j < 3; ++j)
      if (starts[i] < ends[j] && starts[j] < ends[i])
        return BOSCAME_GEMM_OVERLAPPING;
  return BOSCAME_GEMM_OK;
}

BoscameGemmStatus boscame_gemm_validate_roots(const int8_t *a, const int8_t *b,
                                              int32_t *c) {
  return check_ranges(a, BOSCAME_GEMM_A_BYTES, b, BOSCAME_GEMM_B_BYTES, c,
                      BOSCAME_GEMM_C_BYTES);
}

BoscameGemmStatus boscame_gemm_run_program(const int8_t *a, const int8_t *b,
                                           int32_t *c) {
  BoscameGemmStatus status =
      check_ranges(a, BOSCAME_GEMM_PROGRAM_A_BYTES, b, BOSCAME_GEMM_B_BYTES, c,
                   BOSCAME_GEMM_PROGRAM_C_BYTES);
  if (status != BOSCAME_GEMM_OK)
    return status;

  const BoscameI8MemRef2 a_desc = {
      a, a, 0, {BOSCAME_GEMM_BM, BOSCAME_GEMM_K}, {BOSCAME_GEMM_K, 1}};
  const BoscameI8MemRef2 b_desc = {
      b, b, 0, {BOSCAME_GEMM_K, BOSCAME_GEMM_N}, {1, BOSCAME_GEMM_K}};
  const BoscameI32MemRef2 c_desc = {
      c, c, 0, {BOSCAME_GEMM_BM, BOSCAME_GEMM_N}, {BOSCAME_GEMM_N, 1}};
  gemm_program(a_desc.allocated, a_desc.aligned, a_desc.offset, a_desc.sizes[0],
               a_desc.sizes[1], a_desc.strides[0], a_desc.strides[1],
               b_desc.allocated, b_desc.aligned, b_desc.offset, b_desc.sizes[0],
               b_desc.sizes[1], b_desc.strides[0], b_desc.strides[1],
               c_desc.allocated, c_desc.aligned, c_desc.offset, c_desc.sizes[0],
               c_desc.sizes[1], c_desc.strides[0], c_desc.strides[1]);
  return BOSCAME_GEMM_OK;
}
