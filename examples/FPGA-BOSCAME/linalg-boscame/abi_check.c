//===- abi_check.c - Host invocation check with a flattened kernel stub --===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "abi.h"

_Alignas(BOSCAME_GEMM_ALIGNMENT) static int8_t a[BOSCAME_GEMM_A_BYTES];
_Alignas(BOSCAME_GEMM_ALIGNMENT) static int8_t b[BOSCAME_GEMM_B_BYTES];
_Alignas(
    BOSCAME_GEMM_ALIGNMENT) static int32_t c[BOSCAME_GEMM_M * BOSCAME_GEMM_N];
static size_t calls;
static int failures;

// Substitute only the generated function. The real adapter and launcher are
// linked unchanged, so this checks their flattened call and pointer slicing.
void gemm_program(const int8_t *a_allocated, const int8_t *a_aligned,
                  int64_t a_offset, int64_t a_size0, int64_t a_size1,
                  int64_t a_stride0, int64_t a_stride1,
                  const int8_t *b_allocated, const int8_t *b_aligned,
                  int64_t b_offset, int64_t b_size0, int64_t b_size1,
                  int64_t b_stride0, int64_t b_stride1, int32_t *c_allocated,
                  int32_t *c_aligned, int64_t c_offset, int64_t c_size0,
                  int64_t c_size1, int64_t c_stride0, int64_t c_stride1) {
  if (calls >= 2) {
    ++failures;
    return;
  }
  const int8_t *want_a = a + calls * 64 * 1024;
  int32_t *want_c = c + calls * 64 * 128;
  if (a_allocated != want_a || a_aligned != want_a || a_offset != 0 ||
      a_size0 != 64 || a_size1 != 1024 || a_stride0 != 1024 || a_stride1 != 1 ||
      b_allocated != b || b_aligned != b || b_offset != 0 || b_size0 != 1024 ||
      b_size1 != 128 || b_stride0 != 1 || b_stride1 != 1024 ||
      c_allocated != want_c || c_aligned != want_c || c_offset != 0 ||
      c_size0 != 64 || c_size1 != 128 || c_stride0 != 128 || c_stride1 != 1)
    ++failures;
  ++calls;
}

int main(void) {
  if (boscame_gemm_launch(a, b, c) != BOSCAME_GEMM_OK || calls != 2)
    ++failures;
  if (boscame_gemm_launch(NULL, b, c) != BOSCAME_GEMM_INVALID_POINTER)
    ++failures;
  if (boscame_gemm_launch(a + 1, b, c) != BOSCAME_GEMM_UNALIGNED)
    ++failures;
  if (boscame_gemm_launch(a, a, c) != BOSCAME_GEMM_OVERLAPPING)
    ++failures;
  if (calls != 2)
    ++failures;
  return failures != 0;
}
