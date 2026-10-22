//===- layernorm_abi_check.c - Host flattened LayerNorm invocation check -===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "layernorm_abi.h"

#include <stddef.h>

_Alignas(64) static float x[768], weight[768], bias[768], out[768];
static unsigned calls;
static unsigned failures;

void layernorm_program(const float *x_allocated, const float *x_aligned,
                       int64_t x_offset, int64_t x_size, int64_t x_stride,
                       const float *w_allocated, const float *w_aligned,
                       int64_t w_offset, int64_t w_size, int64_t w_stride,
                       const float *b_allocated, const float *b_aligned,
                       int64_t b_offset, int64_t b_size, int64_t b_stride,
                       float *o_allocated, float *o_aligned, int64_t o_offset,
                       int64_t o_size, int64_t o_stride) {
  if (x_allocated != x || x_aligned != x || x_offset != 0 || x_size != 768 ||
      x_stride != 1 || w_allocated != weight || w_aligned != weight ||
      w_offset != 0 || w_size != 768 || w_stride != 1 || b_allocated != bias ||
      b_aligned != bias || b_offset != 0 || b_size != 768 || b_stride != 1 ||
      o_allocated != out || o_aligned != out || o_offset != 0 ||
      o_size != 768 || o_stride != 1)
    ++failures;
  ++calls;
}

int main(void) {
  if (boscame_layernorm_run_program(x, weight, bias, out) !=
          BOSCAME_LAYERNORM_OK ||
      calls != 1)
    ++failures;
  if (boscame_layernorm_run_program(NULL, weight, bias, out) !=
      BOSCAME_LAYERNORM_INVALID_POINTER)
    ++failures;
  if (boscame_layernorm_run_program(x + 1, weight, bias, out) !=
      BOSCAME_LAYERNORM_UNALIGNED)
    ++failures;
  if (boscame_layernorm_run_program(x, x, bias, out) !=
      BOSCAME_LAYERNORM_OVERLAPPING)
    ++failures;
  if (calls != 1)
    ++failures;
  return failures != 0;
}
