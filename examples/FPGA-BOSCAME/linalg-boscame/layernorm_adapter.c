//===- layernorm_adapter.c - Rank-one row descriptors to LLVM ABI --------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "layernorm_abi.h"

#include <stddef.h>

// Four rank-one memrefs, each flattened as
// allocated/aligned/offset/size/stride.
extern void layernorm_program(
    const float *x_allocated, const float *x_aligned, int64_t x_offset,
    int64_t x_size, int64_t x_stride, const float *w_allocated,
    const float *w_aligned, int64_t w_offset, int64_t w_size, int64_t w_stride,
    const float *b_allocated, const float *b_aligned, int64_t b_offset,
    int64_t b_size, int64_t b_stride, float *o_allocated, float *o_aligned,
    int64_t o_offset, int64_t o_size, int64_t o_stride);

BoscameLayernormStatus boscame_layernorm_run_program(const float *x,
                                                     const float *weight,
                                                     const float *bias,
                                                     float *out) {
  uintptr_t starts[4] = {(uintptr_t)x, (uintptr_t)weight, (uintptr_t)bias,
                         (uintptr_t)out};
  uintptr_t ends[4];
  for (size_t i = 0; i < 4; ++i) {
    if (starts[i] == 0 || BOSCAME_LAYERNORM_ROW_BYTES > UINTPTR_MAX - starts[i])
      return BOSCAME_LAYERNORM_INVALID_POINTER;
    if (starts[i] % BOSCAME_LAYERNORM_ALIGNMENT != 0)
      return BOSCAME_LAYERNORM_UNALIGNED;
    ends[i] = starts[i] + BOSCAME_LAYERNORM_ROW_BYTES;
  }
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = i + 1; j < 4; ++j)
      if (starts[i] < ends[j] && starts[j] < ends[i])
        return BOSCAME_LAYERNORM_OVERLAPPING;
  layernorm_program(x, x, 0, BOSCAME_LAYERNORM_WIDTH, 1, weight, weight, 0,
                    BOSCAME_LAYERNORM_WIDTH, 1, bias, bias, 0,
                    BOSCAME_LAYERNORM_WIDTH, 1, out, out, 0,
                    BOSCAME_LAYERNORM_WIDTH, 1);
  return BOSCAME_LAYERNORM_OK;
}
