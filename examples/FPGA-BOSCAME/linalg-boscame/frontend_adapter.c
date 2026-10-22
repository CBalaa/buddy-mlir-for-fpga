//===- frontend_adapter.c - Real frontend unranked C wrapper launches ----===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "frontend_abi.h"

#include "abi.h"
#include "layernorm_abi.h"

#include <stddef.h>

typedef struct {
  const void *allocated;
  const void *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} FrontendMemRef1;

typedef struct {
  int64_t rank;
  FrontendMemRef1 *descriptor;
} FrontendUnranked;

_Static_assert(sizeof(FrontendMemRef1) == 40 && sizeof(FrontendUnranked) == 16,
               "real frontend C wrappers use typed unranked descriptors");

extern void _mlir_ciface_triton_gemm(FrontendUnranked *, FrontendUnranked *,
                                     FrontendUnranked *, int32_t, int32_t,
                                     int32_t, int32_t, int32_t, int32_t);
extern void _mlir_ciface_triton_layernorm(FrontendUnranked *,
                                          FrontendUnranked *,
                                          FrontendUnranked *,
                                          FrontendUnranked *, int32_t, int32_t,
                                          int32_t, int32_t, int32_t, int32_t);

static int check_ranges(const uintptr_t *starts, const size_t *lengths,
                        size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!starts[i] || lengths[i] > UINTPTR_MAX - starts[i])
      return 1;
    if (starts[i] % 64 != 0)
      return 2;
  }
  for (size_t i = 0; i < count; ++i)
    for (size_t j = i + 1; j < count; ++j)
      if (starts[i] < starts[j] + lengths[j] &&
          starts[j] < starts[i] + lengths[i])
        return 3;
  return 0;
}

int boscame_frontend_gemm_launch(const int8_t *a, const int8_t *b, int32_t *c) {
  uintptr_t starts[3] = {(uintptr_t)a, (uintptr_t)b, (uintptr_t)c};
  size_t lengths[3] = {BOSCAME_GEMM_A_BYTES, BOSCAME_GEMM_B_BYTES,
                       BOSCAME_GEMM_C_BYTES};
  int status = check_ranges(starts, lengths, 3);
  if (status)
    return status;
  FrontendMemRef1 descriptors[3] = {
      {a, a, 0, BOSCAME_GEMM_M * BOSCAME_GEMM_K, 1},
      {b, b, 0, BOSCAME_GEMM_N * BOSCAME_GEMM_K, 1},
      {c, c, 0, BOSCAME_GEMM_M * BOSCAME_GEMM_N, 1}};
  FrontendUnranked roots[3] = {
      {1, &descriptors[0]}, {1, &descriptors[1]}, {1, &descriptors[2]}};
  for (int32_t pid = BOSCAME_GEMM_GRID_M - 1; pid >= 0; --pid)
    _mlir_ciface_triton_gemm(&roots[0], &roots[1], &roots[2],
                             BOSCAME_GEMM_GRID_M, BOSCAME_GEMM_GRID_N, 1, pid,
                             0, 0);
  return 0;
}

int boscame_frontend_layernorm_launch(const float *x, const float *weight,
                                      const float *bias, float *out) {
  uintptr_t starts[4] = {(uintptr_t)x, (uintptr_t)weight, (uintptr_t)bias,
                         (uintptr_t)out};
  size_t lengths[4] = {BOSCAME_LAYERNORM_ROWS * BOSCAME_LAYERNORM_ROW_BYTES,
                       BOSCAME_LAYERNORM_ROW_BYTES, BOSCAME_LAYERNORM_ROW_BYTES,
                       BOSCAME_LAYERNORM_ROWS * BOSCAME_LAYERNORM_ROW_BYTES};
  int status = check_ranges(starts, lengths, 4);
  if (status)
    return status;
  FrontendMemRef1 descriptors[4] = {
      {x, x, 0, BOSCAME_LAYERNORM_ROWS * BOSCAME_LAYERNORM_WIDTH, 1},
      {weight, weight, 0, BOSCAME_LAYERNORM_WIDTH, 1},
      {bias, bias, 0, BOSCAME_LAYERNORM_WIDTH, 1},
      {out, out, 0, BOSCAME_LAYERNORM_ROWS * BOSCAME_LAYERNORM_WIDTH, 1}};
  FrontendUnranked roots[4] = {{1, &descriptors[0]},
                               {1, &descriptors[1]},
                               {1, &descriptors[2]},
                               {1, &descriptors[3]}};
  for (int32_t row = BOSCAME_LAYERNORM_ROWS - 1; row >= 0; --row)
    _mlir_ciface_triton_layernorm(&roots[0], &roots[1], &roots[2], &roots[3],
                                  BOSCAME_LAYERNORM_ROWS, 1, 1, row, 0, 0);
  return 0;
}
