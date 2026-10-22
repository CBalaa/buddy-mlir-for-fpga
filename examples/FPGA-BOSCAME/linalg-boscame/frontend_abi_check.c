//===- frontend_abi_check.c - Host actual frontend launch ABI check ------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "frontend_abi.h"

#include <stddef.h>

typedef struct {
  const void *allocated;
  const void *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} MemRef1;

typedef struct {
  int64_t rank;
  MemRef1 *descriptor;
} Unranked;

_Alignas(64) static int8_t a[128 * 1024], b[128 * 1024];
_Alignas(64) static int32_t c[128 * 128];
_Alignas(64) static float x[197 * 768], weight[768], bias[768], out[197 * 768];
static unsigned gemm_calls, layernorm_calls, failures;

static void check_root(const Unranked *root, const void *pointer,
                       int64_t size) {
  if (root->rank != 1 || root->descriptor->allocated != pointer ||
      root->descriptor->aligned != pointer || root->descriptor->offset != 0 ||
      root->descriptor->size != size || root->descriptor->stride != 1)
    ++failures;
}

void _mlir_ciface_triton_gemm(Unranked *a_root, Unranked *b_root,
                              Unranked *c_root, int32_t grid_x, int32_t grid_y,
                              int32_t grid_z, int32_t pid_x, int32_t pid_y,
                              int32_t pid_z) {
  check_root(a_root, a, 128 * 1024);
  check_root(b_root, b, 128 * 1024);
  check_root(c_root, c, 128 * 128);
  if (grid_x != 2 || grid_y != 1 || grid_z != 1 ||
      pid_x != 1 - (int32_t)gemm_calls || pid_y != 0 || pid_z != 0)
    ++failures;
  ++gemm_calls;
}

void _mlir_ciface_triton_layernorm(Unranked *x_root, Unranked *w_root,
                                   Unranked *b_root, Unranked *o_root,
                                   int32_t grid_x, int32_t grid_y,
                                   int32_t grid_z, int32_t pid_x, int32_t pid_y,
                                   int32_t pid_z) {
  check_root(x_root, x, 197 * 768);
  check_root(w_root, weight, 768);
  check_root(b_root, bias, 768);
  check_root(o_root, out, 197 * 768);
  if (grid_x != 197 || grid_y != 1 || grid_z != 1 ||
      pid_x != 196 - (int32_t)layernorm_calls || pid_y != 0 || pid_z != 0)
    ++failures;
  ++layernorm_calls;
}

int main(void) {
  if (boscame_frontend_gemm_launch(a, b, c) != 0 || gemm_calls != 2)
    ++failures;
  if (boscame_frontend_gemm_launch(NULL, b, c) != 1 ||
      boscame_frontend_gemm_launch(a + 1, b, c) != 2 ||
      boscame_frontend_gemm_launch(a, a, c) != 3 || gemm_calls != 2)
    ++failures;
  if (boscame_frontend_layernorm_launch(x, weight, bias, out) != 0 ||
      layernorm_calls != 197)
    ++failures;
  if (boscame_frontend_layernorm_launch(NULL, weight, bias, out) != 1 ||
      boscame_frontend_layernorm_launch(x + 1, weight, bias, out) != 2 ||
      boscame_frontend_layernorm_launch(x, x, bias, out) != 3 ||
      layernorm_calls != 197)
    ++failures;
  return failures != 0;
}
