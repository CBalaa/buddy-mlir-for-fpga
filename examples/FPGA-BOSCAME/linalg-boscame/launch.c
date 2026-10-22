//===- launch.c - Canonical serial 2x1 standalone GEMM launch ------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "abi.h"

BoscameGemmStatus boscame_gemm_launch(const int8_t *a, const int8_t *b,
                                      int32_t *c) {
  BoscameGemmStatus status = boscame_gemm_validate_roots(a, b, c);
  if (status != BOSCAME_GEMM_OK)
    return status;

  // N is explicitly fused into each program. The shared fixed SPM workspace
  // permits serial calls; concurrent calls require external serialization.
  for (size_t pid_m = 0; pid_m < BOSCAME_GEMM_GRID_M; ++pid_m) {
    size_t m0 = pid_m * BOSCAME_GEMM_BM;
    status = boscame_gemm_run_program(a + m0 * BOSCAME_GEMM_K, b,
                                      c + m0 * BOSCAME_GEMM_N);
    if (status != BOSCAME_GEMM_OK)
      return status;
  }
  return BOSCAME_GEMM_OK;
}
