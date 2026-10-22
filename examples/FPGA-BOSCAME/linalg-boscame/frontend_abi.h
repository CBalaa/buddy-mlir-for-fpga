//===- frontend_abi.h - Full-root real Triton frontend launch ABI ---------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDY_EXAMPLES_LINALG_BOSCAME_FRONTEND_ABI_H
#define BUDDY_EXAMPLES_LINALG_BOSCAME_FRONTEND_ABI_H

#include <stdint.h>

// Return 0 on success, 1 for null/overflowed pointers, 2 for misalignment,
// and 3 for overlapping full root ranges. Calls run serially in reverse pid
// order, with persistent fixed-address SPM. No buffers are allocated here.
//
// GEMM roots: A[128,1024] i8, B physical [128,1024] i8, C[128,128] i32.
int boscame_frontend_gemm_launch(const int8_t *a, const int8_t *b, int32_t *c);

// LayerNorm roots: X/Out[197,768] f32, shared Weight/Bias[768] f32.
int boscame_frontend_layernorm_launch(const float *x, const float *weight,
                                      const float *bias, float *out);

#endif // BUDDY_EXAMPLES_LINALG_BOSCAME_FRONTEND_ABI_H
