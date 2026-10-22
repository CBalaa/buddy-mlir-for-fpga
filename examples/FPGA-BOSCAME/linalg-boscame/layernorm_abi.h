//===- layernorm_abi.h - Ranked standalone LayerNorm contract ------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDY_EXAMPLES_LINALG_BOSCAME_LAYERNORM_ABI_H
#define BUDDY_EXAMPLES_LINALG_BOSCAME_LAYERNORM_ABI_H

#include <stdint.h>

enum {
  BOSCAME_LAYERNORM_ROWS = 197,
  BOSCAME_LAYERNORM_WIDTH = 768,
  BOSCAME_LAYERNORM_BLOCK = 1024,
  BOSCAME_LAYERNORM_ROW_BYTES = BOSCAME_LAYERNORM_WIDTH * 4,
  BOSCAME_LAYERNORM_ALIGNMENT = 64
};

_Static_assert(
    sizeof(void *) == 8 && sizeof(int64_t) == 8 && sizeof(float) == 4,
    "the generated LayerNorm ABI requires 64-bit pointers/index and f32");
_Static_assert(
    BOSCAME_LAYERNORM_ROW_BYTES == 3072 &&
        BOSCAME_LAYERNORM_ROW_BYTES % BOSCAME_LAYERNORM_ALIGNMENT == 0 &&
        BOSCAME_LAYERNORM_BLOCK == 1024,
    "the valid DDR prefix and padded reduction shape must match MLIR");

typedef enum {
  BOSCAME_LAYERNORM_OK = 0,
  BOSCAME_LAYERNORM_INVALID_POINTER = 1,
  BOSCAME_LAYERNORM_UNALIGNED = 2,
  BOSCAME_LAYERNORM_OVERLAPPING = 3
} BoscameLayernormStatus;

// One caller-sliced row, with shared Weight/Bias. Each buffer exposes 768 f32.
// The caller owns allocation capacity, platform setup and DMA runtime setup.
BoscameLayernormStatus boscame_layernorm_run_program(const float *x,
                                                     const float *weight,
                                                     const float *bias,
                                                     float *out);

#endif // BUDDY_EXAMPLES_LINALG_BOSCAME_LAYERNORM_ABI_H
