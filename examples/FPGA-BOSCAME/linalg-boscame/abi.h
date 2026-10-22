//===- abi.h - Canonical standalone BOSCAME GEMM contract ----------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDY_EXAMPLES_LINALG_BOSCAME_ABI_H
#define BUDDY_EXAMPLES_LINALG_BOSCAME_ABI_H

#include <stddef.h>
#include <stdint.h>

enum {
  BOSCAME_GEMM_M = 128,
  BOSCAME_GEMM_N = 128,
  BOSCAME_GEMM_K = 1024,
  BOSCAME_GEMM_BM = 64,
  BOSCAME_GEMM_BN = 128,
  BOSCAME_GEMM_BK = 1024,
  BOSCAME_GEMM_GRID_M = BOSCAME_GEMM_M / BOSCAME_GEMM_BM,
  BOSCAME_GEMM_GRID_N = BOSCAME_GEMM_N / BOSCAME_GEMM_BN,
  BOSCAME_GEMM_ALIGNMENT = 64,
  BOSCAME_GEMM_A_BYTES = BOSCAME_GEMM_M * BOSCAME_GEMM_K,
  BOSCAME_GEMM_B_BYTES = BOSCAME_GEMM_N * BOSCAME_GEMM_K,
  BOSCAME_GEMM_C_BYTES = BOSCAME_GEMM_M * BOSCAME_GEMM_N * 4,
  BOSCAME_GEMM_PROGRAM_A_BYTES = BOSCAME_GEMM_BM * BOSCAME_GEMM_K,
  BOSCAME_GEMM_PROGRAM_C_BYTES = BOSCAME_GEMM_BM * BOSCAME_GEMM_N * 4
};

_Static_assert(
    sizeof(void *) == 8 && sizeof(int64_t) == 8,
    "the generated ranked-memref ABI requires 64-bit pointers/index");
_Static_assert(sizeof(int8_t) == 1 && sizeof(int32_t) == 4,
               "the generated GEMM ABI requires i8/i32 storage");
_Static_assert(BOSCAME_GEMM_M % BOSCAME_GEMM_BM == 0 &&
                   BOSCAME_GEMM_N % BOSCAME_GEMM_BN == 0,
               "the canonical launch has no partial programs");
_Static_assert(BOSCAME_GEMM_GRID_M == 2 && BOSCAME_GEMM_GRID_N == 1 &&
                   BOSCAME_GEMM_BK == BOSCAME_GEMM_K,
               "the canonical launch is a serial 2x1 grid with full K");
_Static_assert(BOSCAME_GEMM_PROGRAM_A_BYTES == 65536 &&
                   BOSCAME_GEMM_B_BYTES == 131072 &&
                   BOSCAME_GEMM_PROGRAM_C_BYTES == 32768,
               "the canonical DMA windows must match the saved MLIR");
_Static_assert(BOSCAME_GEMM_PROGRAM_A_BYTES % BOSCAME_GEMM_ALIGNMENT == 0 &&
                   BOSCAME_GEMM_PROGRAM_C_BYTES % BOSCAME_GEMM_ALIGNMENT == 0,
               "every program slice must preserve DMA alignment");

typedef struct {
  const int8_t *allocated;
  const int8_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} BoscameI8MemRef2;

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} BoscameI32MemRef2;

_Static_assert(sizeof(BoscameI8MemRef2) == 56 &&
                   offsetof(BoscameI8MemRef2, offset) == 16 &&
                   offsetof(BoscameI8MemRef2, sizes) == 24 &&
                   offsetof(BoscameI8MemRef2, strides) == 40 &&
                   sizeof(BoscameI32MemRef2) == sizeof(BoscameI8MemRef2),
               "rank-2 descriptor fields must match the lowered memref ABI");

typedef enum {
  BOSCAME_GEMM_OK = 0,
  BOSCAME_GEMM_INVALID_POINTER = 1,
  BOSCAME_GEMM_UNALIGNED = 2,
  BOSCAME_GEMM_OVERLAPPING = 3
} BoscameGemmStatus;

// A is row-major [M,K], B is physical row-major [N,K], C is row-major [M,N].
// These calls require complete caller-owned storage with the declared extents.
// Validation checks alignment, address overflow and disjointness, not capacity.
BoscameGemmStatus boscame_gemm_validate_roots(const int8_t *a, const int8_t *b,
                                              int32_t *c);

// Adapt one [BM,K] A slab and [BM,N] C slab. B is shared physical [N,K].
BoscameGemmStatus boscame_gemm_run_program(const int8_t *a, const int8_t *b,
                                           int32_t *c);

// Run the complete operation as two serial calls. C is an input/output buffer.
// The caller initializes the platform and DMA runtime before entry.
BoscameGemmStatus boscame_gemm_launch(const int8_t *a, const int8_t *b,
                                      int32_t *c);

#endif // BUDDY_EXAMPLES_LINALG_BOSCAME_ABI_H
