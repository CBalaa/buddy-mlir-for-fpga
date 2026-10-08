#ifndef VIT_LINEAR_AME_DMA_SPM_H
#define VIT_LINEAR_AME_DMA_SPM_H

#include <stdint.h>

/* ViT-B/16 patch projection dimensions. */
#define VIT_LINEAR_M 196u
#define VIT_LINEAR_N 768u
#define VIT_LINEAR_K 768u
#define VIT_LINEAR_TILE 64u
#define VIT_LINEAR_MB 4u
#define VIT_LINEAR_NB 12u
#define VIT_LINEAR_KB 12u

#define VIT_LINEAR_A_TILE_BYTES (VIT_LINEAR_TILE * VIT_LINEAR_TILE)
#define VIT_LINEAR_B_TILE_BYTES (VIT_LINEAR_TILE * VIT_LINEAR_TILE)
#define VIT_LINEAR_C_TILE_BYTES \
  (VIT_LINEAR_TILE * VIT_LINEAR_TILE * sizeof(int32_t))
#define VIT_LINEAR_A_TILES (VIT_LINEAR_MB * VIT_LINEAR_KB)
#define VIT_LINEAR_B_TILES (VIT_LINEAR_NB * VIT_LINEAR_KB)
#define VIT_LINEAR_O_TILES (VIT_LINEAR_MB * VIT_LINEAR_NB)
#define VIT_LINEAR_A_BYTES (VIT_LINEAR_A_TILES * VIT_LINEAR_A_TILE_BYTES)
#define VIT_LINEAR_B_BYTES (VIT_LINEAR_B_TILES * VIT_LINEAR_B_TILE_BYTES)
#define VIT_LINEAR_O_BYTES (VIT_LINEAR_O_TILES * VIT_LINEAR_C_TILE_BYTES)

/*
 * A and B use tile-major storage:
 *
 *   A[(mb*KB + kb)][mi][ki]  (64x64, zero padded at the M tail)
 *   B[(nb*KB + kb)][nj][ki]  (64x64)
 *
 * O has the corresponding tile-major [MB][NB][64][64] int32 layout.  The
 * layout makes every DMA transfer one contiguous, 64-byte aligned range;
 * no DMA descriptor is asked to perform a strided two-dimensional copy.
 */
int vit_linear_ame_i8(const int8_t *a_tiles, const int8_t *b_tiles,
                      int32_t *o_tiles);

/* FP32 convenience wrapper.  It performs symmetric per-row A and per-output
 * channel B quantization, invokes the INT8 AME path, then dequantizes to C.
 * AME is an INT8/FP8 engine, so this is an explicitly approximate path and
 * must be accuracy-tested before replacing the model's FP32 linear call. */
int vit_linear_ame_f32(const float *a, const float *b, const float *bias,
                       float *out);

#endif
