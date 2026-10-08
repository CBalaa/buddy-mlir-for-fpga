/*===- launch.c - EfficientNet operator oracle -------------------*- C -*-===*
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * See the LICENSE file at the repository root.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated C oracle. Do not hand-edit; regenerate with generate_cases.py.
 *
 *===----------------------------------------------------------------------===*/
#include "support.h"
#ifdef HOST_TEST
typedef double oracle_float;
static float oracle_exp(float x) { return (float)exp((double)x); }
#else
typedef float oracle_float;
static float oracle_exp(float x) { return expf(x); }
#endif
static void compare(float actual, float expected, float atol, float rtol,
                    int *errors, float *maximum) {
  if (actual == expected) return;
  if (!__builtin_isfinite(actual) || !__builtin_isfinite(expected)) {
    ++*errors; *maximum = __builtin_inff(); return;
  }
  if (!check_close(actual, expected, atol, rtol)) ++*errors;
  float d = actual - expected; if (d < 0) d = -d;
  if (d > *maximum) *maximum = d;
}
extern void _mlir_ciface_kernel_se_scale_9x9x32(MemRef4 *, MemRef2 *, MemRef4 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef4 m_x = make_4(x, 1, 32, 9, 9);
  float *scale = (float *)workspace(10368u);
  MemRef2 m_scale = make_2(scale, 1, 32);
  float *out = (float *)workspace(10496u);
  MemRef4 m_out = make_4(out, 1, 32, 9, 9);
  for (int i=0; i<2592; ++i) { x[i]=(float)((i*9)%29-14)*0.0625f; out[i]=9999.0f; }
  for (int i=0; i<32; ++i) scale[i]=(float)((i%13)+1)*0.125f;
  _mlir_ciface_kernel_se_scale_9x9x32(&m_x, &m_scale, &m_out);
  for (int ch=0; ch<32; ++ch) for (int p=0; p<81; ++p)
    compare(out[ch*81+p], x[ch*81+p]*scale[ch], 2e-6f, 3e-5f, &errors, &maximum);
  return print_check("se_scale_9x9x32", errors, maximum);
}
