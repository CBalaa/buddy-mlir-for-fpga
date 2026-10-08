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
extern void _mlir_ciface_kernel_linear_1x384x16(MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *a = (float *)workspace(0u);
  MemRef2 m_a = make_2(a, 1, 16);
  float *w = (float *)workspace(64u);
  MemRef2 m_w = make_2(w, 384, 16);
  float *bias = (float *)workspace(24640u);
  MemRef1 m_bias = make_1(bias, 384);
  float *out = (float *)workspace(26176u);
  MemRef2 m_out = make_2(out, 1, 384);
  for (int i=0; i<16; ++i) a[i]=(float)((i*11)%17-8)*0.0625f;
  for (int i=0; i<6144; ++i) w[i]=(float)((i*3)%19-9)*0.03125f;
  for (int i=0; i<384; ++i) bias[i]=(float)((i%7)-3)*0.125f;
  for (int i=0; i<384; ++i) out[i]=9999.0f;
  _mlir_ciface_kernel_linear_1x384x16(&m_a, &m_w, &m_bias, &m_out);
  for (int i=0; i<1; ++i) for (int j=0; j<384; ++j) {
    oracle_float acc = bias[j];
    for (int t=0; t<16; ++t) acc += a[i*16+t] * w[j*16+t];
    compare(out[i*384+j], (float)acc, 5e-5f, 1e-4f, &errors, &maximum);
  }
  return print_check("linear_1x384x16", errors, maximum);
}
