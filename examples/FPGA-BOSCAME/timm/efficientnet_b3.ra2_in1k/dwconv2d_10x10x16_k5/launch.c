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
extern void _mlir_ciface_kernel_dwconv2d_10x10x16_k5(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef4 m_x = make_4(x, 1, 16, 10, 10);
  float *weight = (float *)workspace(6400u);
  MemRef4 m_weight = make_4(weight, 16, 1, 5, 5);
  float *bias = (float *)workspace(8000u);
  MemRef1 m_bias = make_1(bias, 16);
  float *out = (float *)workspace(8064u);
  MemRef4 m_out = make_4(out, 1, 16, 10, 10);
  for (int i=0; i<1600; ++i) x[i]=(float)((i*7)%23-11)*0.03125f;
  for (int i=0; i<400; ++i) weight[i]=(float)((i*3)%11-5)*0.0625f;
  for (int i=0; i<16; ++i) bias[i]=(float)((i%5)-2)*0.125f;
  for (int i=0; i<1600; ++i) out[i]=9999.0f;
  _mlir_ciface_kernel_dwconv2d_10x10x16_k5(&m_x, &m_weight, &m_bias, &m_out);
  for (int ch=0; ch<16; ++ch) for (int oh=0; oh<10; ++oh) for (int ow=0; ow<10; ++ow) {
    oracle_float acc = bias[ch];
    for (int kh=0; kh<5; ++kh) for (int kw=0; kw<5; ++kw) {
      int ih = oh*1 - 2 + kh;
      int iw = ow*1 - 2 + kw;
      if (ih>=0 && iw>=0 && ih<10 && iw<10)
        acc += x[(ch*10+ih)*10+iw] * weight[(ch*5+kh)*5+kw];
    }
    compare(out[(ch*10+oh)*10+ow], (float)acc, 8e-5f, 2e-4f, &errors, &maximum);
  }
  return print_check("dwconv2d_10x10x16_k5", errors, maximum);
}
