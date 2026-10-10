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
extern void _mlir_ciface_kernel_dwconv2d_144x144x40_k3_s1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef4 m_x = make_4(x, 1, 40, 144, 144);
  float *weight = (float *)workspace(3317760u);
  MemRef4 m_weight = make_4(weight, 40, 1, 3, 3);
  float *bias = (float *)workspace(3319232u);
  MemRef1 m_bias = make_1(bias, 40);
  float *out = (float *)workspace(3319424u);
  MemRef4 m_out = make_4(out, 1, 40, 144, 144);
  for (int i=0; i<829440; ++i) x[i]=(float)((i*7)%23-11)*0.03125f;
  for (int i=0; i<360; ++i) weight[i]=(float)((i*3)%11-5)*0.0625f;
  for (int i=0; i<40; ++i) bias[i]=(float)((i%5)-2)*0.125f;
  for (int i=0; i<829440; ++i) out[i]=9999.0f;
  _mlir_ciface_kernel_dwconv2d_144x144x40_k3_s1(&m_x, &m_weight, &m_bias, &m_out);
  for (int ch=0; ch<40; ++ch) for (int oh=0; oh<144; ++oh) for (int ow=0; ow<144; ++ow) {
    oracle_float acc = bias[ch];
    for (int kh=0; kh<3; ++kh) for (int kw=0; kw<3; ++kw) {
      int ih = oh*1 - 1 + kh;
      int iw = ow*1 - 1 + kw;
      if (ih>=0 && iw>=0 && ih<144 && iw<144)
        acc += x[(ch*144+ih)*144+iw] * weight[(ch*3+kh)*3+kw];
    }
    compare(out[(ch*144+oh)*144+ow], (float)acc, 8e-5f, 2e-4f, &errors, &maximum);
  }
  return print_check("dwconv2d_144x144x40_k3_s1", errors, maximum);
}
