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
extern void _mlir_ciface_kernel_dwconv2d_72x72x192_k3_s1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef4 m_x = make_4(x, 1, 192, 72, 72);
  float *weight = (float *)workspace(3981312u);
  MemRef4 m_weight = make_4(weight, 192, 1, 3, 3);
  float *bias = (float *)workspace(3988224u);
  MemRef1 m_bias = make_1(bias, 192);
  float *out = (float *)workspace(3988992u);
  MemRef4 m_out = make_4(out, 1, 192, 72, 72);
  for (int i=0; i<995328; ++i) x[i]=(float)((i*7)%23-11)*0.03125f;
  for (int i=0; i<1728; ++i) weight[i]=(float)((i*3)%11-5)*0.0625f;
  for (int i=0; i<192; ++i) bias[i]=(float)((i%5)-2)*0.125f;
  for (int i=0; i<995328; ++i) out[i]=9999.0f;
  _mlir_ciface_kernel_dwconv2d_72x72x192_k3_s1(&m_x, &m_weight, &m_bias, &m_out);
  for (int ch=0; ch<192; ++ch) for (int oh=0; oh<72; ++oh) for (int ow=0; ow<72; ++ow) {
    oracle_float acc = bias[ch];
    for (int kh=0; kh<3; ++kh) for (int kw=0; kw<3; ++kw) {
      int ih = oh*1 - 1 + kh;
      int iw = ow*1 - 1 + kw;
      if (ih>=0 && iw>=0 && ih<72 && iw<72)
        acc += x[(ch*72+ih)*72+iw] * weight[(ch*3+kh)*3+kw];
    }
    compare(out[(ch*72+oh)*72+ow], (float)acc, 8e-5f, 2e-4f, &errors, &maximum);
  }
  return print_check("dwconv2d_72x72x192_k3_s1", errors, maximum);
}
