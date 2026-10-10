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
extern void _mlir_ciface_kernel_conv2d_18x18_288x96_k1_s1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef4 m_x = make_4(x, 1, 288, 18, 18);
  float *weight = (float *)workspace(373248u);
  MemRef4 m_weight = make_4(weight, 96, 288, 1, 1);
  float *bias = (float *)workspace(483840u);
  MemRef1 m_bias = make_1(bias, 96);
  float *out = (float *)workspace(484224u);
  MemRef4 m_out = make_4(out, 1, 96, 18, 18);
  for (int i=0; i<93312; ++i) x[i]=(float)((i*13)%21-10)*0.03125f;
  for (int i=0; i<27648; ++i) weight[i]=(float)((i*5)%15-7)*0.03125f;
  for (int i=0; i<96; ++i) bias[i]=(float)((i%5)-2)*0.125f;
  for (int i=0; i<31104; ++i) out[i]=9999.0f;
  _mlir_ciface_kernel_conv2d_18x18_288x96_k1_s1(&m_x, &m_weight, &m_bias, &m_out);
  for (int oc=0; oc<96; ++oc) for (int oh=0; oh<18; ++oh) for (int ow=0; ow<18; ++ow) {
    oracle_float acc = bias[oc];
    for (int ic=0; ic<288; ++ic) for (int kh=0; kh<1; ++kh) for (int kw=0; kw<1; ++kw) {
      int ih = oh*1 - 0 + kh;
      int iw = ow*1 - 0 + kw;
      if (ih>=0 && iw>=0 && ih<18 && iw<18)
        acc += x[(ic*18+ih)*18+iw] *
               weight[((oc*288+ic)*1+kh)*1+kw];
    }
    compare(out[(oc*18+oh)*18+ow], (float)acc, 8e-5f, 2e-4f, &errors, &maximum);
  }
  return print_check("conv2d_18x18_288x96_k1_s1", errors, maximum);
}
