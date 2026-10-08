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
extern void _mlir_ciface_kernel_sigmoid_1x384(MemRef2 *, MemRef2 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef2 m_x = make_2(x, 1, 384);
  float *out = (float *)workspace(1536u);
  MemRef2 m_out = make_2(out, 1, 384);
  for (int i=0; i<384; ++i) { x[i]=(float)((i*13)%129-64)*0.125f; out[i]=9999.0f; }
  _mlir_ciface_kernel_sigmoid_1x384(&m_x, &m_out);
  for (int i=0; i<384; ++i) compare(out[i], 1.0f / (1.0f + oracle_exp(-x[i])), 2e-6f, 3e-5f, &errors, &maximum);
  return print_check("sigmoid_1x384", errors, maximum);
}
