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
extern void _mlir_ciface_kernel_gap_144x144x24(MemRef4 *, MemRef2 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef4 m_x = make_4(x, 1, 24, 144, 144);
  float *out = (float *)workspace(1990656u);
  MemRef2 m_out = make_2(out, 1, 24);
  for (int i=0; i<497664; ++i) x[i]=(float)((i*5)%41-20)*0.03125f;
  for (int i=0; i<24; ++i) out[i]=9999.0f;
  _mlir_ciface_kernel_gap_144x144x24(&m_x, &m_out);
  for (int ch=0; ch<24; ++ch) {
    oracle_float s=0;
    for (int p=0; p<20736; ++p) s += x[ch*20736+p];
    compare(out[ch], (float)(s/20736.0), 3e-6f, 5e-5f, &errors, &maximum);
  }
  return print_check("gap_144x144x24", errors, maximum);
}
