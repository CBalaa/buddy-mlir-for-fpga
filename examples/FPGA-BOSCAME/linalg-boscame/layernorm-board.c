// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "board.h"
#include "layernorm_abi.h"
#if BOSCAME_BOARD_FRONTEND
#include "frontend_abi.h"
#endif

// Retain the existing independent 1024-lane C oracle, guarded DDR layout,
// two input rounds and exact input/tail checks. Only the compiler ABI changes.
#define e6_main reference_layernorm_main
#define nh_e6_app_report reference_layernorm_report
#define nh_e6_app_tag reference_layernorm_tag
#include "../layernorm_197x768/launch.c"
#undef e6_main
#undef nh_e6_app_report
#undef nh_e6_app_tag

void _mlir_ciface_kernel_layernorm_197x768(MemRef2 *x, MemRef1 *weight,
                                           MemRef1 *bias, MemRef2 *out) {
  unsigned round = board_round = layernorm_round;
  const float *xp = (const float *)x->aligned + x->offset;
  const float *wp = (const float *)weight->aligned + weight->offset;
  const float *bp = (const float *)bias->aligned + bias->offset;
  float *op = (float *)out->aligned + out->offset;
  uint64_t begin = board_cycles();
#if BOSCAME_BOARD_FRONTEND
  if (boscame_frontend_layernorm_launch(xp, wp, bp, op))
    __builtin_trap();
  board_programs[round] = ROWS;
#else
  for (unsigned index = 0; index < ROWS; ++index) {
    unsigned row = ROWS - 1 - index;
    nh_e6_progress(0x1000u + row);
    if (boscame_layernorm_run_program(xp + row * WIDTH, wp, bp,
                                      op + row * WIDTH) != BOSCAME_LAYERNORM_OK)
      __builtin_trap();
    ++board_programs[round];
  }
#endif
  board_kernel_cycles[round] = board_cycles() - begin;
}

int e6_main(void) {
  uint64_t begin = board_cycles();
  int status = reference_layernorm_main();
  board_app_cycles = board_cycles() - begin;
  return status;
}
const char *nh_e6_app_tag(void) { return "linalg-boscame-layernorm"; }
void nh_e6_app_report(void) {
  reference_layernorm_report();
  board_report();
}
