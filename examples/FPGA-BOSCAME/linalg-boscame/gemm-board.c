// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "abi.h"
#include "board.h"
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"
#if BOSCAME_BOARD_FRONTEND
#include "frontend_abi.h"
#endif

#define M BOSCAME_GEMM_M
#define N BOSCAME_GEMM_N
#define K BOSCAME_GEMM_K
#define WORDS 256u
#define GUARD UINT32_C(0x6e5a1234)

typedef struct {
  uint32_t before[WORDS];
  int8_t values[M * K];
  uint32_t after[WORDS];
} AStorage;
typedef struct {
  uint32_t before[WORDS];
  int8_t values[N * K];
  uint32_t after[WORDS];
} BStorage;
typedef struct {
  uint32_t before[WORDS];
  uint32_t values[M * N];
  uint32_t after[WORDS];
} CStorage;
AStorage gemm_a __attribute__((section(".dma_source"), aligned(4096), used));
BStorage gemm_b __attribute__((section(".dma_source"), aligned(4096), used));
CStorage gemm_c __attribute__((section(".dma_output"), aligned(4096), used));
volatile uint32_t gemm_round, gemm_checked, gemm_exact;
volatile uint32_t gemm_errors, gemm_guard_errors;

static int8_t a_value(unsigned row, unsigned k, unsigned round) {
  return (int8_t)((int)((row * 13u + k * 7u + round * 71u) % 256u) - 128);
}
static int8_t b_value(unsigned column, unsigned k, unsigned round) {
  return (int8_t)((int)((column * 11u + k * 5u + round * 37u) % 256u) - 128);
}
static uint32_t c_value(unsigned row, unsigned column, unsigned round) {
  return UINT32_C(0x7fffff00) + (row * 3u + column + round * 97u) % 512u;
}
static void guards(uint32_t *before, uint32_t *after) {
  for (unsigned i = 0; i < WORDS; ++i)
    before[i] = after[i] = GUARD;
}
static uint32_t check_guards(const uint32_t *before, const uint32_t *after) {
  uint32_t errors = 0;
  for (unsigned i = 0; i < WORDS; ++i)
    errors += (before[i] != GUARD) + (after[i] != GUARD);
  return errors;
}
static void initialize(unsigned round) {
  guards(gemm_a.before, gemm_a.after);
  guards(gemm_b.before, gemm_b.after);
  guards(gemm_c.before, gemm_c.after);
  for (unsigned row = 0; row < M; ++row)
    for (unsigned k = 0; k < K; ++k)
      gemm_a.values[row * K + k] = a_value(row, k, round);
  for (unsigned column = 0; column < N; ++column)
    for (unsigned k = 0; k < K; ++k)
      gemm_b.values[column * K + k] = b_value(column, k, round);
  for (unsigned row = 0; row < M; ++row)
    for (unsigned column = 0; column < N; ++column)
      gemm_c.values[row * N + column] = c_value(row, column, round);
}
static void validate(unsigned round) {
  for (unsigned row = 0; row < M; ++row) {
    for (unsigned column = 0; column < N; ++column) {
      uint32_t wanted = c_value(row, column, round);
      for (unsigned k = 0; k < K; ++k)
        wanted += (uint32_t)((int32_t)a_value(row, k, round) *
                             (int32_t)b_value(column, k, round));
      uint32_t actual = gemm_c.values[row * N + column];
      gemm_errors += actual != wanted;
      gemm_exact += actual == wanted;
      ++gemm_checked;
    }
    nh_e6_progress(0x2000u + row);
  }
  for (unsigned row = 0; row < M; ++row)
    for (unsigned k = 0; k < K; ++k)
      gemm_errors += gemm_a.values[row * K + k] != a_value(row, k, round);
  for (unsigned column = 0; column < N; ++column)
    for (unsigned k = 0; k < K; ++k)
      gemm_errors += gemm_b.values[column * K + k] != b_value(column, k, round);
  gemm_guard_errors += check_guards(gemm_a.before, gemm_a.after) +
                       check_guards(gemm_b.before, gemm_b.after) +
                       check_guards(gemm_c.before, gemm_c.after);
}

int e6_main(void) {
  uint64_t app_begin = board_cycles();
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  board_poison_spm();
  for (unsigned round = 0; round < 2; ++round) {
    gemm_round = board_round = round;
    nh_e6_cache_writes();
    initialize(round);
    uint64_t begin = board_cycles();
#if BOSCAME_BOARD_FRONTEND
    if (boscame_frontend_gemm_launch(gemm_a.values, gemm_b.values,
                                     (int32_t *)gemm_c.values))
      return 2;
    board_programs[round] = BOSCAME_GEMM_GRID_M;
#else
    // Reverse the two explicit programs, retaining dirty SPM between calls.
    for (unsigned index = 0; index < BOSCAME_GEMM_GRID_M; ++index) {
      unsigned row = (BOSCAME_GEMM_GRID_M - 1 - index) * BOSCAME_GEMM_BM;
      if (boscame_gemm_run_program(gemm_a.values + row * K, gemm_b.values,
                                   (int32_t *)gemm_c.values + row * N) !=
          BOSCAME_GEMM_OK)
        return 2;
      ++board_programs[round];
    }
#endif
    board_kernel_cycles[round] = board_cycles() - begin;
    nh_e6_cache_uncached_loads();
    validate(round);
    nh_e6_cache_writes();
    if (gemm_errors || gemm_guard_errors)
      return 1;
  }
  board_app_cycles = board_cycles() - app_begin;
  return 0;
}

const char *nh_e6_app_tag(void) { return "linalg-boscame-gemm"; }
void nh_e6_app_report(void) {
  nh_e6_print_tag_prefix();
  print_uart("checked=");
  print_uart_dec32(gemm_checked);
  print_uart(" exact=");
  print_uart_dec32(gemm_exact);
  print_uart(" errors=");
  print_uart_dec32(gemm_errors);
  print_uart(" guard_errors=");
  print_uart_dec32(gemm_guard_errors);
  print_uart("\r\n");
  board_report();
}
