/* Host validation of the production invocation-private DDR frame allocator. */
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static _Alignas(64) unsigned char arena[262144];
static uintptr_t window_start, window_end, hart;
static unsigned writes, reads;
#ifndef BOSC_AME_PRIVATE_DDR_TEST_RESET
#define BOSC_AME_PRIVATE_DDR_TEST_RESET() ((void)0)
#endif
#ifndef BOSC_AME_PRIVATE_DDR_TEST_SHARED
#define BOSC_AME_PRIVATE_DDR_TEST_SHARED() ((void)0)
#endif
#define BOSC_AME_PRIVATE_DDR_START ((uintptr_t)arena)
#define BOSC_AME_PRIVATE_DDR_END ((uintptr_t)(arena + sizeof(arena)))
#define BOSC_AME_PRIVATE_DDR_WINDOW_START window_start
#define BOSC_AME_PRIVATE_DDR_WINDOW_END window_end
#define BOSC_AME_PRIVATE_DDR_HART_ID() hart
#define BOSC_AME_DMA_SHARED_WRITES()                                           \
  (++writes, BOSC_AME_PRIVATE_DDR_TEST_SHARED())
#define BOSC_AME_DMA_SHARED_READS()                                            \
  (++reads, BOSC_AME_PRIVATE_DDR_TEST_SHARED())
#include "bosc_ame_dma.c"

static jmp_buf failure;
static uint32_t last_reason;
static void fail_hook(uint32_t reason, uint64_t address, uint64_t bytes) {
  (void)address;
  (void)bytes;
  last_reason = reason;
  longjmp(failure, 1);
}
static void reset(void) {
  memset(arena, 0xA5, sizeof(arena));
  window_start = (uintptr_t)arena;
  window_end = (uintptr_t)(arena + sizeof(arena));
  hart = 0;
  writes = reads = 0;
  bosc_ame_private_ddr_top = bosc_ame_private_ddr_active =
      bosc_ame_private_ddr_owner = 0;
  bosc_ame_private_ddr_busy = 0;
  bosc_ame_private_ddr_peak_bytes = bosc_ame_private_ddr_acquires =
      bosc_ame_private_ddr_releases = 0;
  bosc_ame_private_ddr_fail_hook = fail_hook;
  BOSC_AME_PRIVATE_DDR_TEST_RESET();
}
#define FAILS(reason, action)                                                  \
  do {                                                                         \
    last_reason = 0;                                                           \
    if (setjmp(failure) == 0) {                                                \
      action;                                                                  \
      assert(!"expected checked failure");                                     \
    }                                                                          \
    assert(last_reason == (reason));                                           \
  } while (0)

int main(void) {
  reset();
  uint64_t outer = bosc_ame_private_ddr_acquire(8192);
  assert(outer == (uintptr_t)arena + 64 && !(outer & 63));
  uint64_t inner = bosc_ame_private_ddr_acquire(4096);
  assert(inner == outer + 8192 + 128);
  memset((void *)(uintptr_t)outer, 0x71, 8192);
  memset((void *)(uintptr_t)inner, 0x93, 4096);
  bosc_ame_private_ddr_release(inner, 4096);
  bosc_ame_private_ddr_release(outer, 8192);
  assert(!bosc_ame_private_ddr_top && !bosc_ame_private_ddr_owner);
  assert(writes == 4 && reads == 2);
  assert(bosc_ame_private_ddr_acquires == 2 &&
         bosc_ame_private_ddr_releases == 2 &&
         bosc_ame_private_ddr_peak_bytes == 8192 + 4096 + 256);
  for (unsigned i = 0; i != 3; ++i) {
    uint64_t address = bosc_ame_private_ddr_acquire(8192);
    assert(address == outer);
    bosc_ame_private_ddr_release(address, 8192);
  }

  reset();
  uint64_t maximum =
      bosc_ame_private_ddr_acquire(BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD);
  assert(bosc_ame_private_ddr_peak_bytes == sizeof(arena));
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, bosc_ame_private_ddr_acquire(64));
  reset();
  maximum = bosc_ame_private_ddr_acquire(BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD);
  bosc_ame_private_ddr_release(maximum, BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD);
  reset();
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, bosc_ame_private_ddr_acquire(0));
  reset();
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, bosc_ame_private_ddr_acquire(65));
  reset();
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY,
        bosc_ame_private_ddr_acquire(UINT64_MAX));
  reset();
  window_start += 64;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_WINDOW, bosc_ame_private_ddr_acquire(64));
  reset();
  window_end -= 64;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_WINDOW, bosc_ame_private_ddr_acquire(64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  hart = 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, bosc_ame_private_ddr_acquire(64));
  reset();
  bosc_ame_private_ddr_busy = 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CONCURRENT, bosc_ame_private_ddr_acquire(64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  inner = bosc_ame_private_ddr_acquire(64);
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
        bosc_ame_private_ddr_release(outer, UINT64_MAX));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  bosc_ame_private_ddr_release(outer, 64);
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  arena[0] ^= 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  arena[128] ^= 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  arena[32] ^= 1; /* Parent metadata must be guarded too. */
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
        bosc_ame_private_ddr_release(outer, 64));
  puts("private DDR frame: all checks passed");
  return 0;
}
