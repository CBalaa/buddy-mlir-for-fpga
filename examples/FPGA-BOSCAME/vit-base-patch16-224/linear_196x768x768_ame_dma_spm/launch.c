#include <stddef.h>
#include <stdint.h>
#ifdef HOST_TEST
#include <stdio.h>
#endif

#include "linear_ame_dma_spm.h"

#define MAILBOX UINT64_C(0x50000160)

/* A DDR diagnostic cell is separate from the E6 completion MMIO register.
 * The latter is a hardware-owned status/handshake port; writing intermediate
 * values there makes reads return implementation status words. */
volatile uint32_t e6_diag_state
    __attribute__((section(".bss.e6_diag"), aligned(64)));
volatile uint32_t e6_diag_status
    __attribute__((section(".bss.e6_diag")));
volatile uint32_t e6_diag_errors
    __attribute__((section(".bss.e6_diag")));
volatile uint64_t e6_diag_cycles
    __attribute__((section(".bss.e6_diag")));

#ifndef VIT_LINEAR_TRACE
#define VIT_LINEAR_TRACE 0
#endif

/* The linker places these buffers in the low DDR ranges that the page-table
 * setup in linear_ame_dma_spm.c maps into the DMA virtual window. */
static int8_t a_tiles[VIT_LINEAR_A_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int8_t b_tiles[VIT_LINEAR_B_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int32_t o_tiles[VIT_LINEAR_O_TILES * VIT_LINEAR_TILE * VIT_LINEAR_TILE]
    __attribute__((section(".dma_output"), aligned(4096), used));

static inline void putc_uart(unsigned char value) {
#ifdef HOST_TEST
  (void)value;
#else
  /* E6 does not own the physical UART.  Keep the mailbox reserved for the
   * phase protocol below; NH forwards the final result to its UART. */
  (void)value;
#endif
}

/* The NH launcher polls this register while E6 is running.  Intermediate
 * values are deliberately outside the terminal protocol (2=PASS,
 * 0xed=trap, 0xee=FAIL), so a stalled phase remains observable without
 * requiring the E6 core to own the physical UART. */
static inline void e6_stage(unsigned value) {
#ifndef HOST_TEST
  *(volatile uint32_t *)(uintptr_t)MAILBOX = value;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  e6_diag_state = value;
  __asm__ volatile("fence rw, rw" ::: "memory");
  uintptr_t cfg = UINT64_C(0x0f0f0f0f0f0f0f0f);
  __asm__ volatile("csrw 0xbc0, %0" :: "r"(cfg) : "memory");
#else
  (void)value;
#endif
}
static inline void e6_diag_fence(void) {
#ifndef HOST_TEST
  __asm__ volatile("fence rw, rw" ::: "memory");
#endif
}
static void puts_uart(const char *text) {
#ifdef HOST_TEST
  /* The host harness uses stdout so it can be run without a UART model. */
  while (*text) putchar((unsigned char)*text++);
#else
  while (*text) putc_uart((unsigned char)*text++);
#endif
}

#if VIT_LINEAR_TRACE
#define trace_uart(text) puts_uart(text)
#else
#define trace_uart(text) ((void)0)
#endif
static void hex64(uint64_t value) {
#ifdef HOST_TEST
  printf("%016llx", (unsigned long long)value);
#else
  static const char digits[] = "0123456789abcdef";
  for (int shift = 60; shift >= 0; shift -= 4)
    putc_uart((unsigned char)digits[(value >> shift) & 0xfu]);
#endif
}
static inline uint64_t cycles(void) {
#ifdef HOST_TEST
  return 0;
#else
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
#endif
}

static int8_t a_value(unsigned row, unsigned k) {
  return (int8_t)(((row * 3u + k * 5u) % 13u) - 6);
}
static int8_t b_value(unsigned col, unsigned k) {
  return (int8_t)(((col * 7u + k * 3u) % 11u) - 5);
}

static void fill_tiles(void) {
  for (unsigned mb = 0; mb < VIT_LINEAR_MB; ++mb)
    for (unsigned kb = 0; kb < VIT_LINEAR_KB; ++kb)
      for (unsigned mi = 0; mi < VIT_LINEAR_TILE; ++mi)
        for (unsigned ki = 0; ki < VIT_LINEAR_TILE; ++ki) {
          unsigned row = mb * VIT_LINEAR_TILE + mi;
          unsigned k = kb * VIT_LINEAR_TILE + ki;
          a_tiles[(mb * VIT_LINEAR_KB + kb) * VIT_LINEAR_A_TILE_BYTES +
                  mi * 64 + ki] = row < VIT_LINEAR_M ? a_value(row, k) : 0;
        }
  for (unsigned nb = 0; nb < VIT_LINEAR_NB; ++nb)
    for (unsigned kb = 0; kb < VIT_LINEAR_KB; ++kb)
      for (unsigned nj = 0; nj < VIT_LINEAR_TILE; ++nj)
        for (unsigned ki = 0; ki < VIT_LINEAR_TILE; ++ki) {
          unsigned col = nb * VIT_LINEAR_TILE + nj;
          unsigned k = kb * VIT_LINEAR_TILE + ki;
          b_tiles[(nb * VIT_LINEAR_KB + kb) * VIT_LINEAR_B_TILE_BYTES +
                  nj * 64 + ki] = col < VIT_LINEAR_N ? b_value(col, k) : 0;
        }
  for (unsigned i = 0; i < sizeof(o_tiles) / sizeof(o_tiles[0]); ++i)
    o_tiles[i] = INT32_C(0x5a5a5a5a);
}

static int32_t oracle(unsigned row, unsigned col) {
  int32_t result = 0;
  for (unsigned k = 0; k < VIT_LINEAR_K; ++k)
    result += (int32_t)a_value(row, k) * (int32_t)b_value(col, k);
  return result;
}

/* The deterministic test pattern has a fixed full-output digest.  Recomputing
 * 196*768*768 scalar products on the 5.5296 MHz NH core makes a successful
 * accelerator run look like a hang (and can exceed the NH watchdog).  Scan
 * the complete tile-major result once and compare two independent 64-bit
 * digests instead; a small set of exact scalar points still checks the
 * numerical interpretation.  The host oracle above remains the exhaustive
 * reference used by `make host-check`. */
static int validate_output(unsigned *errors, int32_t *maximum) {
  uint64_t hash1 = UINT64_C(1469598103934665603);
  uint64_t hash2 = UINT64_C(0xcbf29ce484222325);
  uint64_t sum = 0;
  unsigned sample_errors = 0;
  int32_t sample_max = 0;
  for (unsigned mb = 0; mb < VIT_LINEAR_MB; ++mb)
    for (unsigned nb = 0; nb < VIT_LINEAR_NB; ++nb)
      for (unsigned mi = 0; mi < VIT_LINEAR_TILE; ++mi)
        for (unsigned nj = 0; nj < VIT_LINEAR_TILE; ++nj) {
          unsigned row = mb * VIT_LINEAR_TILE + mi;
          unsigned col = nb * VIT_LINEAR_TILE + nj;
          int32_t actual = o_tiles[(mb * VIT_LINEAR_NB + nb) * 4096 +
                                   mi * 64 + nj];
          uint64_t value = (uint32_t)actual;
          hash1 = (hash1 ^ value) * UINT64_C(1099511628211);
          hash2 = (hash2 + value * UINT64_C(0x9e3779b97f4a7c15)) ^
                  (hash2 >> 29);
          sum += value;
          /* About 150 points, each with a 768-term exact oracle. */
          if (row < VIT_LINEAR_M && col < VIT_LINEAR_N &&
              ((row * 131u + col * 17u) % 997u) == 0u) {
            int32_t expected = oracle(row, col);
            int32_t delta = actual - expected;
            if (delta < 0) delta = -delta;
            if (delta > sample_max) sample_max = delta;
            if (delta) ++sample_errors;
          } else if ((row >= VIT_LINEAR_M || col >= VIT_LINEAR_N) && actual) {
            ++sample_errors;
          }
        }
  /* Values were generated by the same periodic host pattern as fill_tiles. */
  const uint64_t expected_hash1 = UINT64_C(0x0e11181c8df22479);
  const uint64_t expected_hash2 = UINT64_C(0xbaf230bb25bf77c8);
  const uint64_t expected_sum = UINT64_C(0x0001280800000018);
  if (hash1 != expected_hash1 || hash2 != expected_hash2 ||
      sum != expected_sum)
    ++sample_errors;
  *errors = sample_errors;
  *maximum = sample_max;
  return sample_errors ? -1 : 0;
}

int e6_main(void) {
  e6_stage(0x11);
  puts_uart("[5.5296MHz][vit-linear-ame] start\r\n");
  trace_uart("[vit-linear-ame] E6 entry\r\n");
  trace_uart("[vit-linear-ame] fill begin\r\n");
  fill_tiles();
  trace_uart("[vit-linear-ame] fill done\r\n");
  e6_stage(0x12);
  trace_uart("[vit-linear-ame] kernel begin\r\n");
  e6_stage(0x20);
  uint64_t begin = cycles();
  int status = vit_linear_ame_i8(a_tiles, b_tiles, o_tiles);
  uint64_t elapsed = cycles() - begin;
  e6_diag_status = (uint32_t)status;
  e6_diag_cycles = elapsed;
  e6_diag_fence();
  e6_stage(status ? 0xe1 : 0x21);
  puts_uart("[vit-linear-ame] status="); hex64((uint64_t)(uint32_t)status);
  puts_uart(" cycles="); hex64(elapsed); puts_uart("\r\n");
  if (status) {
    puts_uart("[vit-linear-ame] FAIL dma_or_ame status\r\n");
    return 1;
  }

  unsigned errors = 0;
  e6_stage(0x30);
  int32_t maximum = 0;
  (void)validate_output(&errors, &maximum);
  puts_uart(errors ? "[vit-linear-ame] FAIL errors="
                   : "[vit-linear-ame] PASS errors=");
  hex64(errors); puts_uart(" max_abs="); hex64((uint32_t)maximum);
  puts_uart("\r\n");
  e6_diag_errors = errors;
  e6_diag_fence();
  e6_stage(errors ? 0xee : 2);
  return errors != 0;
}

#ifdef HOST_TEST
int main(void) { return e6_main(); }
#endif
