// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "board.h"
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"
#include <stddef.h>

volatile uint32_t board_round, board_programs[2];
volatile uint64_t board_dma_starts[2], board_dma_waits[2], board_dma_bytes[2];
volatile uint64_t board_user_reads[2], board_user_writes[2];
volatile uint64_t board_user_read_bytes[2], board_user_write_bytes[2];
volatile uint64_t board_kernel_cycles[2], board_app_cycles;

extern uint64_t __real_bosc_ame_dma_start(uint32_t, uint64_t, uint64_t,
                                          uint64_t);
extern void __real_bosc_ame_dma_wait(uint32_t, uint64_t);

uint64_t __wrap_bosc_ame_dma_start(uint32_t channel, uint64_t src, uint64_t dst,
                                   uint64_t bytes) {
  uint64_t token = __real_bosc_ame_dma_start(channel, src, dst, bytes);
  if (token) {
    unsigned round = board_round;
    ++board_dma_starts[round];
    board_dma_bytes[round] += bytes;
    // The linker puts user arrays in PMP entry 5 and ordinary stack scratch
    // above it. Count actual user traffic separately from scratch staging.
    if (src >= UINT64_C(0x80100000) && src < UINT64_C(0x802e0000)) {
      ++board_user_reads[round];
      board_user_read_bytes[round] += bytes;
    }
    if (dst >= UINT64_C(0x80100000) && dst < UINT64_C(0x802e0000)) {
      ++board_user_writes[round];
      board_user_write_bytes[round] += bytes;
    }
  }
  return token;
}

void __wrap_bosc_ame_dma_wait(uint32_t channel, uint64_t token) {
  __real_bosc_ame_dma_wait(channel, token);
  ++board_dma_waits[board_round];
}

void board_poison_spm(void) {
  for (unsigned bank = 0; bank < 2; ++bank) {
    uintptr_t base = bank ? BOSC_AME_SPM1_BASE : BOSC_AME_SPM0_BASE;
    for (unsigned offset = 0; offset < BOSC_AME_SPM_BANK_BYTES; offset += 64)
      __asm__ volatile(".option push\n.option arch, +v\n"
                       "vsetivli zero, 16, e32, m1, ta, ma\n"
                       "vmv.v.x v0, %1\n"
                       "fence rw, rw\nvse32.v v0, (%0)\nfence rw, rw\n"
                       ".option pop\n"
                       :
                       : "r"(base + offset), "r"(UINT32_C(0x4f400001))
                       : "memory");
  }
}

static void field(const char *name, uint64_t value) {
  print_uart(name);
  print_uart("=0x");
  print_uart_hex64(value);
  print_uart(" ");
}

void board_report(void) {
  for (unsigned round = 0; round < 2; ++round) {
    nh_e6_print_tag_prefix();
    print_uart("round=");
    print_uart_dec32(round);
    print_uart(" ");
    field("programs", board_programs[round]);
    field("starts", board_dma_starts[round]);
    field("waits", board_dma_waits[round]);
    field("bytes", board_dma_bytes[round]);
    field("user_reads", board_user_reads[round]);
    field("user_read_bytes", board_user_read_bytes[round]);
    field("user_writes", board_user_writes[round]);
    field("user_write_bytes", board_user_write_bytes[round]);
    field("kernel_cycles", board_kernel_cycles[round]);
    print_uart("\r\n");
  }
  nh_e6_print_tag_prefix();
  field("app_cycles", board_app_cycles);
  print_uart("\r\n");
}

// Ordinary CPU copies in the explicit fallback have freestanding definitions.
// Compile this file without automatic vectorization or builtins.
void *memcpy(void *dst, const void *src, size_t bytes) {
  unsigned char *output = dst;
  const unsigned char *input = src;
  for (size_t i = 0; i < bytes; ++i)
    output[i] = input[i];
  return dst;
}

void *memset(void *dst, int value, size_t bytes) {
  unsigned char *output = dst;
  for (size_t i = 0; i < bytes; ++i)
    output[i] = (unsigned char)value;
  return dst;
}
