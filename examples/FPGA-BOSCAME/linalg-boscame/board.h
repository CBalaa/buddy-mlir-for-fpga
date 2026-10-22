// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef BUDDY_LINALG_BOSCAME_BOARD_H
#define BUDDY_LINALG_BOSCAME_BOARD_H

#include <stdint.h>

extern volatile uint32_t board_round;
extern volatile uint32_t board_programs[2];
extern volatile uint64_t board_dma_starts[2], board_dma_waits[2];
extern volatile uint64_t board_dma_bytes[2];
extern volatile uint64_t board_user_reads[2], board_user_writes[2];
extern volatile uint64_t board_user_read_bytes[2], board_user_write_bytes[2];
extern volatile uint64_t board_kernel_cycles[2], board_app_cycles;

void board_poison_spm(void);
void board_report(void);
static inline uint64_t board_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}
#endif
