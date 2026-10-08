#include "uart.h"

void init_uart(uint32_t freq, uint32_t baud) {
  /* Keep the NR word-wide MMIO map and use the upstream init contract. */
  uart_configure(uart_divisor_for(freq, baud));
}

void write_serial(char value) { uart_putc(value); }

void print_uart(const char *text) {
  while (*text != '\0')
    uart_putc(*text++);
}

void print_uart_int(uint32_t value) {
  static const char digits[] = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4)
    uart_putc(digits[(value >> shift) & 0xfu]);
}

void print_uart_addr(uint64_t value) {
  print_uart_int((uint32_t)(value >> 32));
  print_uart_int((uint32_t)value);
}
