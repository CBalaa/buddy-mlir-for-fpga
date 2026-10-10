/*
 * Bare single-hart runtime entry point.
 *
 * This file owns the startup side of the bare backend: the C `_init` hook that
 * crt_uart.S calls after it has built the execution environment, the weak
 * sample hooks around `main`, and the terminal trap handler.  The supporting
 * blocks live in their own translation units (see bare_runtime_internal.h);
 * the allocator is reached through the public bare_runtime.h API so `_init`
 * keeps its original call order.
 */

#include "bare_runtime.h"
#include "bare_runtime_internal.h"
#include "uart.h"
#include <stddef.h>
#include <stdint.h>

#ifndef UART_FREQ_HZ
#define UART_FREQ_HZ 5529600u
#endif

#ifndef UART_BAUD
#define UART_BAUD 115200u
#endif

__attribute__((weak)) void bare_runtime_print_banner(void)
{
}

__attribute__((weak)) void bare_runtime_before_main(void)
{
}

__attribute__((weak)) void bare_runtime_after_main(void)
{
}

#if !defined(BARE_RUNTIME_HOST_TEST)
__attribute__((weak)) uintptr_t handle_trap(uintptr_t mcause, uintptr_t mepc, uintptr_t regs)
{
    (void)regs;

    print_uart("\r\nTRAP mcause=0x");
    print_uart_addr((uint64_t)mcause);
    print_uart(" mepc=0x");
    print_uart_addr((uint64_t)mepc);
    uintptr_t mtval;
    __asm__ volatile("csrr %0, mtval" : "=r"(mtval));
    print_uart(" mtval=0x");
    print_uart_addr((uint64_t)mtval);
    print_uart("\r\n");

    while (1)
        __asm__ volatile ("wfi");
}

#if !defined(BARE_RUNTIME_CUSTOM_ENTRY)
void _init(void)
{
    bare_runtime_reset_heap();
    init_uart(UART_FREQ_HZ, UART_BAUD);

    bare_runtime_print_banner();

    extern int main(void);
    bare_runtime_before_main();
    main();
    bare_runtime_after_main();

    while (1)
        __asm__ volatile ("wfi");
}
#endif
#endif
