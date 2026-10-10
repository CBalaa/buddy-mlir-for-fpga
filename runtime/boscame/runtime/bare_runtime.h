#ifndef EXAMPLES_TOOLS_BARE_RUNTIME_H
#define EXAMPLES_TOOLS_BARE_RUNTIME_H

/*
 * Public ABI of the bare single-hart runtime.  The implementation is split by
 * responsibility across bare_runtime.c (C entry, weak hooks, trap handler),
 * bare_runtime_memory.c (heap, malloc/free, memory and string helpers),
 * bare_runtime_memref.c (memrefCopy), bare_runtime_trace.c (buddyTraceCycle*)
 * and bare_runtime_ame.c (ame_fence); see common/README.md for the module
 * table.
 */

#include <stddef.h>
#include <stdint.h>

void *bare_runtime_heap_base(void);
size_t bare_runtime_heap_offset(void);
void bare_runtime_preserve_heap(void);
void bare_runtime_reset_heap(void);

void bare_runtime_print_banner(void);
void bare_runtime_before_main(void);
void bare_runtime_after_main(void);

/* Buddy cycle-trace support for bare-metal targets. */
void bare_trace_reset(void);
void bare_trace_print(void);

void ame_fence(void);

#endif
