#ifndef EXAMPLES_TOOLS_BARE_RUNTIME_INTERNAL_H
#define EXAMPLES_TOOLS_BARE_RUNTIME_INTERNAL_H

/*
 * Private interface of the bare single-hart runtime.
 *
 * bare_runtime.c owns the C entry (`_init`), the weak startup hooks and the
 * terminal trap handler.  The supporting blocks are separate translation units
 * so that a sample links only the parts it uses and the module boundary follows
 * the runtime responsibility, not the include order:
 *
 *   bare_runtime_memory.c  bump heap, malloc/free, memory and string helpers
 *   bare_runtime_memref.c  MLIR memrefCopy ABI
 *   bare_runtime_trace.c   buddyTraceCycle* ABI and the cycle summary
 *   bare_runtime_ame.c     ame_fence() and the AME resync sequence
 *   bare_runtime_w8a8.c    lowered W8A8 RVV accumulate helper ABI
 *
 * Every module includes this header, so the compile-time defaults and the
 * enabled/disabled behaviour of the BARE_RUNTIME_* flags are the same whether a
 * flag is passed on the command line or left at its default.  Samples keep
 * including the public "bare_runtime.h"; this header is not part of the ABI.
 */

#include <stddef.h>
#include <stdint.h>

#ifndef BARE_RUNTIME_HEAP_SIZE
#define BARE_RUNTIME_HEAP_SIZE (128u * 1024u)
#endif

#ifndef BARE_RUNTIME_HEAP_ALIGN
#define BARE_RUNTIME_HEAP_ALIGN 64u
#endif

#ifndef BARE_RUNTIME_RECLAIM_FREE
#define BARE_RUNTIME_RECLAIM_FREE 0
#endif

#ifndef BARE_RUNTIME_TRACE_MALLOC
#define BARE_RUNTIME_TRACE_MALLOC 0
#endif

#ifndef BARE_RUNTIME_TRACE_MEMCPY
#define BARE_RUNTIME_TRACE_MEMCPY 0
#endif

#ifndef BARE_RUNTIME_TRACE_MEMREF_COPY
#define BARE_RUNTIME_TRACE_MEMREF_COPY 0
#endif

#ifndef BARE_RUNTIME_ENABLE_MEMREF_COPY
#define BARE_RUNTIME_ENABLE_MEMREF_COPY 1
#endif

#ifndef BARE_RUNTIME_ENABLE_AME
#define BARE_RUNTIME_ENABLE_AME 1
#endif

#ifndef BARE_RUNTIME_TRACE_MAX_ID
#define BARE_RUNTIME_TRACE_MAX_ID 256
#endif

#ifndef BARE_RUNTIME_TRACE_LIVE
#define BARE_RUNTIME_TRACE_LIVE 0
#endif

#ifndef BARE_RUNTIME_MEMREF_COPY_MAX_ELEMS
#define BARE_RUNTIME_MEMREF_COPY_MAX_ELEMS (16u * 1024u * 1024u)
#endif

#ifndef BARE_RUNTIME_MEMREF_COPY_MAX_RANK
#define BARE_RUNTIME_MEMREF_COPY_MAX_RANK 8
#endif

#if BARE_RUNTIME_TRACE_MALLOC || BARE_RUNTIME_TRACE_MEMCPY || \
    BARE_RUNTIME_TRACE_MEMREF_COPY
#include "uart.h"

static inline void bare_trace_hex_u64(uint64_t value)
{
    print_uart("0x");
    print_uart_addr(value);
}
#endif

/*
 * Byte copy shared by the scalar memory helpers and the memrefCopy walker.
 * Keeping it in the private header preserves the original inlining and avoids
 * an exported symbol for a loop that is only an implementation detail.
 */
static inline void bare_byte_copy(char *dst, const char *src, int64_t size)
{
    for (int64_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

#endif
