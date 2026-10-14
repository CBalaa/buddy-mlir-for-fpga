/*
 * matmul_i8_128x1024x128 launch: application startup, E6 driver and NH report.
 *
 * The image runs on the shared NH/E6 platform (runtime/boscame/nh_e6):
 * NH boots at 0x80000000, owns the UART and the verdict; E6 starts at
 * 0x80010000, and `nh_e6_e6_main()` runs the platform's address-space setup
 * (seven TOR PMP regions, `satp = 0`, `sfence.vma`) and the CSR 0xbc0 cache
 * policy *before* `e6_main()`.  That ordering is what makes the first DMA and
 * the first AME access in the kernel legal: the DMA engine reaches the staging
 * buffers through the PMP region bounded by `__nh_e6_ddr_window_start/end`,
 * which link.ld exports and places `.dma_source`/`.dma_output` inside.
 *
 * This file does three things:
 *
 *   e6_main()          fill the operands, call the kernel through the
 *                      generated ABI bridge, then run the scalar INT32 oracle
 *                      and publish the diagnostics.
 *   nh_e6_app_tag()    report tag for the platform prefixes.
 *   nh_e6_app_report() NH-side report, printed on every outcome (PASS, FAIL,
 *                      TRAP, TIMEOUT), so a hang still produces evidence.
 *
 * The driver is datapath-agnostic: `make` (AME=1) links the kernel rewritten
 * by `--triton-to-boscame` and calls its four-descriptor bridge, `make AME=0`
 * links the scalar RISC-V kernel that skips the pass and calls its
 * three-descriptor bridge.  Operands, timing, oracle and report are identical,
 * so `stats cycles=` can be compared between the two images.
 *
 * Progress markers (`nh_e6_progress`) are published at every phase boundary
 * and periodically inside the oracle: NH prints each new marker while it waits,
 * so the last marker on the console says exactly how far the run got.  A DMA
 * that cannot complete reports channel/reason/STATUS/token through the
 * platform's failure hook and traps instead of hanging silently.
 */
#include <stddef.h>
#include <stdint.h>

#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"

/* ------------------------------------------------------------------ */
/* libc shim for the scalar kernel (AME=0 only)                        */
/* ------------------------------------------------------------------ */

#if !MATMUL_I8_USE_AME
/* The scalar kernel is ordinary CPU code: its accumulator staging lowers to
 * two `llvm.intr.memcpy` calls (1 KiB each).  The host build takes those from
 * libc; this freestanding image has to provide them.  The AME=1 kernel never
 * calls memcpy -- the pass emits explicit loops -- so the shim is compiled in
 * only for AME=0.  Word-wise with an unaligned tail, so it cannot turn back
 * into a call to itself. */
typedef uint64_t __attribute__((may_alias)) CopyWord;
void *memcpy(void *destination, const void *source, size_t bytes) {
  unsigned char *d = (unsigned char *)destination;
  const unsigned char *s = (const unsigned char *)source;
  while (bytes >= sizeof(CopyWord) &&
         (((uintptr_t)d | (uintptr_t)s) & (sizeof(CopyWord) - 1u)) == 0u) {
    *(CopyWord *)d = *(const CopyWord *)s;
    d += sizeof(CopyWord);
    s += sizeof(CopyWord);
    bytes -= sizeof(CopyWord);
  }
  while (bytes-- != 0u)
    *d++ = *s++;
  return destination;
}
#endif

/* ------------------------------------------------------------------ */
/* Problem size (the same numbers the TTIR and build.py use)           */
/* ------------------------------------------------------------------ */

#define M 128u
#define N 1024u
#define K 128u
#define A_BYTES ((size_t)M * K)
#define B_BYTES ((size_t)N * K)
#define C_BYTES ((size_t)M * N * 4u)

/* How much of C the on-board oracle re-computes.  Every row of this shape is
 * tile-aligned, so a prefix is exactly the first MATMUL_I8_CHECK_ROWS rows.
 * The full check is 16.7M multiply-adds; `-DMATMUL_I8_CHECK_ROWS=128`
 * (the default) checks all of them, the debug image checks the first 16. */
#ifndef MATMUL_I8_CHECK_ROWS
#define MATMUL_I8_CHECK_ROWS M
#endif
#if MATMUL_I8_CHECK_ROWS > M
#error "MATMUL_I8_CHECK_ROWS must not exceed M"
#endif
#if defined(MATMUL_I8_BOARD_CHECK) && MATMUL_I8_CHECK_ROWS != M
#error "MATMUL_I8_BOARD_CHECK requires a full-output oracle"
#endif

/* ------------------------------------------------------------------ */
/* ABI bridge interface (build.py --adapter ...)                        *
 *                                                                     *
 * `make AME=0` compiles this driver against the scalar kernel that    *
 * skips `--triton-to-boscame`: that bridge takes three descriptors    *
 * and no scratch, because no DMA/SPM is involved.  Everything else    *
 * (operands, timing, oracle, report) is shared, which is what makes   *
 * the two `stats cycles=` comparable.                                 */
/* ------------------------------------------------------------------ */

#ifndef MATMUL_I8_USE_AME
#define MATMUL_I8_USE_AME 1
#endif

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2;

#if MATMUL_I8_USE_AME
extern void _mlir_ciface_kernel_matmul_i8_128x1024x128(MemRef2 *a, MemRef2 *b,
                                                       MemRef2 *c,
                                                       void *scratch);
#else
extern void _mlir_ciface_kernel_matmul_i8_128x1024x128(MemRef2 *a, MemRef2 *b,
                                                       MemRef2 *c);
#endif

static inline MemRef2 make_2(void *pointer, int64_t rows, int64_t columns) {
  MemRef2 descriptor;
  descriptor.allocated = pointer;
  descriptor.aligned = pointer;
  descriptor.offset = 0;
  descriptor.sizes[0] = rows;
  descriptor.sizes[1] = columns;
  descriptor.strides[0] = columns;
  descriptor.strides[1] = 1;
  return descriptor;
}

/* ------------------------------------------------------------------ */
/* DDR staging windows, pinned by link.ld inside the PMP DDR window    */
/* ------------------------------------------------------------------ */

#if defined(MATMUL_I8_BOARD_CHECK)
/* Separate guard words from the payload without weakening its 64B alignment.
 * The public symbols and fixed 1024B payload offsets also let a host oracle
 * inspect a post-execution DDR dump independently of the E6 diagnostics. */
#define GUARD_WORDS 256u
#define GUARD_VALUE UINT32_C(0x6e5a1234)
typedef struct {
  uint32_t before[GUARD_WORDS];
  int8_t elements[A_BYTES];
  uint32_t after[GUARD_WORDS];
} AStorage;
typedef struct {
  uint32_t before[GUARD_WORDS];
  int8_t elements[B_BYTES];
  uint32_t after[GUARD_WORDS];
} BStorage;
typedef struct {
  uint32_t before[GUARD_WORDS];
  uint32_t elements[M * N];
  uint32_t after[GUARD_WORDS];
} CStorage;
AStorage a_storage __attribute__((section(".dma_source"), aligned(4096), used));
BStorage b_storage __attribute__((section(".dma_source"), aligned(4096), used));
CStorage c_storage __attribute__((section(".dma_output"), aligned(4096), used));
#define a_matrix a_storage.elements
#define b_matrix b_storage.elements
#define c_matrix c_storage.elements
#else
static int8_t a_matrix[A_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int8_t b_matrix[B_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int32_t c_matrix[M * N]
    __attribute__((section(".dma_output"), aligned(4096), used));
#endif
/* Legacy launcher scratch facade. The experimental adapter keeps this
 * parameter for compatibility but the new kernel uses aligned stack packs
 * and does not receive scratch. */
#define SCRATCH_BYTES 1024u
static uint8_t scratch[SCRATCH_BYTES]
    __attribute__((section(".dma_source"), aligned(64), used));

/* ------------------------------------------------------------------ */
/* Diagnostics read by nh_e6_app_report()                              */
/* ------------------------------------------------------------------ */

#define DIAG __attribute__((section(".bss.e6_diag")))

volatile uint32_t e6_diag_phase DIAG;
volatile uint32_t e6_diag_errors DIAG;
volatile uint32_t e6_diag_max_abs DIAG;
volatile uint32_t e6_diag_checked_rows DIAG;
volatile uint64_t e6_diag_cycles DIAG;
#if defined(MATMUL_I8_BOARD_CHECK)
volatile uint32_t e6_diag_round DIAG;
volatile uint32_t e6_diag_guard_errors DIAG;
#endif
#if defined(MATMUL_I8_TRACE)
volatile int32_t e6_diag_first_got[8] DIAG;
volatile int32_t e6_diag_first_want[8] DIAG;
#endif

/*
 * Phase boundary: publish the marker through the platform, then re-select the
 * write-through store policy the delivered E6 needs for the staging writes.
 * The PMP regions and `satp = 0` this policy depends on were installed by
 * `nh_e6_e6_main()` before `e6_main()`.
 */
static inline void e6_stage(uint32_t value) {
  e6_diag_phase = value;
  nh_e6_progress(value);
  nh_e6_cache_writes();
}

static inline uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}

/* The kernel accumulates into C (`mqma.b.mm` has += semantics), so C starts at
 * the same deterministic pattern the oracle expects. */
#if defined(MATMUL_I8_BOARD_CHECK)
/* All arithmetic in the generators is bounded. Product accumulation in the
 * oracle uses uint32_t to model the kernel's exact INT32 wrap semantics.
 * B is physically [N,K], including for the independent host oracle. */
static int32_t board_a(unsigned i, unsigned q, unsigned round) {
  return (int32_t)((i * 13u + q * 7u + round * 71u) % 256u) - 128;
}
static int32_t board_b(unsigned j, unsigned q, unsigned round) {
  return (int32_t)((j * 11u + q * 5u + round * 37u) % 256u) - 128;
}
static uint32_t board_c(unsigned i, unsigned j, unsigned round) {
  return UINT32_C(0x7fffff00) + (i * 3u + j + round * 97u) % 512u;
}

static void fill_operands(unsigned round) {
  for (unsigned g = 0; g < GUARD_WORDS; ++g) {
    a_storage.before[g] = a_storage.after[g] = GUARD_VALUE;
    b_storage.before[g] = b_storage.after[g] = GUARD_VALUE;
    c_storage.before[g] = c_storage.after[g] = GUARD_VALUE;
  }
  for (unsigned i = 0; i < M; ++i)
    for (unsigned q = 0; q < K; ++q)
      a_matrix[(size_t)i * K + q] = (int8_t)board_a(i, q, round);
  for (unsigned j = 0; j < N; ++j)
    for (unsigned q = 0; q < K; ++q)
      b_matrix[(size_t)j * K + q] = (int8_t)board_b(j, q, round);
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j)
      c_matrix[(size_t)i * N + j] = board_c(i, j, round);
}

static uint32_t validate_guards_and_inputs(unsigned round) {
  uint32_t errors = 0;
  for (unsigned g = 0; g < GUARD_WORDS; ++g) {
    errors += a_storage.before[g] != GUARD_VALUE;
    errors += a_storage.after[g] != GUARD_VALUE;
    errors += b_storage.before[g] != GUARD_VALUE;
    errors += b_storage.after[g] != GUARD_VALUE;
    errors += c_storage.before[g] != GUARD_VALUE;
    errors += c_storage.after[g] != GUARD_VALUE;
  }
  for (unsigned i = 0; i < M; ++i)
    for (unsigned q = 0; q < K; ++q)
      errors += a_matrix[(size_t)i * K + q] != board_a(i, q, round);
  for (unsigned j = 0; j < N; ++j)
    for (unsigned q = 0; q < K; ++q)
      errors += b_matrix[(size_t)j * K + q] != board_b(j, q, round);
  return errors;
}

static uint32_t validate(unsigned round) {
  uint32_t errors = 0;
  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      uint32_t want = board_c(i, j, round);
      for (unsigned q = 0; q < K; ++q)
        want += (uint32_t)(board_a(i, q, round) * board_b(j, q, round));
      uint32_t got = c_matrix[(size_t)i * N + j];
#if defined(MATMUL_I8_TRACE)
      if (i == 0u && j < 8u) {
        e6_diag_first_got[j] = (int32_t)got;
        e6_diag_first_want[j] = (int32_t)want;
      }
#endif
      errors += got != want;
    }
    ++e6_diag_checked_rows;
    /* Two full grids share the same SPM contents; only DDR operands are
     * reinitialized between them. Distinct markers identify the oracle row. */
    nh_e6_progress(0x2000u + round * M + i);
  }
  return errors;
}
#else
static void fill_operands(void) {
  for (unsigned i = 0; i < M; ++i)
    for (unsigned q = 0; q < K; ++q)
      a_matrix[(size_t)i * K + q] = (int8_t)(((i * 13u + q * 7u) % 31u) - 15);
  for (unsigned j = 0; j < N; ++j)
    for (unsigned q = 0; q < K; ++q)
      b_matrix[(size_t)j * K + q] = (int8_t)(((j * 11u + q * 5u) % 29u) - 14);
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j)
      c_matrix[(size_t)i * N + j] = (int32_t)((i * 3u + j) % 7u) - 3;
}

/* C[i,j] = previous C[i,j] + sum_q A[i,q] * B[j,q] (B is the physical N x K
 * tile, exactly what MQMA.B.MM expects). */
static uint32_t validate(void) {
  uint32_t errors = 0;
  int32_t max_abs = 0;
  unsigned checked = 0;
  for (unsigned i = 0; i < M; ++i) {
    if (i >= MATMUL_I8_CHECK_ROWS && i < M - (M % 16u))
      continue;
    checked++;
    for (unsigned j = 0; j < N; ++j) {
      int32_t want = (int32_t)((i * 3u + j) % 7u) - 3;
      for (unsigned q = 0; q < K; ++q)
        want += (int32_t)a_matrix[(size_t)i * K + q] *
                (int32_t)b_matrix[(size_t)j * K + q];
      int32_t got = c_matrix[(size_t)i * N + j];
#if defined(MATMUL_I8_TRACE)
      if (i == 0u && j < 8u) {
        e6_diag_first_got[j] = got;
        e6_diag_first_want[j] = want;
      }
#endif
      int32_t delta = got - want;
      if (delta < 0)
        delta = -delta;
      if (delta > max_abs)
        max_abs = delta;
      if (got != want)
        errors++;
    }
    /* Keep the platform's progress report moving during the long check. */
    if ((i & 7u) == 0u)
      nh_e6_progress(0x200u + (i >> 3));
    if (errors != 0u)
      break;
  }
  e6_diag_max_abs = (uint32_t)max_abs;
  e6_diag_checked_rows = checked;
  return errors;
}
#endif

/* ------------------------------------------------------------------ */
/* E6 entry                                                            */
/* ------------------------------------------------------------------ */

int e6_main(void) {
  /* A failed transfer records channel/reason/STATUS/token and traps; the E6
   * trap entry publishes 0xed and NH prints the DMA context line. */
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;

#if defined(MATMUL_I8_BOARD_CHECK)
  e6_diag_errors = 0;
  e6_diag_guard_errors = 0;
  e6_diag_checked_rows = 0;
  e6_diag_cycles = 0;
  for (unsigned round = 0; round < 2u; ++round) {
    e6_diag_round = round;
    e6_stage(0x11u + round);
    fill_operands(round);
    __asm__ volatile("fence rw, rw" ::: "memory");
    MemRef2 a = make_2(a_matrix, M, K);
    MemRef2 b = make_2(b_matrix, N, K);
    MemRef2 c = make_2(c_matrix, M, N);
    e6_stage(0x30u + round * 2u);
    uint64_t begin = cycles();
#if MATMUL_I8_USE_AME
    _mlir_ciface_kernel_matmul_i8_128x1024x128(&a, &b, &c, scratch);
#else
    _mlir_ciface_kernel_matmul_i8_128x1024x128(&a, &b, &c);
#endif
    e6_diag_cycles += cycles() - begin;
    __asm__ volatile("fence rw, rw" ::: "memory");
    e6_stage(0x31u + round * 2u);
    nh_e6_cache_uncached_loads();
    e6_diag_guard_errors += validate_guards_and_inputs(round);
    e6_diag_errors += validate(round);
    __asm__ volatile("fence rw, rw" ::: "memory");
  }
  uint32_t errors = e6_diag_errors + e6_diag_guard_errors;
  e6_stage(errors == 0u ? 0x02u : 0xeeu);
  return errors == 0u ? 0 : 1;
#else
  e6_stage(0x11u);
  /* The operands keep their logical shape; every tile is fully valid. */
  fill_operands();
  __asm__ volatile("fence rw, rw" ::: "memory");

  MemRef2 a = make_2(a_matrix, M, K);
  MemRef2 b = make_2(b_matrix, N, K);
  /* B is physically [N,K], so its descriptor has strides [K,1]. The
   * compiler's logical [K,N] AME view is built inside the kernel. */
  MemRef2 c = make_2(c_matrix, M, N);

  e6_stage(0x30u);
  uint64_t begin = cycles();
#if MATMUL_I8_USE_AME
  _mlir_ciface_kernel_matmul_i8_128x1024x128(&a, &b, &c, scratch);
#else
  _mlir_ciface_kernel_matmul_i8_128x1024x128(&a, &b, &c);
#endif
  e6_diag_cycles = cycles() - begin;
  __asm__ volatile("fence rw, rw" ::: "memory");
  e6_stage(0x31u);

  /* Read DMA-updated C with the reference's uncached-load policy. */
  nh_e6_cache_uncached_loads();
  uint32_t errors = validate();
  e6_diag_errors = errors;
  __asm__ volatile("fence rw, rw" ::: "memory");
  e6_stage(errors == 0u ? 0x02u : 0xeeu);

  return errors == 0u ? 0 : 1;
#endif
}

/* ------------------------------------------------------------------ */
/* NH report                                                           */
/* ------------------------------------------------------------------ */

/* The tag says which datapath the image carries, so two console logs can be
 * told apart; the tester selects the image by file name, not by tag. */
#if MATMUL_I8_USE_AME
#if defined(MATMUL_I8_BOARD_CHECK)
const char *nh_e6_app_tag(void) { return "matmul-i8-128x1024x128-boardcheck"; }
#else
const char *nh_e6_app_tag(void) { return "matmul-i8-128x1024x128"; }
#endif
#else
const char *nh_e6_app_tag(void) { return "matmul-i8-128x1024x128-noame"; }
#endif

void nh_e6_app_report(void) {
  nh_e6_print_mhz_prefix();
  print_uart("stats phase=0x");
  print_uart_hex32(e6_diag_phase);
  print_uart(" cycles=0x");
  print_uart_hex64(e6_diag_cycles);
  print_uart(" checked_rows=");
  print_uart_dec32(e6_diag_checked_rows);
  print_uart(" errors=");
  print_uart_dec32(e6_diag_errors);
#if defined(MATMUL_I8_BOARD_CHECK)
  print_uart(" guard_errors=");
  print_uart_dec32(e6_diag_guard_errors);
  print_uart(" round=");
  print_uart_dec32(e6_diag_round);
#endif
  print_uart(" max_abs=");
  print_uart_dec32(e6_diag_max_abs);
  print_uart("\r\n");
#if defined(MATMUL_I8_TRACE)
  for (unsigned j = 0; j < 8u; ++j) {
    nh_e6_print_mhz_prefix();
    print_uart("sample column=");
    print_uart_dec32(j);
    print_uart(" got=0x");
    print_uart_hex32((uint32_t)e6_diag_first_got[j]);
    print_uart(" want=0x");
    print_uart_hex32((uint32_t)e6_diag_first_want[j]);
    print_uart("\r\n");
  }
#endif
}
