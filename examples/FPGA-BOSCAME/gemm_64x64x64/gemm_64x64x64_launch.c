/*
 * gemm_64x64x64_launch.c -- application startup, E6 driver and NH report for
 * the BOSCAME-dialect 64x64x64 GEMM.
 *
 * This is the only application source of the example:
 *
 *   e6_main()          E6 algorithm entry: DDR staging operands, memref
 *                      descriptors, the MLIR kernel call, the scalar oracle
 *                      and the GEMM statistics.
 *   nh_e6_app_tag()    report tag used by the common platform prefixes.
 *   nh_e6_app_report() NH-side report of the GEMM statistics, printed after
 *                      the platform verdict on every outcome.
 *
 * Boot, dual-core release/wait, progress, timeout, the generic PASS/FAIL/TRAP
 * verdict, the E6 address-space setup (PMP regions, `satp=0`) and the CSR 0xbc0
 * cache policies live in ../../../common/nh_e6; the DMA/SPM register access in
 * ../../../common/runtime/bosc_ame_dma.c and the UART device in
 * ../../../common/uart.  Nothing in those modules knows the matrix sizes or
 * the diagnostic fields below, and the platform runs its own hardware setup
 * before `e6_main()` -- including before this driver's first CSR 0xbc0 policy
 * switch and its first DMA operation.
 *
 * `gemm_64x64x64_ame_dma_spm.mlir` is lowered by `buddy-opt --lower-bosc-ame`
 * into `_mlir_ciface_gemm_64x64x64(MemRef1*, MemRef1*, MemRef1*, MemRef1*)`,
 * so the driver only has to build four rank-1 memref descriptors around the
 * DDR staging buffers and call it:
 *
 *     DDR A/B/Z --DMA--> SPM0/SPM1 --AME tile load--> Tr --MQMA--> Acc
 *               --AME store--> SPM0 --DMA--> DDR C
 *
 * `Z` is a permanent 16 KiB zero tile: the kernel stages it into the SPM
 * accumulator window so that the `mlce32` accumulator load starts from zero
 * (the v0.5 contract has no "zero accumulator register" instruction in the
 * validated FPGA subset).
 *
 * The descriptor layout is the LLVM memref descriptor for a rank-1 memref:
 *
 *     struct { void *allocated, *aligned; int64_t offset;
 *              int64_t sizes[1]; int64_t strides[1]; }
 *
 * E6 does not own the physical UART: every measured value is published into
 * `.bss.e6_diag`, which NH reads and prints after the verdict, including after
 * TIMEOUT or TRAP.
 */
#include <stddef.h>
#include <stdint.h>

#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"

/* ------------------------------------------------------------------ */
/* Kernel ABI                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1;

extern int32_t _mlir_ciface_gemm_64x64x64(MemRef1 *a, MemRef1 *b, MemRef1 *z,
                                          MemRef1 *c);

#define M 64u
#define N 64u
#define K 64u
#define A_BYTES (M * K)           /* 4096 bytes, INT8  */
#define B_BYTES (N * K)           /* 4096 bytes, INT8  */
#define C_BYTES (M * N * 4u)      /* 16384 bytes, INT32 */

static inline MemRef1 make_1(void *pointer, int64_t count) {
  MemRef1 descriptor;
  descriptor.allocated = pointer;
  descriptor.aligned = pointer;
  descriptor.offset = 0;
  descriptor.sizes[0] = count;
  descriptor.strides[0] = 1;
  return descriptor;
}

/* ------------------------------------------------------------------ */
/* DDR staging: the link script pins these windows                     */
/* ------------------------------------------------------------------ */

static int8_t a_matrix[A_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int8_t b_matrix[B_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int8_t z_matrix[C_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int32_t c_matrix[M * N]
    __attribute__((section(".dma_output"), aligned(4096), used));

/* ------------------------------------------------------------------ */
/* DDR diagnostic cells read by nh_e6_app_report()                     */
/*                                                                     */
/* The platform owns the trap, DMA-failure and progress cells; these are */
/* the GEMM-specific measurements.  They live in E6-cleared .bss.       */
/* ------------------------------------------------------------------ */

#define DIAG __attribute__((section(".bss.e6_diag")))

volatile uint32_t e6_diag_status DIAG;
volatile uint32_t e6_diag_errors DIAG;
volatile uint32_t e6_diag_max_abs DIAG;
volatile uint64_t e6_diag_cycles DIAG;
volatile uint32_t e6_diag_acc_errors DIAG;
volatile uint32_t e6_diag_acc_max_abs DIAG;
volatile uint32_t e6_diag_raw_m DIAG;
volatile uint32_t e6_diag_raw_n DIAG;
volatile uint32_t e6_diag_raw_k DIAG;
volatile uint32_t e6_diag_max_m DIAG;
volatile uint32_t e6_diag_max_n DIAG;
volatile uint32_t e6_diag_max_k DIAG;
volatile uint32_t e6_diag_readback DIAG;
volatile uint32_t e6_diag_tr_a_bytes DIAG;
volatile uint32_t e6_diag_tr_b_bytes DIAG;
volatile uint32_t e6_diag_acc_bytes DIAG;
volatile uint32_t e6_diag_mlenb DIAG;
volatile uint32_t e6_diag_mrlenb DIAG;
volatile uint32_t e6_diag_malenb DIAG;
volatile uint32_t e6_diag_clamp_m DIAG;
volatile uint32_t e6_diag_clamp_n DIAG;
volatile uint32_t e6_diag_clamp_k DIAG;
volatile uint32_t e6_diag_probe_ok DIAG;

/*
 * Phase boundary: publish the marker through the platform (patch6 mailbox plus
 * post-mortem copy), then re-select the write-through store policy the
 * delivered E6 needs for the next DMA staging writes.  Both steps and their
 * order live in the platform: `nh_e6_e6_main()` already installed the PMP
 * regions and `satp=0` that the CSR 0xbc0 policy depends on.
 */
static inline void e6_stage(uint32_t value) {
  nh_e6_progress(value);
  nh_e6_cache_writes();
}

static inline uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}

/* Deterministic small-range INT8 operands; the oracle below uses the same. */
static void fill_operands(void) {
  for (unsigned i = 0; i < M; ++i)
    for (unsigned k = 0; k < K; ++k)
      a_matrix[i * K + k] = (int8_t)(((i * 3u + k * 5u) % 13u) - 6);
  for (unsigned j = 0; j < N; ++j)
    for (unsigned k = 0; k < K; ++k)
      b_matrix[j * K + k] = (int8_t)(((j * 7u + k * 3u) % 11u) - 5);
  for (unsigned i = 0; i < C_BYTES; ++i)
    z_matrix[i] = 0;
  for (unsigned i = 0; i < M * N; ++i)
    c_matrix[i] = 0;
}

/* O[i, j] = sum_k A[i, k] * B[j, k] (B is N x K, as MQMA.B.MM expects). */
static uint32_t validate(void) {
  uint32_t errors = 0;
  int32_t max_abs = 0;
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j) {
      int32_t want = 0;
      for (unsigned k = 0; k < K; ++k)
        want += (int32_t)a_matrix[i * K + k] * (int32_t)b_matrix[j * K + k];
      int32_t got = c_matrix[i * N + j];
      int32_t delta = got - want;
      if (delta < 0)
        delta = -delta;
      if (delta > max_abs)
        max_abs = delta;
      if (got != want)
        errors++;
    }
  e6_diag_max_abs = (uint32_t)max_abs;
  return errors;
}

/* ------------------------------------------------------------------ */
/* E6 entry                                                            */
/* ------------------------------------------------------------------ */

int e6_main(void) {
  /* nh_e6_e6_main() has already run the platform's PMP/`satp=0` setup, so the
   * CSR 0xbc0 policy switches below and every DMA/AME access are legal. */
  /* A failed transfer records channel/reason/STATUS/token in the platform
   * diagnostic cells and traps; the E6 trap entry publishes 0xed and NH
   * reports the DMA context line. */
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  /* This example runs the fixed 64x64x64 tile only; publish the capacities it
   * assumes so the NH report stays consistent with the other test. */
  e6_diag_max_m = M;
  e6_diag_max_n = N;
  e6_diag_max_k = K;
  e6_diag_tr_a_bytes = A_BYTES;
  e6_diag_tr_b_bytes = B_BYTES;
  e6_diag_acc_bytes = C_BYTES;
  e6_diag_mlenb = A_BYTES + B_BYTES;
  e6_diag_mrlenb = N * 4u;
  e6_diag_malenb = C_BYTES;
  e6_diag_clamp_m = 0;
  e6_diag_clamp_n = 0;
  e6_diag_clamp_k = 0;
  e6_diag_probe_ok = 0;

  e6_stage(0x11);
  fill_operands();
  __asm__ volatile("fence rw, rw" ::: "memory");

  MemRef1 A = make_1(a_matrix, A_BYTES);
  MemRef1 B = make_1(b_matrix, B_BYTES);
  MemRef1 Z = make_1(z_matrix, C_BYTES);
  MemRef1 C = make_1(c_matrix, C_BYTES);

  e6_stage(0x30);
  uint64_t begin = cycles();
  int32_t status = _mlir_ciface_gemm_64x64x64(&A, &B, &Z, &C);
  e6_diag_cycles = cycles() - begin;
  e6_diag_status = (uint32_t)status;
  __asm__ volatile("fence rw, rw" ::: "memory");
  e6_stage(status ? 0xe1u : 0x31u);

  uint32_t errors = UINT32_MAX;
  if (status == 0) {
    /* Read DMA-updated C with the reference's uncached-load policy. */
    nh_e6_cache_uncached_loads();
    errors = validate();
  }
  e6_diag_errors = errors;
  __asm__ volatile("fence rw, rw" ::: "memory");

  e6_stage(errors == 0 ? 2u : 0xeeu);
  return (status != 0 || errors != 0) ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* NH report                                                           */
/* ------------------------------------------------------------------ */

const char *nh_e6_app_tag(void) { return "ame-64x64x64"; }

void nh_e6_app_report(void) {
  nh_e6_print_mhz_prefix();
  print_uart("stats status=0x");
  print_uart_hex32(e6_diag_status);
  print_uart(" cycles=0x");
  print_uart_hex64(e6_diag_cycles);
  print_uart(" errors=");
  print_uart_dec32(e6_diag_errors);
  print_uart(" max_abs=");
  print_uart_dec32(e6_diag_max_abs);
  print_uart(" acc_errors=");
  print_uart_dec32(e6_diag_acc_errors);
  print_uart(" acc_max_abs=");
  print_uart_dec32(e6_diag_acc_max_abs);

  /* ---- assumed tr/acc capacity --------------------------------------- */
  nh_e6_print_tag_prefix();
  print_uart("tr/acc size probe\r\n");
  print_uart("  msettile readback ok=");
  print_uart_dec32(e6_diag_readback);
  print_uart(" raw_m=");
  print_uart_dec32(e6_diag_raw_m);
  print_uart(" raw_n=");
  print_uart_dec32(e6_diag_raw_n);
  print_uart(" raw_k=");
  print_uart_dec32(e6_diag_raw_k);

  print_uart("\r\n  max_m=");
  print_uart_dec32(e6_diag_max_m);
  print_uart(" max_n=");
  print_uart_dec32(e6_diag_max_n);
  print_uart(" max_k=");
  print_uart_dec32(e6_diag_max_k);

  print_uart("\r\n  tr_a_bytes=");
  print_uart_dec32(e6_diag_tr_a_bytes);
  print_uart(" tr_b_bytes=");
  print_uart_dec32(e6_diag_tr_b_bytes);
  print_uart(" acc_bytes=");
  print_uart_dec32(e6_diag_acc_bytes);

  print_uart("\r\n  mlenb=");
  print_uart_dec32(e6_diag_mlenb);
  print_uart(" mrlenb=");
  print_uart_dec32(e6_diag_mrlenb);
  print_uart(" malenb=");
  print_uart_dec32(e6_diag_malenb);

  print_uart("\r\n  clamp_m=");
  print_uart_dec32(e6_diag_clamp_m);
  print_uart(" clamp_n=");
  print_uart_dec32(e6_diag_clamp_n);
  print_uart(" clamp_k=");
  print_uart_dec32(e6_diag_clamp_k);
  print_uart(" probe_ok=");
  print_uart_dec32(e6_diag_probe_ok);
  print_uart("\r\n");
}
