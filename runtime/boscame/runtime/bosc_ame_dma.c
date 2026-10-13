/*
 * BOSCAME SPM/DMA runtime helper implementation.
 *
 * See bosc_ame_dma.h for the hardware contract and the ABI documentation.
 * This file is additionally designed to be included directly by the host test
 * after it has overridden BOSC_AME_DMA0_BASE / BOSC_AME_DMA1_BASE, so it must
 * not define any name that could collide with the test translation unit.
 */

#include "bosc_ame_dma.h"

/* ------------------------------------------------------------------ */
/* MMIO order barrier                                                 */
/* ------------------------------------------------------------------ */

/*
 * Order barrier for MMIO.  On the BOSCAME RISC-V target this is the same
 * "fence iorw, iorw" used by the board reference runtime.  Everywhere else
 * (host tests) it lowers to a sequentially-consistent atomic fence.
 */
#ifndef BOSC_AME_DMA_FENCE
#if defined(__riscv)
#define BOSC_AME_DMA_FENCE() __asm__ volatile("fence iorw, iorw" ::: "memory")
#else
#define BOSC_AME_DMA_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif
#endif

/* ------------------------------------------------------------------ */
/* Delivered E6 cache policy (CSR 0xbc0)                              */
/* ------------------------------------------------------------------ */

/*
 * The delivered patch7 DMA/SPM reference uses CSR 0xbc0 to select write-through
 * stores and uncached loads. Follow its `shared_writes` before DDR stores or
 * programming SRC/DST/LEN/START and `shared_reads` before DDR loads or STATUS.
 * The read policy left by dma_wait must be switched back before subsequent
 * CPU DDR stores; a read/modify/write needs both transitions. Drain prior
 * accesses before each policy change.
 * The caller must first configure the platform's PMP regions as that reference
 * does; these CSR values must not be used with unspecified region boundaries.
 * See test/src/cases/nh_e6_dma_spm/src/main.c in the hardware delivery.
 *
 * The host test has no such CSR, so the transition compiles to nothing there;
 * the transfer ordering under test is the fence pair.
 */
#ifndef BOSC_AME_DMA_SHARED_WRITES
#if defined(__riscv)
#define BOSC_AME_DMA_SHARED_WRITES()                                           \
  __asm__ volatile(                                                            \
      "fence iorw, iorw\n\tcsrw 0xbc0, %0" ::"r"(UINT64_C(0x0f0f0f0f0f0f0f0f)) \
      : "memory")
#define BOSC_AME_DMA_SHARED_READS()                                            \
  __asm__ volatile(                                                            \
      "fence iorw, iorw\n\tcsrw 0xbc0, %0" ::"r"(UINT64_C(0x0f0f000f000f000f)) \
      : "memory")
#else
#define BOSC_AME_DMA_SHARED_WRITES() ((void)0)
#define BOSC_AME_DMA_SHARED_READS() ((void)0)
#endif
#endif

/*
 * Effective source address of the outstanding launch per channel, or 0.
 *
 * SRC is a write-only MMIO register (patch7 section 2.5), so the token check
 * in `bosc_ame_dma_wait` cannot read it back.  The address is tracked in
 * software instead, exactly like the board reference runtime's `dma_last_src`.
 */
static uint64_t bosc_ame_dma_last_src[BOSC_AME_DMA_CHANNELS];

/* ------------------------------------------------------------------ */
/* Register access                                                    */
/* ------------------------------------------------------------------ */

/* Base address of one DMA channel's register file. */
static inline uintptr_t bosc_ame_dma_base(uint32_t channel) {
  return (channel == 1u) ? (uintptr_t)BOSC_AME_DMA1_BASE
                         : (uintptr_t)BOSC_AME_DMA0_BASE;
}

/*
 * Volatile 64-bit MMIO register pointer.  The pointer is always formed as
 * (volatile uint64_t *)(base + byte_offset) so that overriding the base with
 * the address of a fake register array is enough to redirect every access.
 */
static inline volatile uint64_t *bosc_ame_dma_reg(uint32_t channel,
                                                  uintptr_t byte_offset) {
  return (volatile uint64_t *)(bosc_ame_dma_base(channel) + byte_offset);
}

/* Tests can observe register transactions without making write-only SRC
 * readable. */
#ifndef BOSC_AME_DMA_MMIO_READ64
#define BOSC_AME_DMA_MMIO_READ64(channel, offset)                              \
  (*bosc_ame_dma_reg((channel), (offset)))
#endif
#ifndef BOSC_AME_DMA_MMIO_WRITE64
#define BOSC_AME_DMA_MMIO_WRITE64(channel, offset, value)                      \
  (*bosc_ame_dma_reg((channel), (offset)) = (value))
#endif

static inline int bosc_ame_dma_valid_channel(uint32_t channel) {
  return channel < BOSC_AME_DMA_CHANNELS;
}

/* Both endpoints and the transfer length must be 64-byte aligned/multiples. */
static inline int bosc_ame_dma_valid_transfer(uint64_t src, uint64_t dst,
                                              uint64_t len) {
  const uint64_t align = (uint64_t)BOSC_AME_DMA_ALIGNMENT;
  if (src == BOSC_AME_DMA_INVALID_TOKEN || dst == 0u)
    return 0;
  if (len == 0u)
    return 0;
  if ((len % align) != 0u)
    return 0;
  if (((src | dst) & (align - 1u)) != 0u)
    return 0;
  return 1;
}

/* ------------------------------------------------------------------ */
/* Failure hook                                                       */
/* ------------------------------------------------------------------ */

/*
 * Shipped failure hook: trap.  On the board this reaches the E6 trap entry of
 * the platform startup module (common/nh_e6/nh_e6_boot.S), which captures the
 * exception CSRs, publishes 0xed and stops E6; on the host it is the
 * compiler's trap.  Override bosc_ame_dma_fail_hook before the first
 * bosc_ame_dma_wait call to publish channel/status diagnostics first --
 * common/nh_e6's nh_e6_publish_dma_failure is that override for the NH/E6
 * examples.
 *
 * The hook must not return.  Every call below is followed by
 * __builtin_unreachable() so that a violating hook is undefined behaviour
 * rather than a silent continuation with stale data.
 */
static void bosc_ame_dma_default_fail(uint32_t channel, uint32_t reason,
                                      uint64_t status, uint64_t src_token) {
  (void)channel;
  (void)reason;
  (void)status;
  (void)src_token;
  __builtin_trap();
}

bosc_ame_dma_fail_fn bosc_ame_dma_fail_hook = bosc_ame_dma_default_fail;

/* ------------------------------------------------------------------ */
/* Public ABI                                                         */
/* ------------------------------------------------------------------ */

void bosc_ame_ddr_read_begin(void) { BOSC_AME_DMA_SHARED_READS(); }

void bosc_ame_ddr_write_begin(void) { BOSC_AME_DMA_SHARED_WRITES(); }

uint64_t bosc_ame_dma_start(uint32_t channel, uint64_t src, uint64_t dst,
                            uint64_t len) {
  /*
   * Invalid arguments are a programming error: trap in debug builds (the
   * default BOSC_AME_DMA_ASSERT), and with NDEBUG report the failure through
   * the defined BOSC_AME_DMA_INVALID_TOKEN return without touching any
   * register, so a malformed call can never launch a transfer or corrupt the
   * channel configuration.
   */
  if (!bosc_ame_dma_valid_channel(channel) ||
      !bosc_ame_dma_valid_transfer(src, dst, len)) {
    BOSC_AME_DMA_ASSERT(0 &&
                        "bosc_ame_dma_start: bad channel/length/alignment");
    return BOSC_AME_DMA_INVALID_TOKEN;
  }

  /* Publish the CPU-side buffers before handing the channel to the engine. */
  BOSC_AME_DMA_SHARED_WRITES();

  /* Config writes in the required order: SRC, DST, LEN. */
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_SRC_OFFSET, src);
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_DST_OFFSET, dst);
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_LEN_OFFSET, len);

  /*
   * Fence #1, after the config writes and before START: makes SRC/DST/LEN
   * visible to the DMA engine before the launch store is observed.  Matches
   * the board reference runtime (linear_ame_dma_spm.c dma_start).
   */
  BOSC_AME_DMA_FENCE();

  /* START is the last store; it launches the transfer and clears bus_error. */
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_START_OFFSET,
                            BOSC_AME_DMA_START_COMMAND);

  /*
   * Fence #2, after START: keeps the launch store ordered ahead of any later
   * MMIO access in program order, again matching the board reference runtime.
   */
  BOSC_AME_DMA_FENCE();

  /* The lowered IR carries the effective source address as the token. */
  bosc_ame_dma_last_src[channel] = src;
  return src;
}

void bosc_ame_dma_wait(uint32_t channel, uint64_t src_token) {
  if (!bosc_ame_dma_valid_channel(channel)) {
    bosc_ame_dma_fail_hook(channel, BOSC_AME_DMA_FAIL_BAD_CHANNEL, 0u,
                           src_token);
    __builtin_unreachable();
  }

  /*
   * Token 0 is bosc_ame_dma_start's "not launched" sentinel: the request was
   * rejected before it programmed any register, so there is nothing to
   * complete.  A channel without an outstanding launch likewise has no
   * completion barrier to perform.  Reject both cases before MMIO.
   */
  if (src_token == BOSC_AME_DMA_INVALID_TOKEN ||
      bosc_ame_dma_last_src[channel] == BOSC_AME_DMA_INVALID_TOKEN) {
    bosc_ame_dma_fail_hook(channel, BOSC_AME_DMA_FAIL_INVALID_TOKEN, 0u,
                           src_token);
    __builtin_unreachable();
  }

  /*
   * Token validity: the token carried by the lowered IR is the effective SRC
   * of the request being completed.  SRC is write-only, so the expected value
   * comes from the software record written by `bosc_ame_dma_start`.  The check
   * only records the bug; the write-back below is always performed because it
   * is the completion barrier, and skipping it would leave an active transfer
   * behind.
   */
  const uint64_t launched_src = bosc_ame_dma_last_src[channel];
  const int token_mismatch = (launched_src != src_token);

  /*
   * SRC write-back: the completion barrier required by the hardware contract.
   * The engine only accepts configuration writes while it is idle, so this
   * store cannot be observed to complete before the transfer it completes.
   * It only re-arms the source address; only a START store launches a
   * transfer, so the write-back never restarts anything.
   */
  BOSC_AME_DMA_SHARED_WRITES();
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_SRC_OFFSET, launched_src);

  /*
   * Order barrier: wait for the configuration write transaction (and the
   * request it retires) and order it ahead of the STATUS observation that
   * follows.  busy is live state, so this version deliberately performs a
   * single STATUS read instead of polling it.
   */
  BOSC_AME_DMA_FENCE();

  /* STATUS is a read transaction: take the read side of the transition. */
  BOSC_AME_DMA_SHARED_READS();
  uint64_t status =
      BOSC_AME_DMA_MMIO_READ64(channel, BOSC_AME_DMA_STATUS_OFFSET);

  /* Order the STATUS read before any later access to the transferred buffer. */
  BOSC_AME_DMA_FENCE();

  if ((status & BOSC_AME_DMA_STATUS_BUSY) == 0u)
    bosc_ame_dma_last_src[channel] = BOSC_AME_DMA_INVALID_TOKEN;

  /*
   * The checks below only use the defined STATUS bits.  `bus_error` means the
   * transferred data is invalid; a `busy` that is still visible after the
   * completion barrier means the completion guarantee did not hold.  Both are
   * failures even though they are reported with different reasons.
   */
  if (token_mismatch) {
    bosc_ame_dma_fail_hook(channel, BOSC_AME_DMA_FAIL_TOKEN_MISMATCH, status,
                           src_token);
    __builtin_unreachable();
  }
  if ((status & (BOSC_AME_DMA_STATUS_BUSY | BOSC_AME_DMA_STATUS_BUS_ERROR)) !=
      0u) {
    uint32_t reason = (status & BOSC_AME_DMA_STATUS_BUS_ERROR) != 0u
                          ? BOSC_AME_DMA_FAIL_BUS_ERROR
                          : BOSC_AME_DMA_FAIL_BUSY;
    bosc_ame_dma_fail_hook(channel, reason, status, src_token);
    __builtin_unreachable();
  }
}
