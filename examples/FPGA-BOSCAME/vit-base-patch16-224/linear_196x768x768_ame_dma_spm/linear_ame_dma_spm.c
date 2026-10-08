#include "linear_ame_dma_spm.h"

#ifndef HOST_TEST
#include <stddef.h>

#ifndef VIT_LINEAR_TRACE
#define VIT_LINEAR_TRACE 0
#endif

#if VIT_LINEAR_TRACE
#define VIT_LINEAR_MAILBOX UINT64_C(0x50000160)
static void vit_trace_text(const char *text) {
  while (*text) {
    *(volatile uint32_t *)(uintptr_t)VIT_LINEAR_MAILBOX =
        (unsigned char)*text++;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
  }
}
static void vit_trace_u32(unsigned value) {
  static const char digits[] = "0123456789";
  char text[11];
  unsigned length = 0;
  do {
    text[length++] = digits[value % 10u];
    value /= 10u;
  } while (value != 0u);
  while (length != 0u) {
    char digit = text[--length];
    *(volatile uint32_t *)(uintptr_t)VIT_LINEAR_MAILBOX =
        (unsigned char)digit;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
  }
}
static void vit_trace_panel(unsigned mb, unsigned nb) {
  vit_trace_text("[vit-linear-ame] panel mb=");
  vit_trace_u32(mb);
  vit_trace_text(" nb=");
  vit_trace_u32(nb);
  vit_trace_text("\r\n");
}
#endif

/* patch6 memory map.  The AME and DMA ports consume Sv39 virtual addresses;
 * the page tables below map these windows to the low-DDR staging buffers and
 * to the two physical SPM banks. */
#define DMA0_CTRL UINT64_C(0x60000000)
#define DMA1_CTRL UINT64_C(0x60010000)
#define SPM0_PA UINT64_C(0x20000000)
#define SPM1_PA UINT64_C(0x20400000)
#define SPM_BYTES UINT64_C(0x20000)
#define SPM_PAGE_BYTES UINT64_C(0x1000)
#define SPM_PAGES 32u
#define VSPM0 UINT64_C(0x50000000)
#define VSPM1 UINT64_C(0x50400000)

/* One 2-MiB Sv39 L1 span holds the three low-DDR source ranges. */
#define VA_SRC UINT64_C(0x40000000)
#define VA_A VA_SRC
#define VA_B (VA_SRC + UINT64_C(0x00040000))
#define VA_ZERO (VA_SRC + UINT64_C(0x00140000))
#define VA_OUT UINT64_C(0x40200000)

/*
 * Keep the complete A panel resident in SPM0 while the twelve N panels are
 * consumed.  A 64x64 int8 tile is 4 KiB, so one M panel needs 48 KiB.  The
 * two 16 KiB C buffers then fit below the 128 KiB SPM bank limit and let the
 * DMA write back C[n-2] while AME computes C[n].
 */
#define SPM_A_BASE UINT64_C(0x0000)
#define SPM_C0 UINT64_C(0xc000)
#define SPM_C1 UINT64_C(0x10000)
#define SPM_B0 UINT64_C(0x0000)
#define SPM_B1 UINT64_C(0x1000)

#define AME_MTYPE_INT8 UINT64_C(0x10010)
#define AME_MTYPE_INT32 UINT64_C(0x10042)

#define PTE_V UINT64_C(0x001)
#define PTE_R UINT64_C(0x002)
#define PTE_W UINT64_C(0x004)
#define PTE_A UINT64_C(0x040)
#define PTE_D UINT64_C(0x080)
#define PTE_PTR(pa) ((((uint64_t)(pa)) >> 12) << 10 | PTE_V)
#define PTE_RW(pa) ((((uint64_t)(pa)) >> 12) << 10 | PTE_V | PTE_R | PTE_W | PTE_A | PTE_D)

/* These arrays live in ordinary low DDR BSS.  The E6 page walker reads them
 * through physical addresses, while AME/DMA use the VA windows above. */
static uint64_t root_pt[512] __attribute__((aligned(4096)));
static uint64_t l1_pt[512] __attribute__((aligned(4096)));
static uint64_t l0_src_pt[512] __attribute__((aligned(4096)));
static uint64_t l0_out_pt[512] __attribute__((aligned(4096)));
static uint64_t l0_spm0_pt[512] __attribute__((aligned(4096)));
static uint64_t l0_spm1_pt[512] __attribute__((aligned(4096)));
static uint8_t zero_c[VIT_LINEAR_C_TILE_BYTES]
    __attribute__((aligned(4096)));
static uint8_t page_tables_ready;
static uint8_t pmp_ready;
static uintptr_t mapped_a, mapped_b, mapped_o;

static inline void cpu_fence(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}
static inline void io_fence(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

/* Publish coarse E6 phase markers through the patch6 mailbox.  E6_DONE is a
 * hardware-owned DMA/AME status and must never be used as a software trace
 * register: a write there can be observed as a live engine status word and
 * can make NH mistake an in-flight operation for a terminal result.  The
 * mailbox is the software-owned progress channel polled by NH. */
static inline void linear_stage(unsigned value) {
#ifndef HOST_TEST
  *(volatile uint32_t *)(uintptr_t)UINT64_C(0x50000160) = value;
  io_fence();
#else
  (void)value;
#endif
}

/* Match the patch6 DMA/SPM reference image's TOR PMP aperture.  DMA and AME
 * page walks access the SPM, MMIO and DDR ranges as bus masters; configuring
 * PMP before installing Sv39 makes those accesses legal on E6. */
static void configure_pmp(void) {
  uintptr_t value;
  value = SPM0_PA >> 2;
  __asm__ volatile("csrw pmpaddr0, %0" :: "r"(value) : "memory");
  value = (SPM1_PA + SPM_BYTES) >> 2;
  __asm__ volatile("csrw pmpaddr1, %0" :: "r"(value) : "memory");
  value = DMA0_CTRL >> 2;
  __asm__ volatile("csrw pmpaddr2, %0" :: "r"(value) : "memory");
  value = (DMA1_CTRL + UINT64_C(0x10000)) >> 2;
  __asm__ volatile("csrw pmpaddr3, %0" :: "r"(value) : "memory");
  value = UINT64_C(0x80100000) >> 2;
  __asm__ volatile("csrw pmpaddr4, %0" :: "r"(value) : "memory");
  value = UINT64_C(0x80300000) >> 2;
  __asm__ volatile("csrw pmpaddr5, %0" :: "r"(value) : "memory");
  value = UINT64_C(0x003fffffffffffff);
  __asm__ volatile("csrw pmpaddr6, %0" :: "r"(value) : "memory");
  value = UINT64_C(0x000f0f0f0f0f0f0f);
  __asm__ volatile("csrw pmpcfg0, %0" :: "r"(value) : "memory");
  io_fence();
}

/* The patch6 DMA test uses CSR 0xbc0 to publish CPU writes and observe DMA
 * responses.  Keep the transition next to the MMIO transaction. */
static inline void shared_writes(void) {
  cpu_fence();
  uintptr_t cfg = UINT64_C(0x0f0f0f0f0f0f0f0f);
  __asm__ volatile("csrw 0xbc0, %0" :: "r"(cfg) : "memory");
}
static inline void shared_reads(void) {
  cpu_fence();
  uintptr_t cfg = UINT64_C(0x0f0f000f000f000f);
  __asm__ volatile("csrw 0xbc0, %0" :: "r"(cfg) : "memory");
}

static inline uint64_t make_satp(uint64_t root) {
  return (UINT64_C(8) << 60) | (root >> 12);
}
static inline void install_satp(uint64_t satp) {
  __asm__ volatile("fence rw, rw\n\tcsrw satp, %0\n\tsfence.vma x0, x0"
                   :: "r"(satp) : "memory");
}

static void map_range(uint64_t *l0, uintptr_t va, uintptr_t pa, size_t bytes) {
  const size_t pages = (bytes + 0xfff) >> 12;
  for (size_t i = 0; i < pages; ++i)
    l0[((va >> 12) + i) & 511u] = PTE_RW(pa + i * 0x1000u);
}

static int setup_address_space(const int8_t *a_tiles, const int8_t *b_tiles,
                               const int32_t *o_tiles) {
  const uintptr_t a_pa = (uintptr_t)a_tiles;
  const uintptr_t b_pa = (uintptr_t)b_tiles;
  const uintptr_t o_pa = (uintptr_t)o_tiles;
  if ((a_pa | b_pa | o_pa) & 0xfffu) return -10;

  /* M-mode CPU accesses remain physical after SATP is installed.  The DMA
   * and AME walkers use the mappings below. */
  shared_writes();
  for (unsigned i = 0; i < 512; ++i) {
    root_pt[i] = 0;
    l1_pt[i] = 0;
    l0_src_pt[i] = 0;
    l0_out_pt[i] = 0;
    l0_spm0_pt[i] = 0;
    l0_spm1_pt[i] = 0;
  }
  root_pt[1] = PTE_PTR(l1_pt);
  l1_pt[0] = PTE_PTR(l0_src_pt);
  l1_pt[1] = PTE_PTR(l0_out_pt);
  l1_pt[128] = PTE_PTR(l0_spm0_pt);
  l1_pt[130] = PTE_PTR(l0_spm1_pt);
  map_range(l0_src_pt, VA_A, a_pa, VIT_LINEAR_A_BYTES);
  map_range(l0_src_pt, VA_B, b_pa, VIT_LINEAR_B_BYTES);
  map_range(l0_src_pt, VA_ZERO, (uintptr_t)zero_c,
            VIT_LINEAR_C_TILE_BYTES);
  map_range(l0_out_pt, VA_OUT, o_pa, VIT_LINEAR_O_BYTES);
  for (unsigned i = 0; i < SPM_PAGES; ++i) {
    l0_spm0_pt[i] = PTE_RW(SPM0_PA + i * SPM_PAGE_BYTES);
    l0_spm1_pt[i] = PTE_RW(SPM1_PA + i * SPM_PAGE_BYTES);
  }
  /* Publish the freshly written page-table leaves before either DMA or AME
   * performs its first translation. */
  shared_writes();
  install_satp(make_satp((uintptr_t)root_pt));
  page_tables_ready = 1;
  mapped_a = a_pa;
  mapped_b = b_pa;
  mapped_o = o_pa;
  return 0;
}

static uintptr_t dma_last_src[2];

static inline volatile uint64_t *dma_regs(unsigned engine) {
  return (volatile uint64_t *)(uintptr_t)(engine ? DMA1_CTRL : DMA0_CTRL);
}

static int dma_start(unsigned engine, uintptr_t src_va, uintptr_t dst_va,
                     size_t bytes) {
  /* Every tile is 64-byte aligned and a whole number of AXI beats. */
  if (engine > 1 || (src_va | dst_va | bytes) & 63u) return -11;
  volatile uint64_t *r = dma_regs(engine);
  dma_last_src[engine] = src_va;
  shared_writes();
  r[0] = src_va;
  r[1] = dst_va;
  r[2] = bytes;
  io_fence();
  r[3] = 1;
  io_fence();
  return 0;
}

static int dma_wait(unsigned engine) {
  if (engine > 1) return -15;
  volatile uint64_t *r = dma_regs(engine);
  /* STATUS is live busy/error.  The patch6 reference runtime rewrites SRC
   * before reading STATUS: the configuration write is accepted only after the
   * descriptor drains, making its bus response the completion barrier for a
   * posted START.  Do not issue a second START or repeatedly rewrite a live
   * descriptor while it is active. */
  shared_writes();
  r[0] = dma_last_src[engine];
  io_fence();
  shared_reads();
  uint64_t status = r[4];
  io_fence();
  return status ? -12 : 0;
}

/* Direct patch6 AME encodings.  The fixed a0/a1 operands are the encoding
 * used by the board's standalone reference tests; keeping them direct avoids
 * the stack save/restore sequence emitted by the generic fixed-register
 * adapter around every tile instruction. */
static inline void ame_msettype(uintptr_t value) {
  register uintptr_t a0 __asm__("a0") = value;
  __asm__ volatile("fence rw, rw\n\t.word 0x00054077\n\tfence rw, rw"
                   : "+r"(a0)
                   :
                   : "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                     "t0", "t1", "t2", "t3", "t4", "t5", "t6",
                     "memory");
}
static inline void ame_msettilem(uintptr_t value) {
  register uintptr_t a0 __asm__("a0") = value;
  __asm__ volatile("fence rw, rw\n\t.word 0x04055077\n\tfence rw, rw"
                   : "+r"(a0) :: "memory");
}
static inline void ame_msettilen(uintptr_t value) {
  register uintptr_t a0 __asm__("a0") = value;
  __asm__ volatile("fence rw, rw\n\t.word 0x04054077\n\tfence rw, rw"
                   : "+r"(a0) :: "memory");
}
static inline void ame_msettilek(uintptr_t value) {
  register uintptr_t a0 __asm__("a0") = value;
  __asm__ volatile("fence rw, rw\n\t.word 0x04056077\n\tfence rw, rw"
                   : "+r"(a0) :: "memory");
}
static inline void ame_mlae8(uintptr_t address, uintptr_t stride) {
  register uintptr_t a0 __asm__("a0") = address;
  register uintptr_t a1 __asm__("a1") = stride;
  __asm__ volatile("fence rw, rw\n\t.word 0x04b50077\n\tfence rw, rw"
                   : "+r"(a0), "+r"(a1) :: "memory");
}
static inline void ame_mlbe8(uintptr_t address, uintptr_t stride) {
  register uintptr_t a0 __asm__("a0") = address;
  register uintptr_t a1 __asm__("a1") = stride;
  __asm__ volatile("fence rw, rw\n\t.word 0x08b500f7\n\tfence rw, rw"
                   : "+r"(a0), "+r"(a1) :: "memory");
}
static inline void ame_mlce32(uintptr_t address, uintptr_t stride) {
  register uintptr_t a0 __asm__("a0") = address;
  register uintptr_t a1 __asm__("a1") = stride;
  __asm__ volatile("fence rw, rw\n\t.word 0x00b52077\n\tfence rw, rw"
                   : "+r"(a0), "+r"(a1) :: "memory");
}
static inline void ame_mqma(void) {
  __asm__ volatile("fence rw, rw\n\t.word 0x28180877\n\tfence rw, rw"
                   ::: "memory");
}
static inline void ame_msce32(uintptr_t address, uintptr_t stride) {
  register uintptr_t a0 __asm__("a0") = address;
  register uintptr_t a1 __asm__("a1") = stride;
  __asm__ volatile("fence rw, rw\n\t.word 0x02b52077\n\tfence rw, rw"
                   : "+r"(a0), "+r"(a1) :: "memory");
}

static inline uintptr_t va_a_tile(unsigned index) {
  return VA_A + (uintptr_t)index * VIT_LINEAR_A_TILE_BYTES;
}
static inline uintptr_t va_b_tile(unsigned index) {
  return VA_B + (uintptr_t)index * VIT_LINEAR_B_TILE_BYTES;
}
static inline uintptr_t va_o_tile(unsigned index) {
  return VA_OUT + (uintptr_t)index * VIT_LINEAR_C_TILE_BYTES;
}

/* Fill one resident A panel.  DMA0 is deliberately used only for this panel
 * while the GEMM for the panel is running; output writeback starts after the
 * last A transfer, so the A bytes stay untouched in SPM0 for all twelve N
 * panels.  DMA1 can therefore prefetch B[0,0] in parallel with this loop. */
static int stage_a_panel(unsigned mb, unsigned tile_bytes) {
  linear_stage(0x500u + mb);
  if (dma_start(0, va_a_tile(mb * VIT_LINEAR_KB),
                VSPM0 + SPM_A_BASE, tile_bytes))
    return -30;
  for (unsigned kb = 0; kb < VIT_LINEAR_KB; ++kb) {
    linear_stage(0x510u + kb);
    int status = dma_wait(0);
    if (status) return -31 - status;
    if (kb + 1 < VIT_LINEAR_KB &&
        dma_start(0, va_a_tile(mb * VIT_LINEAR_KB + kb + 1),
                  VSPM0 + SPM_A_BASE +
                      (uintptr_t)(kb + 1) * tile_bytes, tile_bytes))
      return -32;
  }
  linear_stage(0x520u + mb);
  return 0;
}

static int run_ame_tiles(const int8_t *a_tiles, const int8_t *b_tiles,
                         int32_t *o_tiles) {
  linear_stage(0x300u);
  if (!pmp_ready) {
    configure_pmp();
    pmp_ready = 1;
    linear_stage(0x301u);
#if VIT_LINEAR_TRACE
    vit_trace_text("[vit-linear-ame] pmp done\r\n");
#endif
  }
  if (!page_tables_ready || mapped_a != (uintptr_t)a_tiles ||
      mapped_b != (uintptr_t)b_tiles || mapped_o != (uintptr_t)o_tiles) {
    linear_stage(0x310u);
    int status = setup_address_space(a_tiles, b_tiles, o_tiles);
    if (status) return status;
    linear_stage(0x311u);
#if VIT_LINEAR_TRACE
    vit_trace_text("[vit-linear-ame] satp done\r\n");
#endif
  } else {
    /* The caller may reuse the same staging buffers for another invocation
     * (the FP32 wrapper does this).  Publish those CPU writes even though the
     * virtual mappings themselves do not need to be rebuilt. */
    shared_writes();
  }
  const uintptr_t zero_va = VA_ZERO;
  const unsigned tile_bytes = VIT_LINEAR_A_TILE_BYTES;
  const unsigned c_bytes = VIT_LINEAR_C_TILE_BYTES;
  const uintptr_t c_stride = VIT_LINEAR_TILE * sizeof(int32_t);
  int dma0_output_busy = 0;

  for (unsigned mb = 0; mb < VIT_LINEAR_MB; ++mb) {
    linear_stage(0x400u + mb);
    /* The last M panel is zero-padded by the packer.  Keep the hardware tile
     * at 64 rows so msce32 writes the padded rows as well; otherwise rows
     * 4..63 of the final panel would retain stale data in the ping-pong C
     * buffers and the tile-major output would fail its zero-padding oracle. */
    const unsigned m = VIT_LINEAR_TILE;
#if VIT_LINEAR_TRACE
    vit_trace_text("[vit-linear-ame] mb begin ");
    vit_trace_u32(mb);
    vit_trace_text("\r\n");
#endif
    /* The first B tile travels on DMA1 while DMA0 fills the resident A panel. */
    linear_stage(0x410u + mb);
    if (dma_start(1, va_b_tile(0), VSPM1 + SPM_B0, tile_bytes)) return -33;
    int status = stage_a_panel(mb, tile_bytes);
    if (status) return status;
    linear_stage(0x420u + mb);

    ame_msettilem(m);
    ame_msettilen(VIT_LINEAR_TILE);

    for (unsigned nb = 0; nb < VIT_LINEAR_NB; ++nb) {
      linear_stage(0x430u + nb);
#if VIT_LINEAR_TRACE
      vit_trace_panel(mb, nb);
#endif
      const unsigned c_slot = nb & 1u;
      const uintptr_t c_va = VSPM0 + (c_slot ? SPM_C1 : SPM_C0);

      /* zero_c is a permanent, aligned 16 KiB zero tile in DDR.  Loading C
       * through AME avoids a DMA0 round trip for every output tile, leaving
       * DMA0 free to write back the preceding tile while this one computes. */
      ame_msettype(AME_MTYPE_INT32);
      ame_mlce32(zero_va, c_stride);
      ame_msettype(AME_MTYPE_INT8);

      unsigned current = 0; /* B[nb,0] was prefetched into B0. */
      for (unsigned kb = 0; kb < VIT_LINEAR_KB; ++kb) {
        linear_stage(0x440u + kb);
        status = dma_wait(1);
        if (status) return -40 - status;
        /* Reissue K after the i32->i8 type transition for every depth tile.
         * This is the ordering used by the validated Qwen/NR kernels and
         * avoids relying on msettype or mlce32 preserving tile-K state. */
        ame_msettilek(VIT_LINEAR_TILE);
        const uintptr_t a_spm = VSPM0 + SPM_A_BASE +
                                (uintptr_t)kb * tile_bytes;
        const uintptr_t b_spm = VSPM1 + (current ? SPM_B1 : SPM_B0);

        /* Issue the next B transfer before MQMA.  The alternate SPM1 bank is
         * not read by this MQMA, so DMA1 and AME operate concurrently.  At the
         * final K tile, prefetch the next N panel's first tile into B0; K=12
         * is even, hence the current final bank is always B1. */
        if (kb + 1 < VIT_LINEAR_KB) {
          const unsigned next = current ^ 1u;
          if (dma_start(1, va_b_tile(nb * VIT_LINEAR_KB + kb + 1),
                        VSPM1 + (next ? SPM_B1 : SPM_B0), tile_bytes))
            return -41;
          current = next;
        } else if (nb + 1 < VIT_LINEAR_NB &&
                   dma_start(1, va_b_tile((nb + 1) * VIT_LINEAR_KB),
                             VSPM1 + SPM_B0, tile_bytes)) {
          return -42;
        }

        ame_mlae8(a_spm, VIT_LINEAR_TILE);
        ame_mlbe8(b_spm, VIT_LINEAR_TILE);
        ame_mqma();
      }

      ame_msettype(AME_MTYPE_INT32);
      ame_msce32(c_va, c_stride);
      linear_stage(0x450u + nb);

      /* DMA0 is single-descriptor.  Wait for the previous output only after
       * this tile's AME work has overlapped it, then launch this writeback.
       * Since launches are serialized, the alternating C buffers are free by
       * the time their slot is selected again. */
      if (dma0_output_busy) {
        status = dma_wait(0);
        if (status) return -50 - status;
      }
      if (dma_start(0, c_va, va_o_tile(mb * VIT_LINEAR_NB + nb), c_bytes))
        return -51;
      dma0_output_busy = 1;
    }

    /* DMA0 must be idle before its descriptor is reused to stage the next A
     * panel.  The wait also makes the final AME->DMA->CPU ownership handoff
     * explicit for this M panel. */
    if (dma0_output_busy) {
      linear_stage(0x460u + mb);
      status = dma_wait(0);
      if (status) return -60 - status;
      dma0_output_busy = 0;
    }
  }
  shared_reads();
  cpu_fence();
  return 0;
}

#else /* HOST_TEST */

static int run_ame_tiles(const int8_t *a_tiles, const int8_t *b_tiles,
                         int32_t *o_tiles) {
  for (unsigned mb = 0; mb < VIT_LINEAR_MB; ++mb)
    for (unsigned nb = 0; nb < VIT_LINEAR_NB; ++nb)
      for (unsigned mi = 0; mi < VIT_LINEAR_TILE; ++mi)
        for (unsigned nj = 0; nj < VIT_LINEAR_TILE; ++nj) {
          int32_t acc = 0;
          const unsigned row = mb * VIT_LINEAR_TILE + mi;
          const unsigned col = nb * VIT_LINEAR_TILE + nj;
          if (row < VIT_LINEAR_M && col < VIT_LINEAR_N)
            for (unsigned kb = 0; kb < VIT_LINEAR_KB; ++kb)
              for (unsigned ki = 0; ki < VIT_LINEAR_TILE; ++ki)
                acc += (int32_t)a_tiles[(mb * VIT_LINEAR_KB + kb) *
                                        VIT_LINEAR_A_TILE_BYTES + mi * 64 + ki] *
                       (int32_t)b_tiles[(nb * VIT_LINEAR_KB + kb) *
                                        VIT_LINEAR_B_TILE_BYTES + nj * 64 + ki];
          o_tiles[(mb * VIT_LINEAR_NB + nb) * (VIT_LINEAR_TILE * VIT_LINEAR_TILE) +
                  mi * 64 + nj] = acc;
        }
  return 0;
}

#endif

int vit_linear_ame_i8(const int8_t *a_tiles, const int8_t *b_tiles,
                      int32_t *o_tiles) {
  if (!a_tiles || !b_tiles || !o_tiles) return -1;
  return run_ame_tiles(a_tiles, b_tiles, o_tiles);
}

/* The FP32 wrapper owns aligned staging so callers (including a graph adapter)
 * need not know the DMA VA/page-table contract. */
static int8_t q_a[VIT_LINEAR_A_BYTES] __attribute__((aligned(4096)));
static int8_t q_b[VIT_LINEAR_B_BYTES] __attribute__((aligned(4096)));
static int32_t q_o[VIT_LINEAR_O_TILES * VIT_LINEAR_TILE * VIT_LINEAR_TILE]
    __attribute__((aligned(4096)));
static float a_scale[VIT_LINEAR_M] __attribute__((aligned(64)));
static float b_scale[VIT_LINEAR_N] __attribute__((aligned(64)));

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static int8_t quantize(float value, float scale) {
  float scaled = value / scale;
  float rounded = scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
  if (rounded > 127.0f) rounded = 127.0f;
  if (rounded < -127.0f) rounded = -127.0f;
  return (int8_t)(int32_t)rounded;
}

static void quantize_f32(const float *a, const float *b) {
  for (unsigned row = 0; row < VIT_LINEAR_M; ++row) {
    float maximum = 0.0f;
    for (unsigned k = 0; k < VIT_LINEAR_K; ++k) {
      float value = abs_f32(a[row * VIT_LINEAR_K + k]);
      if (value > maximum) maximum = value;
    }
    a_scale[row] = maximum == 0.0f ? 1.0f : maximum / 127.0f;
  }
  for (unsigned col = 0; col < VIT_LINEAR_N; ++col) {
    float maximum = 0.0f;
    for (unsigned k = 0; k < VIT_LINEAR_K; ++k) {
      float value = abs_f32(b[col * VIT_LINEAR_K + k]);
      if (value > maximum) maximum = value;
    }
    b_scale[col] = maximum == 0.0f ? 1.0f : maximum / 127.0f;
  }
  for (unsigned mb = 0; mb < VIT_LINEAR_MB; ++mb)
    for (unsigned kb = 0; kb < VIT_LINEAR_KB; ++kb)
      for (unsigned mi = 0; mi < VIT_LINEAR_TILE; ++mi)
        for (unsigned ki = 0; ki < VIT_LINEAR_TILE; ++ki) {
          unsigned row = mb * VIT_LINEAR_TILE + mi;
          unsigned k = kb * VIT_LINEAR_TILE + ki;
          q_a[(mb * VIT_LINEAR_KB + kb) * VIT_LINEAR_A_TILE_BYTES + mi * 64 + ki] =
              row < VIT_LINEAR_M ? quantize(a[row * VIT_LINEAR_K + k], a_scale[row]) : 0;
        }
  for (unsigned nb = 0; nb < VIT_LINEAR_NB; ++nb)
    for (unsigned kb = 0; kb < VIT_LINEAR_KB; ++kb)
      for (unsigned nj = 0; nj < VIT_LINEAR_TILE; ++nj)
        for (unsigned ki = 0; ki < VIT_LINEAR_TILE; ++ki) {
          unsigned col = nb * VIT_LINEAR_TILE + nj;
          unsigned k = kb * VIT_LINEAR_TILE + ki;
          q_b[(nb * VIT_LINEAR_KB + kb) * VIT_LINEAR_B_TILE_BYTES + nj * 64 + ki] =
              quantize(b[col * VIT_LINEAR_K + k], b_scale[col]);
        }
}

int vit_linear_ame_f32(const float *a, const float *b, const float *bias,
                       float *out) {
  if (!a || !b || !out) return -1;
  quantize_f32(a, b);
  int status = vit_linear_ame_i8(q_a, q_b, q_o);
  if (status) return status;
  for (unsigned mb = 0; mb < VIT_LINEAR_MB; ++mb)
    for (unsigned nb = 0; nb < VIT_LINEAR_NB; ++nb)
      for (unsigned mi = 0; mi < VIT_LINEAR_TILE; ++mi)
        for (unsigned nj = 0; nj < VIT_LINEAR_TILE; ++nj) {
          unsigned row = mb * VIT_LINEAR_TILE + mi;
          unsigned col = nb * VIT_LINEAR_TILE + nj;
          if (row < VIT_LINEAR_M && col < VIT_LINEAR_N) {
            int32_t acc = q_o[(mb * VIT_LINEAR_NB + nb) * 4096 + mi * 64 + nj];
            float value = (float)acc * a_scale[row] * b_scale[col];
            out[row * VIT_LINEAR_N + col] = value + (bias ? bias[col] : 0.0f);
          }
        }
  return 0;
}
