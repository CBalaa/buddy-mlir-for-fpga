OUTPUT_ARCH("riscv")
ENTRY(_start)

/* NH/E6 image layout for the INT8 matmul.  Boot, the dual-core release and
 * the platform verdict live in runtime/boscame/nh_e6; this script only fixes
 * where the image and the DDR staging buffers go.
 *
 * The staging sections must be inside the DDR window the platform exports
 * (`__nh_e6_ddr_window_start/end`): E6 installs that window as the TOR PMP
 * entry the DMA/AME bus masters use before the application's first transfer. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 4M }

SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM

  . = ALIGN(16);
  .bss (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*)
    *(COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > RAM

  .stacks 0x800e0000 (NOLOAD) : {
    __nh_stack_bottom = .; . += 64K; __nh_stack_top = .;
  } > RAM

  /* A[128,128] i8 = 16384 B, B[1024,128] i8 = 131072 B and the 1024 B accumu-
   * lator scratch window; C[128,1024] i32 = 524288 B is the output window.
   * Both fit the 1 MiB they are given, and the ASSERTs below keep it that way. */
  /* NOLOAD: e6_main() fills both windows before the kernel reads them, so the
   * image does not have to carry ~660 KiB of zeros. */
  .dma_source 0x80100000 (NOLOAD) : { KEEP(*(.dma_source)) } > RAM
  .dma_output 0x80200000 (NOLOAD) : { KEEP(*(.dma_output)) } > RAM
  /* The experimental kernel's aligned ordinary packs are DMA endpoints.
   * Its stack therefore belongs to the exported DDR/PMP window as well. */
  .e6_stack 0x802e0000 (NOLOAD) : {
    __e6_stack_bottom = .; . += 64K; __e6_stack_top = .;
  } > RAM
  . = ALIGN(4096);
  _end = .;

  __nh_e6_ddr_window_start = 0x80100000;
  __nh_e6_ddr_window_end = 0x80300000;

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
  ASSERT(_start == 0x80000000, "NH entry must be 0x80000000")
  ASSERT(e6_start == 0x80010000, "E6 entry must be 0x80010000")
  ASSERT(ADDR(.dma_source) == 0x80100000, "source staging moved")
  ASSERT(ADDR(.dma_output) == 0x80200000, "output staging moved")
  ASSERT(__nh_e6_ddr_window_start <= ADDR(.dma_source) &&
         ADDR(.dma_source) + SIZEOF(.dma_source) <= __nh_e6_ddr_window_end,
         "source staging outside the PMP DDR window")
  ASSERT(__nh_e6_ddr_window_start <= ADDR(.dma_output) &&
         ADDR(.dma_output) + SIZEOF(.dma_output) <= __nh_e6_ddr_window_end,
         "output staging outside the PMP DDR window")
  ASSERT(ADDR(.dma_source) + SIZEOF(.dma_source) <= ADDR(.dma_output),
         "source staging overlaps output")
  ASSERT(ADDR(.dma_output) + SIZEOF(.dma_output) <= __e6_stack_bottom,
         "output staging overlaps the E6 stack")
  ASSERT(__nh_e6_ddr_window_start <= __e6_stack_bottom &&
         __e6_stack_top <= __nh_e6_ddr_window_end,
         "E6 DMA stack outside the PMP DDR window")
  ASSERT(__bss_end <= __nh_stack_bottom, "image overlaps the NH stack")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds 4MiB")
}
