OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 4M
}

SECTIONS
{
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM

  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM

  . = ALIGN(256);
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM

  .bss ALIGN(16) (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*)
    *(COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > RAM

  .stacks 0x800e0000 (NOLOAD) : {
    __nh_stack_bottom = .;
    . += 64K;
    __nh_stack_top = .;
    __e6_stack_bottom = .;
    . += 64K;
    __e6_stack_top = .;
  } > RAM

  .dma_source 0x80100000 : { KEEP(*(.dma_source)) } > RAM
  .dma_output 0x80102000 : { KEEP(*(.dma_output)) } > RAM
  .ptroot 0x80310000 : { KEEP(*(.ptroot)) } > RAM
  .ptl1 0x80311000 : { KEEP(*(.ptl1)) } > RAM
  .ptl0 0x80312000 : { KEEP(*(.ptl0)) } > RAM
  .ptspm0 0x80313000 : { KEEP(*(.ptspm0)) } > RAM
  .ptspm1 0x80314000 : { KEEP(*(.ptspm1)) } > RAM
  .ptrootb 0x80315000 : { KEEP(*(.ptrootb)) } > RAM
  .ptl1b 0x80316000 : { KEEP(*(.ptl1b)) } > RAM
  .ptl0b 0x80317000 : { KEEP(*(.ptl0b)) } > RAM
  .dma_asid_source0_a 0x80118080 : { KEEP(*(.dma_asid_source0_a)) } > RAM
  .dma_asid_source1_a 0x80119080 : { KEEP(*(.dma_asid_source1_a)) } > RAM
  .dma_asid_dst0_a 0x8011a080 : { KEEP(*(.dma_asid_dst0_a)) } > RAM
  .dma_asid_dst1_a 0x8011b080 : { KEEP(*(.dma_asid_dst1_a)) } > RAM
  .dma_asid_source0_b 0x8011c080 : { KEEP(*(.dma_asid_source0_b)) } > RAM
  .dma_asid_source1_b 0x8011d080 : { KEEP(*(.dma_asid_source1_b)) } > RAM
  .dma_asid_dst0_b 0x8011e080 : { KEEP(*(.dma_asid_dst0_b)) } > RAM
  .dma_asid_dst1_b 0x8011f080 : { KEEP(*(.dma_asid_dst1_b)) } > RAM
  .dma_asid_dst0_b_remap 0x80206080 : { KEEP(*(.dma_asid_dst0_b_remap)) } > RAM
  .dma_asid_dst1_b_remap 0x80207080 : { KEEP(*(.dma_asid_dst1_b_remap)) } > RAM
  .dma_super_source0 0x80201080 : { KEEP(*(.dma_super_source0)) } > RAM
  .dma_super_source1 0x80202080 : { KEEP(*(.dma_super_source1)) } > RAM
  .dma_super_dst0 0x80203080 : { KEEP(*(.dma_super_dst0)) } > RAM
  .dma_super_dst1 0x80204080 : { KEEP(*(.dma_super_dst1)) } > RAM
  _end = .;

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }

  ASSERT(_start == 0x80000000, "NH entry must be 0x80000000")
  ASSERT(e6_start == 0x80010000, "E6 entry must be 0x80010000")
  ASSERT((__bss_start & 7) == 0, "DMA/SPM BSS must be 8-byte aligned")
  ASSERT(ADDR(.dma_source) == 0x80100000, "DMA source moved")
  ASSERT(ADDR(.ptroot) == 0x80310000, "root page table moved")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "test image exceeds preload RAM")
}
