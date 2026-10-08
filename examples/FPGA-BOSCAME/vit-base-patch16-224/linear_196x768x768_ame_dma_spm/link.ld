OUTPUT_ARCH("riscv")
ENTRY(_start)

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
    __e6_stack_bottom = .; . += 64K; __e6_stack_top = .;
  } > RAM

  .dma_source 0x80100000 : { KEEP(*(.dma_source)) } > RAM
  .dma_output 0x80200000 : { KEEP(*(.dma_output)) } > RAM
  . = ALIGN(4096);
  _end = .;

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
  ASSERT(_start == 0x80000000, "NH entry must be 0x80000000")
  ASSERT(e6_start == 0x80010000, "E6 entry must be 0x80010000")
  ASSERT(ADDR(.dma_source) == 0x80100000, "source staging moved")
  ASSERT(ADDR(.dma_output) == 0x80200000, "output staging moved")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds 4MiB")
}
