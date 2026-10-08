  .option norvc

  .equ RA_BASE,       0x50000000
  .equ E6_MAILBOX,    0x50000160
  .equ E6_DONE,       0x50000120
  .equ NH_RESULT,     0x50000140
  .equ NH_UART_BASE,  0x310b0000

  .section .text.nh_start, "ax", @progbits
  .globl _start
  .type _start, @function
_start:
  la sp, __nh_stack_top
  la t0, nh_trap
  csrw mtvec, t0
  call nh_uart_init
  la a0, msg_start
  call nh_uart_puts

  li t0, E6_DONE
  sw zero, 0(t0)
  # E6 publishes bounded progress markers through the patch6 mailbox.  Clear
  # any stale value left by a previous image before releasing E6.
  li t0, E6_MAILBOX
  sw zero, 0(t0)
  li t0, NH_RESULT
  sw zero, 0(t0)
  fence iorw, iorw

  # NH programs and releases E6 through RA SYSCTRL.
  li t0, RA_BASE
  la t1, e6_start
  sw t1, 0x100(t0)
  sw zero, 0x060(t0)
  li t2, 1
  sw t2, 0x050(t0)
  sw t2, 0x060(t0)
  fence iorw, iorw

  # DMA/SPM are controlled by E6. NH owns the bounded wait and final verdict.
  li t0, E6_DONE
  li t3, 100000000
  li t6, 0
  li t1, E6_MAILBOX
.Lnh_wait_e6:
  fence r, r
  lw t4, 0(t0)
  lw a1, 0(t1)
  li t5, 2
  beq t4, t5, .Lnh_verdict
  li t5, 0xed
  beq t4, t5, .Lnh_verdict
  li t5, 0xee
  beq t4, t5, .Lnh_verdict
  beqz a1, .Lnh_wait_no_stage
  beq a1, t6, .Lnh_wait_no_stage
  mv t6, a1
  la a0, msg_e6_progress
  call nh_uart_puts
  mv a0, a1
  call nh_uart_hex32
  la a0, msg_crlf
  call nh_uart_puts
  li t0, E6_DONE
  li t1, E6_MAILBOX
.Lnh_wait_no_stage:
  addi t3, t3, -1
  bnez t3, .Lnh_wait_e6
  la a0, msg_timeout
  li t5, 0xee
  j .Lnh_publish
.Lnh_verdict:
  li t5, 2
  beq t4, t5, .Lnh_pass
  li t5, 0xed
  beq t4, t5, .Lnh_trap
  la a0, msg_fail
  li t5, 0xee
  j .Lnh_publish
.Lnh_pass:
  la a0, msg_pass
  li t5, 0xff
  j .Lnh_publish
.Lnh_trap:
  la a0, msg_e6_trap
  li t5, 0xee
.Lnh_publish:
  li t0, NH_RESULT
  sw t5, 0(t0)
  fence iorw, iorw
  call nh_uart_puts
  la a0, msg_stats
  call nh_uart_puts
  la t0, e6_diag_status
  lw a0, 0(t0)
  call nh_uart_hex32
  la a0, msg_cycles
  call nh_uart_puts
  la t0, e6_diag_cycles
  ld a0, 0(t0)
  call nh_uart_hex64
  la a0, msg_errors
  call nh_uart_puts
  la t0, e6_diag_errors
  lw a0, 0(t0)
  call nh_uart_hex32
  la a0, msg_crlf
  call nh_uart_puts
  # Keep the VCS completion protocol in the same ELF as FPGA UART reporting.
  li t0, E6_MAILBOX
  sw t5, 0(t0)
  fence iorw, iorw
.Lnh_halt:
  wfi
  j .Lnh_halt
  .size _start, . - _start

  .align 2
nh_trap:
  li t0, NH_RESULT
  li t1, 0xee
  sw t1, 0(t0)
  fence iorw, iorw
  la a0, msg_nh_trap
  call nh_uart_puts
  li t0, E6_MAILBOX
  sw t1, 0(t0)
  fence iorw, iorw
  j .Lnh_halt

  # 5.5296 MHz / (16 * 3) = 115200 baud exactly.
  .section .text.nh_uart, "ax", @progbits
  .type nh_uart_init, @function
nh_uart_init:
  li t0, NH_UART_BASE
  sw zero, 0x04(t0)
  li t1, 0x83
  sw t1, 0x0c(t0)
1:
  lw t1, 0x7c(t0)
  andi t1, t1, 1
  bnez t1, 1b
  sw zero, 0x04(t0)
  li t1, 3
  sw t1, 0x00(t0)
  li t1, 3
  sw t1, 0x0c(t0)
  li t1, 1
  sw t1, 0x08(t0)
  li t1, 3
  sw t1, 0x10(t0)
  fence iorw, iorw
  ret
  .size nh_uart_init, . - nh_uart_init

  .type nh_uart_putc, @function
nh_uart_putc:
  li t0, NH_UART_BASE
1:
  lw t1, 0x14(t0)
  andi t1, t1, 0x20
  beqz t1, 1b
  sw a0, 0x20(t0)
  ret
  .size nh_uart_putc, . - nh_uart_putc

  .type nh_uart_puts, @function
nh_uart_puts:
  addi sp, sp, -16
  sd ra, 8(sp)
  sd s0, 0(sp)
  mv s0, a0
1:
  lbu a0, 0(s0)
  beqz a0, 2f
  call nh_uart_putc
  addi s0, s0, 1
  j 1b
2:
  ld s0, 0(sp)
  ld ra, 8(sp)
  addi sp, sp, 16
  ret
  .size nh_uart_puts, . - nh_uart_puts

  # Print a progress marker as eight hexadecimal digits.  This routine only
  # uses caller-saved temporaries t0..t2, leaving the NH polling state in t3,
  # t4, and t6 intact across the diagnostic call.
  .type nh_uart_hex32, @function
nh_uart_hex32:
  addi sp, sp, -16
  sd ra, 8(sp)
  sd s0, 0(sp)
  mv s0, a0
  li t2, 28
1:
  srl t1, s0, t2
  andi t1, t1, 15
  li t0, 10
  bltu t1, t0, 2f
  addi t1, t1, 87
  j 3f
2:
  addi t1, t1, 48
3:
  mv a0, t1
  call nh_uart_putc
  addi t2, t2, -4
  bgez t2, 1b
  ld s0, 0(sp)
  ld ra, 8(sp)
  addi sp, sp, 16
  ret
  .size nh_uart_hex32, . - nh_uart_hex32

  .type nh_uart_hex64, @function
nh_uart_hex64:
  addi sp, sp, -16
  sd ra, 8(sp)
  sd s0, 0(sp)
  mv s0, a0
  srli a0, s0, 32
  call nh_uart_hex32
  mv a0, s0
  call nh_uart_hex32
  ld s0, 0(sp)
  ld ra, 8(sp)
  addi sp, sp, 16
  ret
  .size nh_uart_hex64, . - nh_uart_hex64

  .section .text.e6_start, "ax", @progbits
  .globl e6_start
  .type e6_start, @function
e6_start:
  la sp, __e6_stack_top
  # Enable FP/vector/AME state.  AME MS[26:25] are part of the patch6
  # execution contract; omitting 0x06000000 makes the first AME instruction
  # trap even though the core itself has started.
  li t0, 0x0601e600
  csrs mstatus, t0
  la t0, e6_trap
  csrw mtvec, t0

  la t0, __bss_start
  la t1, __bss_end
.Le6_bss_loop:
  bgeu t0, t1, .Le6_bss_done
  sd zero, 0(t0)
  addi t0, t0, 8
  j .Le6_bss_loop
.Le6_bss_done:
  fence rw, rw
  call e6_main

  li t2, 2
  beqz a0, .Le6_publish
  li t2, 0xee
.Le6_publish:
  li t0, E6_DONE
  sw t2, 0(t0)
  fence iorw, iorw

  li t0, NH_RESULT
.Le6_wait_nh:
  fence r, r
  lw t1, 0(t0)
  beqz t1, .Le6_wait_nh
  li t0, E6_MAILBOX
  sw t1, 0(t0)
  fence iorw, iorw
.Le6_halt:
  wfi
  j .Le6_halt
  .size e6_start, . - e6_start

  .align 2
e6_trap:
  li t0, E6_DONE
  # Let NH emit a distinct UART TRAP verdict before publishing 0xee to VCS.
  li t1, 0xed
  sw t1, 0(t0)
  fence iorw, iorw
  j .Le6_halt

  .section .rodata.messages, "a", @progbits
msg_start:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] start\r\n"
msg_pass:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] PASS\r\n"
msg_fail:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] FAIL\r\n"
msg_timeout:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] TIMEOUT\r\n"
msg_nh_trap:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] NH TRAP\r\n"
msg_e6_trap:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] E6 TRAP\r\n"
msg_e6_progress:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] E6 progress=0x"
msg_crlf:
  .asciz "\r\n"
msg_stats:
  .asciz "[5.5296MHz][NH->E6->DMA/SPM] stats status=0x"
msg_cycles:
  .asciz " cycles=0x"
msg_errors:
  .asciz " errors=0x"
