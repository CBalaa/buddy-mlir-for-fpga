// RUN: cc -std=c11 -Wall -Wextra -Werror -I %S/../../../examples/FPGA-BOSCAME/common/runtime %s -o %t.debug && %t.debug | FileCheck %s --check-prefix=CHECK
// RUN: cc -std=c11 -Wall -Wextra -Werror -DNDEBUG -I %S/../../../examples/FPGA-BOSCAME/common/runtime %s -o %t.ndebug && %t.ndebug | FileCheck %s --check-prefix=NDEBUG

// REQUIRES: host-c-compiler

// Host-side check of the SPM/DMA runtime helper, i.e. of the ABI the
// `lower-bosc-ame` pass emits calls to: the launch register layout,
// the token returned by `bosc_ame_dma_start` being the effective source
// address, the SRC write-back that is ordered before the single STATUS load in
// `bosc_ame_dma_wait`, the per-channel software token records, MMIO direction
// checks (STATUS is the only readable register), every defined wait
// failure path (bus_error, busy, bad channel, "not launched" token, token
// mismatch, missing launch and duplicate wait) and a simulated transfer whose
// staged bytes are compared against a
// DDR-direct copy (with a modelled bus error taking the error path and leaving
// the destination byte for byte untouched).
//
// The translation unit below is the very file the FPGA examples ship, so a
// board build and this regression test cannot drift apart.
//
// `bosc_ame_dma_wait` checks the status itself and has no result; the test
// replaces bosc_ame_dma_fail_hook with a setjmp/longjmp hook.  The wait checks
// do not use BOSC_AME_DMA_ASSERT, so the same failure suite runs with and
// without NDEBUG; only the `bosc_ame_dma_start` argument checks are gated on
// NDEBUG because a debug build deliberately asserts there.

// CHECK: bosc_ame_dma: all checks passed
// NDEBUG: bosc_ame_dma: all checks passed (NDEBUG: start argument checks included)

#include "../../../examples/FPGA-BOSCAME/common/runtime/test_bosc_ame_dma.c"
