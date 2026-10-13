// DDR scratch needs an explicit alignment/nonaliasing contract. CPU copies
// cannot replace DMA into SPM on this target.

// RUN: not buddy-opt %S/Inputs/single-dot-64x64x64.ttir --triton-to-boscame 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOABI

// NOABI: error: SPM cannot be accessed by the CPU; the BOSCAME path requires 64 B aligned, non-overlapping DDR buffers and scratch
// NOABI-NOT: bosc_ame.dma.start

// RUN: buddy-opt %S/Inputs/unaligned-a.ttir --triton-to-boscame=dma-abi=args_aligned64_nonalias \
// RUN:   | FileCheck %s --check-prefix=UNALIGNED

// Unaligned original rows are packed into DDR scratch element by element ...
// UNALIGNED: scf.for
// UNALIGNED: memref.load %{{.*}}[%{{.*}}] : memref<64xi8>
// UNALIGNED: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x64xi8>
// ... and only the packed aligned row is sent to SPM.
// UNALIGNED: bosc_ame.dma.start
// UNALIGNED: bosc_ame.tile.mma
