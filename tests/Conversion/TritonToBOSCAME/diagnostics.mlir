// Every target region is analysed as a whole before anything is rewritten, so
// an unsupported shape, layout, mask or footprint is reported with a reason
// instead of being approximated silently.

// RUN: not buddy-opt %S/Inputs/diag-too-wide.ttir --triton-to-boscame=dma-abi=args_aligned64_nonalias 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIDE
// WIDE: error: cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: the AME accumulator tile is at most 64x64, but this dot has M = 128 and N = 64

// RUN: not buddy-opt %S/Inputs/diag-transposed-b.ttir --triton-to-boscame=dma-abi=args_aligned64_nonalias 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TRANSPOSED
// TRANSPOSED: error: cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: B must be stored as the physical N x K tile

// RUN: not buddy-opt %S/Inputs/diag-coupled-mask.ttir --triton-to-boscame=dma-abi=args_aligned64_nonalias 2>&1 \
// RUN:   | FileCheck %s --check-prefix=COUPLED
// COUPLED: error: cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: a mask term couples the row and the column index

// RUN: not buddy-opt %S/Inputs/diag-mask-coefficient.ttir --triton-to-boscame=dma-abi=args_aligned64_nonalias 2>&1 \
// RUN:   | FileCheck %s --check-prefix=COEFFICIENT
// COEFFICIENT: error: cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: a column condition has a coefficient other than 1

// RUN: not buddy-opt %S/Inputs/diag-dynamic-other.ttir --triton-to-boscame=dma-abi=args_aligned64_nonalias 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OTHER
// OTHER: error: cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: the A load: the `other` value is not a splat constant

// RUN: not buddy-opt %S/Inputs/diag-spm-capacity.ttir --triton-to-boscame=dma-abi=args_aligned64_nonalias 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CAPACITY
// CAPACITY: error: cannot lower this dot to the BOSCAME SPM/DMA/AME datapath: the accumulator window of 16384 bytes does not fit in SPM bank 0
