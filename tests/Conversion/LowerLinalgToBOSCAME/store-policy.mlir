// The independent buffer-form pass records the external-store policy in the
// module contract even when the module has no Linalg consumers yet.
// RUN: buddy-opt %s --linalg-to-boscame | FileCheck %s --check-prefix=DEFAULT
// RUN: not buddy-opt %s --linalg-to-boscame='store-policy=unknown' 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not buddy-opt %s --linalg-to-boscame='store-policy=spm-dma' 2>&1 | FileCheck %s --check-prefix=SPM-ABI
// RUN: buddy-opt %s --linalg-to-boscame='store-policy=spm-dma enable-spm-dma=true dma-abi=args_aligned64_nonalias' | FileCheck %s --check-prefix=SPM
// RUN: buddy-opt %s --linalg-to-boscame-store=spm-dma --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias' | FileCheck %s --check-prefix=ALIAS

module {
}

// DEFAULT: bosc_ame.store_policy = "direct-ddr"
// INVALID: invalid store-policy: unknown
// SPM-ABI: store-policy=spm-dma requires enable-spm-dma=true
// SPM: bosc_ame.store_policy = "spm-dma"
// ALIAS: bosc_ame.store_policy = "spm-dma"
