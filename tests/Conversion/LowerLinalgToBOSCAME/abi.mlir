// RUN: split-file %s %t
// RUN: buddy-opt %t/missing.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true target=nr-fpga' | FileCheck %s --check-prefix=MISSING
// RUN: not buddy-opt %t/missing.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=unknown target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=CLI
// RUN: not buddy-opt %t/invalid-module.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=MODULE
// RUN: not buddy-opt %t/invalid-module.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %t/wrong-type.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=TYPE
// RUN: buddy-opt %t/inherited.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true target=nr-fpga' | FileCheck %s --check-prefix=INHERIT
// RUN: buddy-opt %t/aligned.mlir --lower-linalg-to-boscame | FileCheck %s --check-prefix=ALIGNED
// RUN: buddy-opt %t/missing.mlir --lower-linalg-to-boscame='dma-abi=args_aligned64' | FileCheck %s --check-prefix=ALIGNED

// A CLI promise must agree with the module's caller contract.  In particular,
// an incorrectly typed module attribute cannot silently turn into an absent
// attribute and then be replaced by the CLI value.
// A kernel without external transfers does not need a DMA caller promise.
// MISSING-NOT: bosc_ame.dma.abi
// MISSING: func.func @empty
// CLI: invalid dma-abi: unknown
// MODULE: invalid dma-abi: unknown
// CONFLICT: invalid or conflicting explicit DMA ABI contract
// TYPE: invalid or conflicting explicit DMA ABI contract
// INHERIT: bosc_ame.dma.abi = "args_aligned64_nonalias"
// INHERIT-NOT: bosc_ame.enable_spm_dma
// INHERIT: contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
// ALIGNED: bosc_ame.dma.abi = "args_aligned64"
// ALIGNED: contract.abi = {args_aligned_bytes = 64 : i64}
// ALIGNED-NOT: args_disjoint

//--- missing.mlir
module {
  func.func @empty() {
    return
  }
}

//--- invalid-module.mlir
module attributes {bosc_ame.dma.abi = "unknown"} {
  func.func @empty() {
    return
  }
}

//--- wrong-type.mlir
module attributes {bosc_ame.dma.abi = 64 : i64} {
  func.func @empty() {
    return
  }
}

//--- inherited.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @empty() {
    return
  }
}

//--- aligned.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64"} {
  func.func @empty() {
    return
  }
}
