// RUN: split-file %s %t
// RUN: buddy-opt %t/saved.mlir --linalg-to-boscame | FileCheck %s --check-prefix=SAVED
// RUN: not buddy-opt %t/saved.mlir --linalg-to-boscame='enable-spm-dma=false' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %t/saved.mlir --linalg-to-boscame='enable-ame=false' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %t/saved.mlir --linalg-to-boscame='bm=64' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %t/saved.mlir --linalg-to-boscame='bn=64' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %t/saved.mlir --linalg-to-boscame='bk=128' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %t/saved.mlir --linalg-to-boscame='target=' 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: not buddy-opt %t/invalid.mlir --linalg-to-boscame 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %t/lost-marker.mlir --linalg-to-boscame 2>&1 | FileCheck %s --check-prefix=MARKER
// SAVED: bosc_ame.bk = 64 : i64
// SAVED: bosc_ame.bm = 16 : i64
// SAVED: bosc_ame.bn = 32 : i64
// CONFLICT: invalid or conflicting BOSCAME module configuration
// TARGET: invalid or conflicting BOSCAME module target contract
// MARKER: BOSCAME AME marker count changed across explicit bufferization

//--- saved.mlir
module attributes {bosc_ame.enable_spm_dma = true, bosc_ame.enable_ame = true, bosc_ame.bm = 16 : i64, bosc_ame.bn = 32 : i64, bosc_ame.bk = 64 : i64, bosc_ame.target = "nr-fpga", bosc_ame.dma.abi = "args_aligned64_nonalias"} {}

//--- invalid.mlir
module attributes {bosc_ame.enable_spm_dma = "true"} {}

//--- lost-marker.mlir
module attributes {bosc_ame.expected_ame_matmuls = 1 : i64} {}
