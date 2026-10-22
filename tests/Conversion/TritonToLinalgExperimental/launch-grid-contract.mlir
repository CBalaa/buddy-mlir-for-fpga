// RUN: %PYTHON %S/launch-grid-frontend-test.py
// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=kernel:2,3,4;preserved:8,9,10' -o %t
// RUN: FileCheck %s --check-prefix=VALID < %t
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=missing:1,2,3' 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=not_func:1,2,3' 2>&1 | FileCheck %s --check-prefix=NOT-FUNC
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=kernel:2,3' 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=kernel:2,0,4' 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=kernel:2,2147483648,4' 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=preserved:1,2,3' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias launch-grid=kernel:1,2,3;kernel:1,2,3' 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not buddy-opt %s --triton-to-linalg-experimental='launch-grid=kernel:2,3,4' 2>&1 | FileCheck %s --check-prefix=REQUIRES-BOSCAME

// VALID: func.func @kernel
// VALID-SAME: gpu.known_grid_size = array<i32: 2, 3, 4>
// VALID-SAME: test.custom = "retained"
// VALID: func.func @preserved
// VALID-SAME: gpu.known_grid_size = array<i32: 8, 9, 10>
// VALID-SAME: test.custom = "also_retained"
// UNKNOWN: invalid launch-grid entry
// NOT-FUNC: invalid launch-grid entry
// MALFORMED: invalid launch-grid entry
// CONFLICT: launch-grid conflicts with existing gpu.known_grid_size on @preserved
// DUPLICATE: duplicate launch-grid contract for @kernel
// REQUIRES-BOSCAME: launch-grid requires boscame=true

module {
  tt.func public @kernel() attributes {test.custom = "retained"} {
    tt.return
  }
  tt.func public @preserved() attributes {gpu.known_grid_size = array<i32: 8, 9, 10>, test.custom = "also_retained"} {
    tt.return
  }
  func.func private @not_func() {
    return
  }
}
